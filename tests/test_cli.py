"""CLI surface tests — --version flag and honest override diagnostics (v0.5.0).

Regression tests for two shipped v0.4.0 defects (bug_findings BUG-1 / BUG-2):

* ``zerorun --version`` errored with "No such option: --version" (rc=2) — the
  CLI could not report its own version even though v0.4.0's
  fix-init-version-drift locked ``zerorun.__version__`` against the VERSION
  file.
* An invalid ``--backend`` / ``--quant`` override produced a FALSE "does not
  fit this GPU" diagnostic: ``select()`` returns ``None`` when the filter
  matches nothing, so the CLI could not tell "unknown value" apart from
  "real variant, infeasible GPU". The honest shortfall path ("does not fit"
  + VRAM-short reasons) is only correct for a REAL variant — these tests pin
  all three branches.

All invocations drive the picker through ``--gpu`` simulation, so no NVIDIA
hardware (and no pynvml import) is needed.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("ZERORUN_REGISTRY_DIR", str(ROOT / "registry"))

import typer  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

import zerorun  # noqa: E402
from zerorun.cli import app  # noqa: E402

runner = CliRunner()


class TestVersionFlag(unittest.TestCase):
    def test_version_flag_prints_version_and_exits_zero(self) -> None:
        result = runner.invoke(app, ["--version"])
        self.assertEqual(
            result.exit_code, 0,
            msg=f"--version must exit 0; got {result.exit_code}: {result.output}",
        )
        self.assertIn("zerorun", result.output.lower())
        self.assertIn(
            zerorun.__version__, result.output,
            msg="--version output must carry zerorun.__version__",
        )


class TestOverrideDiagnostics(unittest.TestCase):
    def test_unknown_backend_exits_2_and_names_valid_backends(self) -> None:
        result = runner.invoke(
            app, ["qwen-3.6", "--gpu", "24:8.9", "--backend", "exllama"]
        )
        self.assertEqual(
            result.exit_code, 2,
            msg=(
                "an unknown backend is a usage error (exit 2), not a GPU-fit "
                f"error (exit 1); got {result.exit_code}: {result.output}"
            ),
        )
        self.assertIn("unknown backend", result.output.lower())
        self.assertIn("vllm", result.output)
        self.assertIn("llama.cpp", result.output)
        self.assertNotIn(
            "does not fit this GPU", result.output,
            msg="a nonexistent backend must not be reported as a GPU shortfall",
        )

    def test_unknown_quant_exits_2_and_lists_available_formats(self) -> None:
        result = runner.invoke(
            app, ["qwen-3.6", "--gpu", "24:8.9", "--quant", "Q2_K"]
        )
        self.assertEqual(
            result.exit_code, 2,
            msg=(
                "an unknown quant is a usage error (exit 2); got "
                f"{result.exit_code}: {result.output}"
            ),
        )
        self.assertIn("unknown quant", result.output.lower())
        # The entry's real formats must be surfaced so the user can self-correct.
        for fmt in ("FP8", "Q4_K_M", "Q8_0"):
            self.assertIn(fmt, result.output)
        self.assertNotIn(
            "does not fit this GPU", result.output,
            msg="a nonexistent quant must not be reported as a GPU shortfall",
        )

    def test_real_but_infeasible_override_keeps_honest_shortfall(self) -> None:
        # NVFP4 IS a real kimi-k3 variant (min_vram_gb 56, compute_cap_min 9.0);
        # a 24 GB / cc 8.9 card cannot run it, so the honest GPU-fit path must
        # stay: exit 1 + "does not fit this GPU" + the VRAM shortfall.
        result = runner.invoke(
            app, ["kimi-k3", "--gpu", "24:8.9", "--quant", "NVFP4"]
        )
        self.assertEqual(
            result.exit_code, 1,
            msg=f"a real infeasible override exits 1; got {result.exit_code}",
        )
        self.assertIn("does not fit this GPU", result.output)
        self.assertIn("VRAM short", result.output)
        # ...and it must describe the REQUESTED variant, not another one: NVFP4's
        # floor is 56 GB (Q4_K_M's is 36), and its arch floor is Hopper.
        self.assertIn("need 56 GB", result.output)
        self.assertIn("compute-cap 8.9 < required 9.0", result.output)


class TestOverrideShortfallDescribesRequestedVariant(unittest.TestCase):
    """v0.6.0 fix-override-shortfall-diagnostics.

    ``_resolve_pick`` used to take ``pick(...)[0]`` for its diagnostic, and
    ``pick`` ranks feasible variants first — so whenever the entry had ANY
    feasible variant, the "reason" printed under "does not fit this GPU" was a
    feasible variant's score breakdown and the requested variant's shortfall was
    never shown. The v0.5.0 test above misses this because kimi-k3 on 24:8.9 has
    no feasible variant at all; these use deepseek-v4 on 32:8.0, where Q4_K_M
    (32 GB) fits while Q8_0 (60 GB) and NVFP4/vllm (48 GB, Hopper) do not.
    """

    def test_infeasible_quant_override_reports_its_own_vram_floor(self) -> None:
        result = runner.invoke(
            app, ["deepseek-v4", "--gpu", "32:8.0", "--quant", "Q8_0"]
        )
        self.assertEqual(result.exit_code, 1, msg=result.output)
        self.assertIn("does not fit this GPU", result.output)
        self.assertIn(
            "VRAM short: have 32.0 GB, need 60 GB (-28.0 GB)", result.output,
            msg="the requested Q8_0 floor must be reported",
        )
        self.assertNotIn(
            "accuracy", result.output,
            msg=(
                "a feasible variant's score breakdown must never be printed as "
                f"the shortfall: {result.output}"
            ),
        )

    def test_infeasible_backend_override_reports_that_variants_floors(self) -> None:
        # --backend vllm on 32 GB / cc 8.0: the vllm variant (NVFP4) is short on
        # both axes, while the llama.cpp Q4_K_M variant fits — the exact shape
        # that produced a bogus score line before v0.6.0.
        result = runner.invoke(
            app, ["deepseek-v4", "--gpu", "32:8.0", "--backend", "vllm"]
        )
        self.assertEqual(result.exit_code, 1, msg=result.output)
        self.assertIn("need 48 GB", result.output)
        self.assertIn("compute-cap 8.0 < required 9.0", result.output)
        self.assertNotIn("accuracy", result.output)

    def test_no_override_still_reports_nearest_infeasible_variant(self) -> None:
        # Regression guard for the path the change must not disturb: with no
        # override and no feasible variant, the closest miss is reported.
        result = runner.invoke(app, ["deepseek-v4", "--no-probe"])
        self.assertEqual(result.exit_code, 1, msg=result.output)
        self.assertIn("no NVIDIA GPU detected", result.output)


class TestOverridePairValidation(unittest.TestCase):
    """v0.6.0 fix-override-pair-validation.

    Each override value was valid on its own, so a pair that no variant has fell
    through to the GPU-fit branch and was reported as a hardware shortfall even
    when every variant of the model fits the probed card.
    """

    def test_contradictory_pair_exits_2_and_names_real_pairs(self) -> None:
        result = runner.invoke(
            app,
            ["deepseek-v4", "--gpu", "80:9.0", "--backend", "vllm",
             "--quant", "Q4_K_M"],
        )
        self.assertEqual(result.exit_code, 2, msg=result.output)
        self.assertIn("no such variant", result.output.lower())
        for pair in ("llama.cpp/Q4_K_M", "llama.cpp/Q8_0", "vllm/NVFP4"):
            self.assertIn(pair, result.output)
        self.assertNotIn("does not fit this GPU", result.output)

    def test_mirror_pair_also_exits_2(self) -> None:
        result = runner.invoke(
            app,
            ["deepseek-v4", "--gpu", "80:9.0", "--backend", "llama.cpp",
             "--quant", "NVFP4"],
        )
        self.assertEqual(result.exit_code, 2, msg=result.output)
        self.assertIn("no such variant", result.output.lower())

    def test_valid_pair_still_launches_the_plan(self) -> None:
        result = runner.invoke(
            app,
            ["deepseek-v4", "--gpu", "80:9.0", "--dry-run",
             "--backend", "vllm", "--quant", "NVFP4"],
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("vllm serve deepseek-ai/DeepSeek-V4-flash-dspark", result.output)


class TestNumericFlagValidation(unittest.TestCase):
    """v0.6.0 fix-cli-numeric-flag-validation."""

    def test_out_of_range_port_exits_2(self) -> None:
        for bad in ("0", "-5", "99999"):
            with self.subTest(port=bad):
                result = runner.invoke(
                    app,
                    ["qwen-3.6", "--gpu", "24:8.9", "--dry-run", "--port", bad],
                )
                self.assertEqual(result.exit_code, 2, msg=result.output)
                self.assertIn("1 and 65535", result.output)
                self.assertNotIn("endpoint:", result.output)

    def test_valid_port_still_works(self) -> None:
        result = runner.invoke(
            app, ["qwen-3.6", "--gpu", "24:8.9", "--dry-run", "--port", "9000"]
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("http://localhost:9000/v1", result.output)

    def test_non_finite_gpu_spec_exits_2(self) -> None:
        for bad in ("80:nan", "nan:9.0", "inf:9.0", "-inf:9.0"):
            with self.subTest(gpu=bad):
                result = runner.invoke(app, ["probe", "--gpu", bad])
                self.assertEqual(result.exit_code, 2, msg=result.output)
                self.assertNotIn("Arch:", result.output)

    def test_finite_gpu_spec_still_works(self) -> None:
        result = runner.invoke(app, ["probe", "--gpu", "80:9.0"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("hopper", result.output)


if __name__ == "__main__":
    unittest.main()
