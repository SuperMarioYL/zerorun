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


if __name__ == "__main__":
    unittest.main()
