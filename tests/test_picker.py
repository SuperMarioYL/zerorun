"""Unit tests for the m2 picker module.

The picker is the m2 core: a pure ``(GPUProbe, ModelEntry) -> ranked Picks``
function with no I/O, so it unit-tests without any hardware. These tests cover
the documented perf-vs-accuracy heuristic, the feasibility hard-gates, ranking
order (feasible-first by score, then infeasible by closeness), the override
selectors, and behavior against the real shipped registry (all 4 hero models).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make src/ importable when tests run without an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerorun.picker import (  # noqa: E402
    Pick,
    accuracy_score,
    backend_throughput,
    best_pick,
    best_pick_per_model,
    feasibility_reasons,
    is_feasible,
    pick,
    score_quant,
    select,
    vram_safety,
)
from zerorun.probe import GPUProbe  # noqa: E402
from zerorun.registry.schema import (  # noqa: E402
    ModelEntry,
    QuantConfig,
    load_model,
    load_registry_dir,
)


def _q(fmt, backend, vram, cc=0.0, args=()):
    """Build a QuantConfig quickly for a test."""
    return QuantConfig(
        format=fmt, backend=backend, min_vram_gb=vram,
        compute_cap_min=cc, args=tuple(args),
    )


def _entry(model="m", hf_id="org/m", quants=(), key="m", template="m-tmpl"):
    """Build a ModelEntry quickly for a test."""
    return ModelEntry(
        model=model, hf_id=hf_id, quants=tuple(quants),
        chat_template_ref=template, registry_key=key,
    )


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

class TestAccuracyScore(unittest.TestCase):
    def test_known_formats_ordered(self):
        # Preserved-precision hierarchy from the table.
        self.assertGreater(accuracy_score("F16"), accuracy_score("Q8_0"))
        self.assertGreater(accuracy_score("Q8_0"), accuracy_score("Q4_K_M"))
        self.assertGreater(accuracy_score("Q4_K_M"), accuracy_score("NVFP4"))
        self.assertGreater(accuracy_score("NVFP4"), accuracy_score("Q2_K"))

    def test_bf16_alias(self):
        self.assertEqual(accuracy_score("BF16"), 1.0)

    def test_unknown_format_uses_default(self):
        # A community-PR'd new format must not silently top the ranking.
        self.assertEqual(accuracy_score("ZZ_CUSTOM"), 0.72)
        self.assertLess(accuracy_score("ZZ_CUSTOM"), accuracy_score("Q8_0"))

    def test_range_bounded(self):
        for fmt in ("F16", "Q8_0", "Q4_K_M", "NVFP4", "Ternary", "Q2_K"):
            s = accuracy_score(fmt)
            self.assertGreater(s, 0.0)
            self.assertLessEqual(s, 1.0)


class TestBackendThroughput(unittest.TestCase):
    def test_vllm_best_on_hopper(self):
        self.assertEqual(backend_throughput("vllm", "hopper"), 0.15)
        self.assertEqual(backend_throughput("vllm", "blackwell"), 0.15)

    def test_vllm_mid_on_ampere(self):
        self.assertEqual(backend_throughput("vllm", "ampere"), 0.08)
        self.assertEqual(backend_throughput("vllm", "ada"), 0.08)

    def test_vllm_low_on_turing(self):
        self.assertEqual(backend_throughput("vllm", "turing"), 0.04)
        self.assertEqual(backend_throughput("vllm", "unknown"), 0.04)

    def test_llama_cpp_constant(self):
        for fam in ("hopper", "ampere", "ada", "turing", "unknown"):
            self.assertEqual(backend_throughput("llama.cpp", fam), 0.05)

    def test_unknown_backend_zero(self):
        self.assertEqual(backend_throughput("exl2", "hopper"), 0.0)

    def test_bonus_never_dominates_accuracy(self):
        # Accuracy is the primary axis: the max backend bonus (0.15) must be
        # strictly smaller than any *non-adjacent* accuracy gap so a backend
        # can only close near-ties, never flip a clear accuracy ordering.
        # The smallest adjacent gap (Q8_0 0.95 -> Q4_K_M 0.80 = 0.15) equals
        # the max bonus by design (a near-tie the bonus may resolve); use a
        # clearly-larger gap here.
        max_bonus = max(
            backend_throughput("vllm", "hopper"),
            backend_throughput("llama.cpp", "hopper"),
        )
        # Q8_0 -> NVFP4 = 0.95 - 0.74 = 0.21 > 0.15 max bonus.
        self.assertLess(max_bonus,
                        accuracy_score("Q8_0") - accuracy_score("NVFP4"))


class TestVramSafety(unittest.TestCase):
    def test_exact_fit_zero(self):
        self.assertEqual(vram_safety(48, 48), 0.0)

    def test_headroom_grows_then_caps(self):
        # 1.5x headroom -> 0.04 * 0.5 = 0.02
        self.assertAlmostEqual(vram_safety(48, 32), 0.02, places=4)
        # 3x headroom -> 0.04 * 2.0 = 0.08 (cap)
        self.assertEqual(vram_safety(96, 32), 0.08)
        # cap holds even with huge headroom
        self.assertEqual(vram_safety(1000, 32), 0.08)

    def test_zero_floor_zero(self):
        self.assertEqual(vram_safety(80, 0), 0.0)

    def test_under_floor_zero(self):
        self.assertEqual(vram_safety(16, 32), 0.0)


# ---------------------------------------------------------------------------
# Feasibility hard-gate
# ---------------------------------------------------------------------------

class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.h100 = GPUProbe(
            vram_gb=80, compute_cap=9.0, arch_family="hopper",
            name="H100", source="simulated",
        )
        self.rtx4090 = GPUProbe(
            vram_gb=24, compute_cap=8.9, arch_family="ada",
            name="4090", source="simulated",
        )
        self.no_gpu = GPUProbe()

    def test_feasible_when_fits(self):
        q = _q("Q4_K_M", "llama.cpp", 24, cc=7.5)
        self.assertTrue(is_feasible(self.rtx4090, q))

    def test_vram_too_small_infeasible(self):
        q = _q("Q8_0", "llama.cpp", 60, cc=7.5)
        self.assertFalse(is_feasible(self.rtx4090, q))

    def test_compute_cap_below_floor_infeasible(self):
        # vLLM NVFP4 needs Hopper (9.0); a 4090 is ada (8.9).
        q = _q("NVFP4", "vllm", 24, cc=9.0)
        self.assertFalse(is_feasible(self.rtx4090, q))

    def test_zero_compute_cap_floor_passes_arch(self):
        # A 0 floor means "any arch" and always passes the arch gate.
        q = _q("Q4", "llama.cpp", 24, cc=0.0)
        self.assertTrue(is_feasible(self.rtx4090, q))

    def test_no_gpu_infeasible_for_everything(self):
        q = _q("Q4_K_M", "llama.cpp", 1, cc=0.0)
        self.assertFalse(is_feasible(self.no_gpu, q))

    def test_h100_meets_nvfp4_floor(self):
        q = _q("NVFP4", "vllm", 48, cc=9.0)
        self.assertTrue(is_feasible(self.h100, q))


class TestFeasibilityReasons(unittest.TestCase):
    def test_feasible_empty_reasons(self):
        probe = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                         name="H100", source="simulated")
        q = _q("NVFP4", "vllm", 48, cc=9.0)
        self.assertEqual(feasibility_reasons(probe, q), [])

    def test_vram_shortfall_quantified(self):
        probe = GPUProbe(vram_gb=24, compute_cap=9.0, arch_family="hopper",
                         name="H100", source="simulated")
        q = _q("NVFP4", "vllm", 48, cc=9.0)
        reasons = feasibility_reasons(probe, q)
        self.assertEqual(len(reasons), 1)
        self.assertIn("VRAM short", reasons[0])
        self.assertIn("48", reasons[0])
        self.assertIn("24", reasons[0])

    def test_arch_shortfall_quantified(self):
        probe = GPUProbe(vram_gb=80, compute_cap=8.9, arch_family="ada",
                         name="4090", source="simulated")
        q = _q("NVFP4", "vllm", 48, cc=9.0)
        reasons = feasibility_reasons(probe, q)
        self.assertEqual(len(reasons), 1)
        self.assertIn("arch short", reasons[0])
        self.assertIn("9.0", reasons[0])

    def test_no_gpu_reason(self):
        probe = GPUProbe()
        q = _q("Q4_K_M", "llama.cpp", 1)
        reasons = feasibility_reasons(probe, q)
        self.assertEqual(len(reasons), 1)
        self.assertIn("no NVIDIA GPU", reasons[0])


# ---------------------------------------------------------------------------
# score_quant
# ---------------------------------------------------------------------------

class TestScoreQuant(unittest.TestCase):
    def test_feasible_score_is_accuracy_plus_bonuses(self):
        probe = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                         name="H100", source="simulated")
        q = _q("Q8_0", "llama.cpp", 60, cc=7.5)
        score, reasons = score_quant(probe, q)
        expected = accuracy_score("Q8_0") + backend_throughput(
            "llama.cpp", "hopper"
        ) + vram_safety(80, 60)
        self.assertAlmostEqual(score, expected, places=6)
        self.assertTrue(any("score:" in r for r in reasons))

    def test_infeasible_score_is_negative_closeness(self):
        probe = GPUProbe(vram_gb=24, compute_cap=9.0, arch_family="hopper",
                         name="H100", source="simulated")
        q = _q("Q8_0", "llama.cpp", 60, cc=7.5)
        score, reasons = score_quant(probe, q)
        self.assertLess(score, 0)  # infeasible -> negative
        self.assertTrue(any("infeasible" in r for r in reasons))

    def test_feasible_score_higher_than_infeasible(self):
        # A feasible variant always outscores an infeasible one (positive vs neg).
        probe = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                         name="H100", source="simulated")
        feas = _q("Q4_K_M", "llama.cpp", 32, cc=7.5)
        infeas = _q("NVFP4", "vllm", 200, cc=9.0)  # huge VRAM floor
        self.assertGreater(score_quant(probe, feas)[0],
                           score_quant(probe, infeas)[0])


# ---------------------------------------------------------------------------
# pick / best_pick ranking
# ---------------------------------------------------------------------------

class TestPickRanking(unittest.TestCase):
    def setUp(self):
        self.h100 = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                             name="H100", source="simulated")
        self.rtx4090 = GPUProbe(vram_gb=24, compute_cap=8.9, arch_family="ada",
                                name="4090", source="simulated")
        # A model with a feasible Q4_K_M, a feasible Q8_0 (needs headroom),
        # and an infeasible NVFP4/vllm (Hopper-only, large floor).
        self.entry = _entry(quants=[
            _q("NVFP4", "vllm", 48, cc=9.0),    # H100 feasible; 4090 infeasible (arch)
            _q("Q4_K_M", "llama.cpp", 32, cc=7.5),  # infeasible on 4090 (VRAM)
            _q("Q8_0", "llama.cpp", 60, cc=7.5),     # H100 feasible
        ])

    def test_feasible_first_then_infeasible(self):
        ranked = pick(self.h100, self.entry)
        feas_flags = [p.feasible for p in ranked]
        # feasible (True) entries come before infeasible (False).
        self.assertEqual(feas_flags, sorted(feas_flags, reverse=True))

    def test_feasible_sorted_by_score_desc(self):
        ranked = pick(self.h100, self.entry)
        feasible = [p for p in ranked if p.feasible]
        scores = [p.score for p in feasible]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_infeasible_sorted_by_closeness_desc(self):
        # On a 4090: NVFP4 (arch+VRAM short), Q4_K_M (VRAM short 8GB),
        # Q8_0 (VRAM short 36gb). Q4_K_M is closest to feasible.
        ranked = pick(self.rtx4090, self.entry)
        infeasible = [p for p in ranked if not p.feasible]
        self.assertEqual(len(infeasible), 3)
        # Closest-to-feasible first (least negative).
        closeness = [p.score for p in infeasible]
        self.assertEqual(closeness, sorted(closeness, reverse=True))
        # Q4_K_M (needs 32, have 24) is closer than Q8_0 (needs 60, have 24).
        names = [p.format for p in infeasible]
        self.assertLess(names.index("Q4_K_M"), names.index("Q8_0"))

    def test_pick_returns_one_entry_per_variant(self):
        ranked = pick(self.h100, self.entry)
        self.assertEqual(len(ranked), len(self.entry.quants))

    def test_pick_carries_full_pick_payload(self):
        ranked = pick(self.h100, self.entry)
        for p in ranked:
            self.assertEqual(p.model, self.entry.model)
            self.assertEqual(p.hf_id, self.entry.hf_id)
            self.assertEqual(p.chat_template_ref, self.entry.chat_template_ref)
            self.assertEqual(p.registry_key, self.entry.registry_key)
            self.assertIsInstance(p.args, tuple)

    def test_summary_tag_feasibility(self):
        ranked = pick(self.h100, self.entry)
        for p in ranked:
            tag = "ok" if p.feasible else "no-fit"
            self.assertIn(tag, p.summary())


class TestBestPick(unittest.TestCase):
    def setUp(self):
        self.h100 = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                             name="H100", source="simulated")
        self.no_gpu = GPUProbe()
        self.entry = _entry(quants=[
            _q("Q4_K_M", "llama.cpp", 32, cc=7.5),
            _q("Q8_0", "llama.cpp", 60, cc=7.5),
        ])

    def test_returns_highest_feasible(self):
        chosen = best_pick(self.h100, self.entry)
        self.assertIsNotNone(chosen)
        self.assertTrue(chosen.feasible)
        # Q8_0 (acc 0.95) beats Q4_K_M (acc 0.80) on an 80GB card with
        # headroom — accuracy is the primary axis.
        self.assertEqual(chosen.format, "Q8_0")

    def test_returns_none_when_nothing_fits(self):
        self.assertIsNone(best_pick(self.no_gpu, self.entry))


class TestBestPickPerModel(unittest.TestCase):
    def test_maps_key_to_pick_or_none(self):
        h100 = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                        name="H100", source="simulated")
        no_gpu = GPUProbe()
        entries = {
            "fits": _entry(key="fits", quants=[
                _q("Q4_K_M", "llama.cpp", 32, cc=7.5),
            ]),
            "doesnotfit": _entry(key="doesnotfit", quants=[
                _q("NVFP4", "vllm", 200, cc=9.0),
            ]),
        }
        ranked_h100 = best_pick_per_model(h100, entries)
        self.assertIsNotNone(ranked_h100["fits"])
        self.assertIsNone(ranked_h100["doesnotfit"])

        ranked_none = best_pick_per_model(no_gpu, entries)
        self.assertIsNone(ranked_none["fits"])
        self.assertIsNone(ranked_none["doesnotfit"])


# ---------------------------------------------------------------------------
# select — override-aware
# ---------------------------------------------------------------------------

class TestSelect(unittest.TestCase):
    def setUp(self):
        self.h100 = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                             name="H100", source="simulated")
        self.entry = _entry(quants=[
            _q("NVFP4", "vllm", 48, cc=9.0),
            _q("Q4_K_M", "llama.cpp", 32, cc=7.5),
            _q("Q8_0", "llama.cpp", 60, cc=7.5),
        ])

    def test_default_picks_best_feasible(self):
        chosen = select(self.h100, self.entry)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.format, "Q8_0")  # accuracy-primary

    def test_backend_override_restricts_set(self):
        chosen = select(self.h100, self.entry, backend="vllm")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.backend, "vllm")
        self.assertEqual(chosen.format, "NVFP4")  # only vllm feasible variant

    def test_quant_override_restricts_set(self):
        chosen = select(self.h100, self.entry, quant_format="Q4_K_M")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.format, "Q4_K_M")

    def test_override_not_feasible_returns_none(self):
        # NVFP4 needs Hopper (9.0); force a 4090 (ada 8.9) and ask for vllm.
        rtx4090 = GPUProbe(vram_gb=24, compute_cap=8.9, arch_family="ada",
                           name="4090", source="simulated")
        chosen = select(rtx4090, self.entry, backend="vllm")
        self.assertIsNone(chosen)  # the override does not fit this GPU

    def test_both_overrides_honored(self):
        chosen = select(self.h100, self.entry, backend="llama.cpp",
                        quant_format="Q8_0")
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.backend, "llama.cpp")
        self.assertEqual(chosen.format, "Q8_0")


# ---------------------------------------------------------------------------
# Pick dataclass
# ---------------------------------------------------------------------------

class TestPickDataclass(unittest.TestCase):
    def test_launch_args_mutable_list(self):
        p = Pick(registry_key="k", model="m", hf_id="org/m",
                 quant=_q("Q4_K_M", "llama.cpp", 32),
                 backend="llama.cpp", chat_template_ref="t",
                 args=("-c", "8192"), score=0.9, feasible=True)
        la = p.launch_args
        self.assertIsInstance(la, list)
        self.assertEqual(la, ["-c", "8192"])
        # mutating the returned list must not affect the immutable Pick.
        la.append("HACK")
        self.assertNotIn("HACK", p.launch_args)

    def test_format_property(self):
        p = Pick(registry_key="k", model="m", hf_id="org/m",
                 quant=_q("NVFP4", "vllm", 48, cc=9.0),
                 backend="vllm", chat_template_ref="t",
                 args=(), score=0.9, feasible=True)
        self.assertEqual(p.format, "NVFP4")

    def test_frozen(self):
        p = Pick(registry_key="k", model="m", hf_id="org/m",
                 quant=_q("Q4", "vllm", 32),
                 backend="vllm", chat_template_ref="t",
                 args=(), score=0.9, feasible=True)
        with self.assertRaises(Exception):
            p.score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Against the real shipped registry (all 4 hero models)
# ---------------------------------------------------------------------------

class TestShippedRegistry(unittest.TestCase):
    """The real registry/*.yaml shipped in this build (4 hero models)."""

    def setUp(self):
        self.reg_dir = ROOT / "registry"
        if not self.reg_dir.is_dir():
            self.skipTest("registry/ dir not shipped in this tree")
        self.entries = load_registry_dir(self.reg_dir)
        self.h100 = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                             name="H100", source="simulated")
        self.rtx4090 = GPUProbe(vram_gb=24, compute_cap=8.9, arch_family="ada",
                                name="4090", source="simulated")
        self.no_gpu = GPUProbe()

    def test_four_hero_models_present(self):
        for key in ("deepseek-v4", "qwen-3.6", "glm-5.2", "kimi-k3"):
            self.assertIn(key, self.entries)

    def test_each_entry_has_vllm_and_llama_cpp(self):
        for key, entry in self.entries.items():
            backends = {q.backend for q in entry.quants}
            self.assertIn("vllm", backends, f"{key} missing a vllm variant")
            self.assertIn("llama.cpp", backends,
                          f"{key} missing a llama.cpp variant")
            self.assertGreaterEqual(len(entry.quants), 2,
                                    f"{key} needs >=2 variants")

    def test_h100_fits_every_hero(self):
        # An 80GB Hopper card clears at least one variant per hero model.
        ranked = best_pick_per_model(self.h100, self.entries)
        for key in self.entries:
            self.assertIsNotNone(ranked[key],
                                 f"{key} has no feasible variant on H100")

    def test_h100_best_pick_is_feasible_and_scored(self):
        for key, entry in self.entries.items():
            chosen = best_pick(self.h100, entry)
            self.assertIsNotNone(chosen)
            self.assertTrue(chosen.feasible)
            self.assertGreater(chosen.score, 0.0)

    def test_4090_only_fits_small_moe(self):
        # A 24GB Ada card: only qwen-3.6 (the small A3B) should fit; the
        # three heavier MoEs exceed 24GB on every variant.
        ranked = best_pick_per_model(self.rtx4090, self.entries)
        self.assertIsNotNone(ranked["qwen-3.6"])
        self.assertIsNone(ranked["deepseek-v4"])
        self.assertIsNone(ranked["glm-5.2"])
        self.assertIsNone(ranked["kimi-k3"])

    def test_no_gpu_yields_all_none(self):
        ranked = best_pick_per_model(self.no_gpu, self.entries)
        self.assertTrue(all(v is None for v in ranked.values()))

    def test_override_picks_respect_feasibility(self):
        # Forcing vllm on a 4090 must yield None (NVFP4 needs Hopper).
        for key, entry in self.entries.items():
            if not any(q.backend == "vllm" for q in entry.quants):
                continue
            chosen = select(self.rtx4090, entry, backend="vllm")
            self.assertIsNone(chosen,
                              f"{key} vllm override should be infeasible on 4090")


if __name__ == "__main__":
    unittest.main()
