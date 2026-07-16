"""Unit tests for the m1 GPU probe module.

No hardware required: the architecture mapping and nvidia-smi parser are pure
functions, and the probe_gpu() orchestration is exercised against mocked
internal helpers (pynvml is imported lazily inside the module, so importing
zerorun.probe never needs pynvml installed).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make src/ importable when tests run without an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerorun.probe import (  # noqa: E402
    GPUProbe,
    arch_family_from_compute_cap,
    parse_nvidia_smi_output,
    probe_gpu,
)


class TestArchFamily(unittest.TestCase):
    """arch_family_from_compute_cap maps CUDA CC to coarse arch buckets."""

    def test_hopper(self):
        self.assertEqual(arch_family_from_compute_cap(9.0), "hopper")

    def test_ada_lovelace(self):
        self.assertEqual(arch_family_from_compute_cap(8.9), "ada")

    def test_ampere_a100(self):
        self.assertEqual(arch_family_from_compute_cap(8.0), "ampere")

    def test_ampere_ga10x(self):
        self.assertEqual(arch_family_from_compute_cap(8.6), "ampere")

    def test_turing(self):
        self.assertEqual(arch_family_from_compute_cap(7.5), "turing")

    def test_volta(self):
        self.assertEqual(arch_family_from_compute_cap(7.0), "volta")

    def test_blackwell(self):
        self.assertEqual(arch_family_from_compute_cap(10.0), "blackwell")
        self.assertEqual(arch_family_from_compute_cap(12.0), "blackwell")

    def test_unknown_on_zero(self):
        self.assertEqual(arch_family_from_compute_cap(0.0), "unknown")

    def test_unknown_on_negative(self):
        self.assertEqual(arch_family_from_compute_cap(-1), "unknown")


class TestParseNvidiaSmi(unittest.TestCase):
    """parse_nvidia_smi_output turns CSV stdout into a GPUProbe."""

    def test_single_gpu_rtx4090(self):
        p = parse_nvidia_smi_output("NVIDIA GeForce RTX 4090, 24564, 8.9")
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "NVIDIA GeForce RTX 4090")
        self.assertAlmostEqual(p.vram_gb, 24564 / 1024, places=2)
        self.assertEqual(p.compute_cap, 8.9)
        self.assertEqual(p.arch_family, "ada")
        self.assertEqual(p.source, "nvidia-smi")

    def test_hopper_h100(self):
        p = parse_nvidia_smi_output("NVIDIA H100 80GB HBM3, 81559, 9.0")
        self.assertIsNotNone(p)
        self.assertEqual(p.arch_family, "hopper")
        self.assertEqual(p.compute_cap, 9.0)
        self.assertGreater(p.vram_gb, 79)

    def test_multi_gpu_rig_takes_device_zero(self):
        out = (
            "NVIDIA RTX 6000 Ada, 49140, 8.9\n"
            "NVIDIA RTX 6000 Ada, 49140, 8.9"
        )
        p = parse_nvidia_smi_output(out)
        self.assertIsNotNone(p)
        self.assertEqual(p.compute_cap, 8.9)
        self.assertEqual(p.vram_gb, round(49140 / 1024, 2))

    def test_no_devices_string(self):
        self.assertIsNone(parse_nvidia_smi_output("No devices were found"))
        self.assertIsNone(parse_nvidia_smi_output("no devices"))
        self.assertIsNone(parse_nvidia_smi_output(""))

    def test_unparseable_line_skipped(self):
        p = parse_nvidia_smi_output("garbage line no commas\nRTX 4090, 24564, 8.9")
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "RTX 4090")


class TestGPUProbeDataclass(unittest.TestCase):
    def test_has_gpu_true(self):
        p = GPUProbe(vram_gb=24, compute_cap=8.9, arch_family="ada",
                     name="4090", source="nvidia-smi")
        self.assertTrue(p.has_gpu)

    def test_has_gpu_false_without_vram(self):
        self.assertFalse(GPUProbe(vram_gb=0, compute_cap=8.9).has_gpu)

    def test_has_gpu_false_without_compute_cap(self):
        self.assertFalse(GPUProbe(vram_gb=24, compute_cap=0).has_gpu)

    def test_default_is_no_gpu(self):
        p = GPUProbe()
        self.assertFalse(p.has_gpu)
        self.assertEqual(p.arch_family, "unknown")
        self.assertEqual(p.source, "none")

    def test_meets_compute_cap(self):
        p = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                     name="H100", source="pynvml")
        self.assertTrue(p.meets_compute_cap(9.0))
        self.assertTrue(p.meets_compute_cap(8.0))
        self.assertFalse(p.meets_compute_cap(9.5))

    def test_frozen(self):
        p = GPUProbe(vram_gb=24, compute_cap=8.9, arch_family="ada",
                     name="4090", source="nvidia-smi")
        with self.assertRaises(Exception):
            p.vram_gb = 99  # type: ignore[misc]


class TestProbeGpuOrchestration(unittest.TestCase):
    """probe_gpu() merges pynvml + nvidia-smi; mock both internal helpers."""

    def setUp(self):
        import zerorun.probe as probe_mod
        self.probe_mod = probe_mod
        self._orig_pyn = probe_mod._probe_via_pynvml
        self._orig_smi = probe_mod._probe_via_nvidia_smi

    def tearDown(self):
        self.probe_mod._probe_via_pynvml = self._orig_pyn
        self.probe_mod._probe_via_nvidia_smi = self._orig_smi

    def _patch(self, pyn=None, smi=None):
        if pyn is not None:
            self.probe_mod._probe_via_pynvml = lambda: pyn
        if smi is not None:
            self.probe_mod._probe_via_nvidia_smi = lambda: smi

    def test_complete_pynvml_probe_wins(self):
        pyn = GPUProbe(vram_gb=80, compute_cap=9.0, arch_family="hopper",
                       name="H100", source="pynvml")
        self._patch(pyn=pyn, smi=None)
        self.assertEqual(probe_gpu(), pyn)

    def test_pynvml_vram_only_merges_compute_cap_from_smi(self):
        pyn = GPUProbe(vram_gb=80, compute_cap=0.0, arch_family="unknown",
                       name="H100", source="pynvml")
        smi = GPUProbe(vram_gb=79, compute_cap=9.0, arch_family="hopper",
                       name="H100", source="nvidia-smi")
        self._patch(pyn=pyn, smi=smi)
        result = probe_gpu()
        self.assertEqual(result.vram_gb, 80)          # pynvml VRAM retained
        self.assertEqual(result.compute_cap, 9.0)     # filled from nvidia-smi
        self.assertEqual(result.arch_family, "hopper")
        self.assertEqual(result.source, "pynvml+nvidia-smi")

    def test_pynvml_none_falls_back_to_nvidia_smi(self):
        smi = GPUProbe(vram_gb=24, compute_cap=8.9, arch_family="ada",
                       name="4090", source="nvidia-smi")
        self._patch(pyn=None, smi=smi)
        self.assertEqual(probe_gpu(), smi)

    def test_both_none_returns_no_gpu_sentinel(self):
        self._patch(pyn=None, smi=None)
        result = probe_gpu()
        self.assertFalse(result.has_gpu)
        self.assertEqual(result.source, "none")
        self.assertEqual(result.compute_cap, 0.0)
        self.assertEqual(result.name, "no-nvidia-gpu")

    def test_pynvml_vram_only_with_no_smi_returns_partial(self):
        # pynvml knew VRAM but no compute cap; nvidia-smi also unavailable.
        pyn = GPUProbe(vram_gb=80, compute_cap=0.0, arch_family="unknown",
                       name="H100", source="pynvml")
        self._patch(pyn=pyn, smi=None)
        result = probe_gpu()
        self.assertEqual(result.vram_gb, 80)
        self.assertEqual(result.compute_cap, 0.0)
        self.assertFalse(result.has_gpu)

    def test_smi_with_zero_compute_cap_not_used_as_complete(self):
        # A degenerate smi probe (cc=0) must not satisfy the merge path.
        pyn = GPUProbe(vram_gb=80, compute_cap=0.0, name="H100", source="pynvml")
        smi = GPUProbe(vram_gb=79, compute_cap=0.0, name="H100",
                       source="nvidia-smi")
        self._patch(pyn=pyn, smi=smi)
        result = probe_gpu()
        self.assertEqual(result.compute_cap, 0.0)
        self.assertEqual(result.vram_gb, 80)


if __name__ == "__main__":
    unittest.main()
