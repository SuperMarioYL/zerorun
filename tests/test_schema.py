"""Unit tests for the m1 registry schema module.

Exercises QuantConfig/ModelEntry validation, the convenience selectors the
picker (m2) will lean on, and the YAML loaders against both a temp registry
and the real shipped hero entry (registry/deepseek-v4.yaml).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerorun.registry.schema import (  # noqa: E402
    VALID_BACKENDS,
    ModelEntry,
    QuantConfig,
    RegistryError,
    default_registry_dir,
    list_registry_keys,
    load_model,
    load_registry_dir,
)


HERO_YAML = """
model: deepseek-v4-flash-dspark
hf_id: deepseek-ai/DeepSeek-V4-flash-dspark
description: hero model
quants:
  - format: NVFP4
    backend: vllm
    min_vram_gb: 48
    compute_cap_min: 9.0
    args: ["--kv-cache-dtype", "fp8"]
  - format: Q4_K_M
    backend: llama.cpp
    min_vram_gb: 32
    args: ["-c", "8192"]
chat_template_ref: deepseek-v4
staleness_pin: "deepseek-v4@2026-07"
"""


class TestQuantConfig(unittest.TestCase):
    def test_from_dict_full(self):
        q = QuantConfig.from_dict({
            "format": "NVFP4", "backend": "vllm",
            "min_vram_gb": 48, "compute_cap_min": 9.0,
            "args": ["--kv-cache-dtype", "fp8"],
        })
        self.assertEqual(q.format, "NVFP4")
        self.assertEqual(q.backend, "vllm")
        self.assertEqual(q.min_vram_gb, 48.0)
        self.assertEqual(q.compute_cap_min, 9.0)
        self.assertEqual(q.args, ("--kv-cache-dtype", "fp8"))
        self.assertEqual(q.launch_args, ["--kv-cache-dtype", "fp8"])

    def test_from_dict_defaults(self):
        q = QuantConfig.from_dict(
            {"format": "Q4_K_M", "backend": "llama.cpp", "min_vram_gb": 32}
        )
        self.assertEqual(q.compute_cap_min, 0.0)
        self.assertEqual(q.args, ())

    def test_vram_as_string_coerced(self):
        q = QuantConfig.from_dict(
            {"format": "Q4", "backend": "vllm", "min_vram_gb": "32"}
        )
        self.assertEqual(q.min_vram_gb, 32.0)

    def test_missing_format(self):
        with self.assertRaises(RegistryError):
            QuantConfig.from_dict({"backend": "vllm", "min_vram_gb": 48})

    def test_missing_backend(self):
        with self.assertRaises(RegistryError):
            QuantConfig.from_dict({"format": "NVFP4", "min_vram_gb": 48})

    def test_invalid_backend_rejected(self):
        with self.assertRaises(RegistryError):
            QuantConfig.from_dict(
                {"format": "X", "backend": "exl2", "min_vram_gb": 48}
            )

    def test_missing_min_vram(self):
        with self.assertRaises(RegistryError):
            QuantConfig.from_dict({"format": "X", "backend": "vllm"})

    def test_negative_vram_rejected(self):
        with self.assertRaises(RegistryError):
            QuantConfig.from_dict(
                {"format": "X", "backend": "vllm", "min_vram_gb": -1}
            )

    def test_args_not_list_rejected(self):
        with self.assertRaises(RegistryError):
            QuantConfig.from_dict({
                "format": "X", "backend": "vllm", "min_vram_gb": 1,
                "args": "nope",
            })

    def test_valid_backends_set(self):
        self.assertIn("vllm", VALID_BACKENDS)
        self.assertIn("llama.cpp", VALID_BACKENDS)

    def test_frozen(self):
        q = QuantConfig.from_dict(
            {"format": "Q4", "backend": "vllm", "min_vram_gb": 32}
        )
        with self.assertRaises(Exception):
            q.format = "Q8"  # type: ignore[misc]


class TestModelEntry(unittest.TestCase):
    def _raw(self, **over):
        base = {
            "model": "m1", "hf_id": "org/m1",
            "quants": [
                {"format": "NVFP4", "backend": "vllm", "min_vram_gb": 48},
            ],
            "chat_template_ref": "m1-tmpl",
            "staleness_pin": "m1@2026-07",
        }
        base.update(over)
        return base

    def test_from_dict_full(self):
        e = ModelEntry.from_dict(self._raw(), registry_key="m1")
        self.assertEqual(e.model, "m1")
        self.assertEqual(e.hf_id, "org/m1")
        self.assertEqual(len(e.quants), 1)
        self.assertEqual(e.chat_template_ref, "m1-tmpl")
        self.assertEqual(e.staleness_pin, "m1@2026-07")
        self.assertEqual(e.registry_key, "m1")

    def test_registry_key_defaults_to_model(self):
        e = ModelEntry.from_dict(self._raw())
        self.assertEqual(e.registry_key, "m1")

    def test_missing_model(self):
        with self.assertRaises(RegistryError):
            ModelEntry.from_dict({
                "hf_id": "x", "quants": [], "chat_template_ref": "x",
            })

    def test_missing_hf_id(self):
        with self.assertRaises(RegistryError):
            ModelEntry.from_dict({
                "model": "x", "quants": [], "chat_template_ref": "x",
            })

    def test_empty_quants_rejected(self):
        with self.assertRaises(RegistryError):
            ModelEntry.from_dict({
                "model": "x", "hf_id": "x", "quants": [],
                "chat_template_ref": "x",
            })

    def test_missing_chat_template_ref(self):
        with self.assertRaises(RegistryError):
            ModelEntry.from_dict({
                "model": "x", "hf_id": "x",
                "quants": [{"format": "Q", "backend": "vllm", "min_vram_gb": 1}],
            })

    def test_quants_for_backend(self):
        e = ModelEntry.from_dict({
            "model": "m", "hf_id": "m",
            "quants": [
                {"format": "NVFP4", "backend": "vllm", "min_vram_gb": 48},
                {"format": "Q4_K_M", "backend": "llama.cpp", "min_vram_gb": 32},
                {"format": "Q8_0", "backend": "llama.cpp", "min_vram_gb": 60},
            ],
            "chat_template_ref": "m",
        })
        self.assertEqual(len(e.quants_for_backend("vllm")), 1)
        self.assertEqual(len(e.quants_for_backend("llama.cpp")), 2)
        self.assertEqual(len(e.quants_for_backend("exl2")), 0)

    def test_fits_vram(self):
        e = ModelEntry.from_dict({
            "model": "m", "hf_id": "m",
            "quants": [
                {"format": "NVFP4", "backend": "vllm", "min_vram_gb": 48},
                {"format": "Q4_K_M", "backend": "llama.cpp", "min_vram_gb": 32},
            ],
            "chat_template_ref": "m",
        })
        self.assertEqual(len(e.fits_vram(24)), 0)
        self.assertEqual(len(e.fits_vram(40)), 1)
        self.assertEqual(len(e.fits_vram(80)), 2)


class TestLoaders(unittest.TestCase):
    def test_load_model_from_tmp(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fake.yaml"
            p.write_text(HERO_YAML, encoding="utf-8")
            e = load_model(p, registry_key="fake")
            self.assertEqual(e.model, "deepseek-v4-flash-dspark")
            self.assertEqual(e.hf_id, "deepseek-ai/DeepSeek-V4-flash-dspark")
            self.assertEqual(e.registry_key, "fake")
            self.assertGreaterEqual(len(e.quants), 2)

    def test_load_model_key_defaults_to_stem(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "alias.yaml"
            p.write_text(HERO_YAML, encoding="utf-8")
            e = load_model(p)
            self.assertEqual(e.registry_key, "alias")

    def test_load_registry_dir_tmp(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a.yaml").write_text(HERO_YAML, encoding="utf-8")
            entries = load_registry_dir(d)
            self.assertEqual(len(entries), 1)
            self.assertIn("a", entries)

    def test_duplicate_key_across_yaml_yml_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "dup.yaml").write_text(HERO_YAML, encoding="utf-8")
            (d / "dup.yml").write_text(HERO_YAML, encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_registry_dir(d)

    def test_missing_dir_raises(self):
        with self.assertRaises(RegistryError):
            load_registry_dir("/nonexistent/zerorun/registry/xyz")

    def test_not_a_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "file.yaml"
            f.write_text(HERO_YAML, encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_registry_dir(f)

    def test_top_level_not_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.yaml"
            p.write_text("- just\n- a\n- list\n", encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_model(p)

    def test_empty_file_loads_as_empty_then_validates(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.yaml"
            p.write_text("", encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_model(p)

    def test_propagates_quant_validation_error(self):
        bad = """
model: m
hf_id: org/m
quants:
  - format: Q
    backend: not-a-backend
    min_vram_gb: 32
chat_template_ref: m
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.yaml"
            p.write_text(bad, encoding="utf-8")
            with self.assertRaises(RegistryError):
                load_model(p)


class TestShippedHeroEntry(unittest.TestCase):
    """The real registry/deepseek-v4.yaml shipped in this build."""

    def setUp(self):
        self.reg_dir = ROOT / "registry"
        self.hero_path = self.reg_dir / "deepseek-v4.yaml"

    def test_hero_file_present(self):
        if not self.hero_path.exists():
            self.skipTest("registry/deepseek-v4.yaml not shipped in this tree")
        e = load_model(self.hero_path)
        self.assertEqual(e.model, "deepseek-v4-flash-dspark")
        self.assertEqual(e.hf_id, "deepseek-ai/DeepSeek-V4-flash-dspark")
        self.assertEqual(e.chat_template_ref, "deepseek-v4")
        self.assertEqual(e.staleness_pin, "deepseek-v4@2026-07")
        # m1 requires >=2 quant/backend variants.
        self.assertGreaterEqual(len(e.quants), 2)
        backends = {q.backend for q in e.quants}
        self.assertIn("vllm", backends)
        self.assertIn("llama.cpp", backends)
        # every quant validates + carries a numeric floor.
        for q in e.quants:
            self.assertGreater(q.min_vram_gb, 0)
            self.assertGreaterEqual(q.compute_cap_min, 0)

    def test_load_registry_dir_finds_hero(self):
        if not self.reg_dir.is_dir():
            self.skipTest("registry/ dir not shipped in this tree")
        entries = load_registry_dir(self.reg_dir)
        self.assertIn("deepseek-v4", entries)
        self.assertEqual(
            entries["deepseek-v4"].model, "deepseek-v4-flash-dspark"
        )

    def test_list_registry_keys(self):
        if not self.reg_dir.is_dir():
            self.skipTest("registry/ dir not shipped in this tree")
        keys = list_registry_keys(self.reg_dir)
        self.assertIn("deepseek-v4", keys)

    def test_default_registry_dir_resolves_repo_root(self):
        # When run from the repo root, default_registry_dir() must find the
        # shipped registry/ directory (not raise / not point into the void).
        resolved = default_registry_dir()
        self.assertTrue(resolved.name == "registry")


if __name__ == "__main__":
    unittest.main()
