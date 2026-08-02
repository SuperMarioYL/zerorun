"""Unit tests for the v0.2 staleness guard (m4_staleness_guard).

The staleness guard is pure: ``check_staleness(pin, *, now) -> StalenessReport``
with no I/O, so it unit-tests without a clock or a network. These tests cover
the fresh/stale/unknown verdicts, the malformed/missing-pin honest-unknown
path, future-dated pins, the whole-month age math, and a sweep over the real
shipped registry (all 4 hero models carry a ``staleness_pin``).
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

# Make src/ importable when tests run without an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerorun.staleness import (  # noqa: E402
    DEFAULT_MAX_AGE_MONTHS,
    StalenessReport,
    check_staleness,
    parse_pin,
    staleness_report_for_entries,
)
from zerorun.registry.schema import load_registry_dir  # noqa: E402


class TestParsePin(unittest.TestCase):
    def test_valid_pin(self):
        self.assertEqual(parse_pin("deepseek-v4@2026-07"),
                         ("deepseek-v4", 2026, 7))

    def test_pin_with_dots_in_model(self):
        self.assertEqual(parse_pin("glm-5.2@2026-07"),
                         ("glm-5.2", 2026, 7))

    def test_none_pin(self):
        self.assertIsNone(parse_pin(None))

    def test_empty_pin(self):
        self.assertIsNone(parse_pin(""))
        self.assertIsNone(parse_pin("   "))

    def test_malformed_date(self):
        # Bad month / bad year shape -> unknown, not a wrong answer.
        self.assertIsNone(parse_pin("deepseek-v4@2026-13"))
        self.assertIsNone(parse_pin("deepseek-v4@26-07"))
        self.assertIsNone(parse_pin("deepseek-v4@2026-7"))

    def test_malformed_shape(self):
        self.assertIsNone(parse_pin("just-a-model"))
        self.assertIsNone(parse_pin("deepseek-v4@2026-07-extra"))
        self.assertIsNone(parse_pin("@2026-07"))


class TestCheckStaleness(unittest.TestCase):
    def test_fresh(self):
        # pin 1 month old, now 2026-08 -> age 1 <= 6 -> fresh
        r = check_staleness("deepseek-v4@2026-07", now=date(2026, 8, 1))
        self.assertEqual(r.status, "fresh")
        self.assertEqual(r.age_months, 1)
        self.assertFalse(r.is_stale)
        self.assertIn("deepseek-v4", r.model)

    def test_stale(self):
        # pin 8 months old -> age 8 > 6 -> stale
        r = check_staleness("qwen-3.6@2025-12", now=date(2026, 8, 1))
        self.assertEqual(r.status, "stale")
        self.assertEqual(r.age_months, 8)
        self.assertTrue(r.is_stale)
        self.assertIn("patch may postdate", r.message)

    def test_boundary_exactly_max_age_is_fresh(self):
        # age == max_age_months -> fresh (the floor is inclusive)
        r = check_staleness(
            "glm-5.2@2026-02", now=date(2026, 8, 1), max_age_months=6
        )
        self.assertEqual(r.status, "fresh")
        self.assertEqual(r.age_months, 6)

    def test_boundary_one_over_max_is_stale(self):
        r = check_staleness(
            "glm-5.2@2026-01", now=date(2026, 8, 1), max_age_months=6
        )
        self.assertEqual(r.status, "stale")
        self.assertEqual(r.age_months, 7)

    def test_future_dated_pin_is_fresh(self):
        # A maintainer anchoring to an upcoming release -> negative age -> fresh.
        r = check_staleness("kimi-k3@2026-12", now=date(2026, 8, 1))
        self.assertEqual(r.status, "fresh")
        self.assertTrue(r.age_months < 0)

    def test_unknown_missing_pin(self):
        r = check_staleness(None, now=date(2026, 8, 1), model="x")
        self.assertEqual(r.status, "unknown")
        self.assertIsNone(r.age_months)
        self.assertFalse(r.is_stale)

    def test_unknown_malformed_pin(self):
        r = check_staleness("not-a-pin", now=date(2026, 8, 1), model="x")
        self.assertEqual(r.status, "unknown")
        self.assertIsNone(r.age_months)

    def test_custom_max_age(self):
        # With max_age=2, a 4-month-old pin flips stale.
        r = check_staleness(
            "deepseek-v4@2026-04", now=date(2026, 8, 1), max_age_months=2
        )
        self.assertEqual(r.status, "stale")
        self.assertEqual(r.age_months, 4)

    def test_model_label_fallback(self):
        # No model passed -> falls back to the pin's model half.
        r = check_staleness("deepseek-v4@2026-07", now=date(2026, 8, 1))
        self.assertEqual(r.model, "deepseek-v4")


class TestStalenessReportForEntries(unittest.TestCase):
    def test_real_registry_all_have_pins(self):
        reg = load_registry_dir(ROOT / "registry")
        reports = staleness_report_for_entries(
            reg, now=date(2026, 8, 1)
        )
        self.assertEqual(set(reports), set(reg))
        # All 4 shipped hero models carry a 2026-07 pin -> fresh at 2026-08.
        for key, rep in reports.items():
            self.assertIsInstance(rep, StalenessReport)
            self.assertEqual(rep.status, "fresh",
                             f"{key} should be fresh, got {rep.status}")
            self.assertEqual(rep.age_months, 1)

    def test_real_registry_one_unknown_when_pin_missing(self):
        reg = load_registry_dir(ROOT / "registry")
        # Surgically drop one pin to confirm the honest-unknown path.
        from zerorun.registry.schema import ModelEntry, QuantConfig
        entry = reg["deepseek-v4"]
        stripped = ModelEntry(
            model=entry.model, hf_id=entry.hf_id, quants=entry.quants,
            chat_template_ref=entry.chat_template_ref,
            staleness_pin=None, description=entry.description,
            registry_key=entry.registry_key,
        )
        reg = {**reg, "deepseek-v4": stripped}
        reports = staleness_report_for_entries(reg, now=date(2026, 8, 1))
        self.assertEqual(reports["deepseek-v4"].status, "unknown")
        self.assertEqual(reports["qwen-3.6"].status, "fresh")


if __name__ == "__main__":
    unittest.main()
