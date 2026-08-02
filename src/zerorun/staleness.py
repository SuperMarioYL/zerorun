"""Staleness guard — turn the registry's ``staleness_pin`` into an enforced drift alert.

The registry ships a ``staleness_pin`` per model (``"<model>@<YYYY-MM>"``) that, in
v0.1, was a documented-but-inert stub (``# m3: drift alert vs upstream``). v0.2
makes it machine-enforced: :func:`check_staleness` parses the pin, measures its
age in months against ``now``, and returns a :class:`StalenessReport` whose
``status`` is one of ``fresh`` / ``stale`` / ``unknown``.

Design notes
------------
* **Pure.** No filesystem, no network, no I/O. Takes an already-loaded
  :class:`~zerorun.registry.schema.ModelEntry` (or the pin string directly) and a
  ``now`` :class:`datetime.date`; returns a report. The CLI supplies ``now``;
  tests inject a fixed date so the heuristic is deterministic without a clock.
* **Honest about the unknown.** A missing/malformed pin is ``unknown``, never
  ``fresh`` — a registry author who forgets the pin is surfaced, not silently
  green-lit. A pin dated in the future (a maintainer anchoring to an upcoming
  release) is ``fresh`` (age <= max), not ``stale``.
* **Age-based, not network-based.** v0.2's guard is the registry pin's age vs
  ``now`` — the offline, testable interpretation of "drift alert vs upstream".
  A future enhancement may compare the pin to the model's Hugging Face
  ``last_modified``; that network path is explicitly out of scope for v0.2 (see
  the v0.2 plan's out_of_scope list) so the pure path stays CI-friendly.
* **Why the pin is the anchor.** The product's wedge (per the GLM-5.2 registry
  entry) is that chat templates ship broken, get patched upstream, but a saved
  GGUF carries no version pin. The ``staleness_pin`` IS ZeroRun's version pin —
  enforcing its age tells a user "this saved config may have drifted; check for
  an upstream patch", which is the alert the launcher exists to surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .registry.schema import ModelEntry

__all__ = [
    "StalenessReport",
    "STALENESS_VERB",
    "DEFAULT_MAX_AGE_MONTHS",
    "parse_pin",
    "check_staleness",
    "staleness_report_for_entries",
]


#: The verb that selects the per-model staleness report (reserved — not a model alias).
STALENESS_VERB: str = "staleness"

#: A pin older than this many months is ``stale``. 6 months mirrors the typical
#: frontier-MoE chat-template patch cadence (templates ship broken, get patched
#: within a release or two; a pin > 6 months old very likely predates a fix).
DEFAULT_MAX_AGE_MONTHS: int = 6


# Matches "<model>@<YYYY-MM>", e.g. "deepseek-v4@2026-07" or "glm-5.2@2026-07".
# The model half is permissive (dots, hyphens, alnum); the date half is strict
# (4-digit year, 2-digit month 01-12) so a malformed date -> unknown, not a
# silent wrong answer.
_PIN_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9_.\-]+)@(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$"
)


@dataclass(frozen=True)
class StalenessReport:
    """The staleness verdict for one registry model's ``staleness_pin``.

    ``status`` is one of:

    * ``fresh``  — the pin is set, parseable, and its age in months is <=
      ``max_age_months``.
    * ``stale``  — the pin is set and parseable, but older than
      ``max_age_months`` (drift alert: an upstream patch may postdate this pin).
    * ``unknown`` — the pin is missing, ``None``, or malformed; the guard
      cannot make a determination and surfaces it rather than guessing ``fresh``.

    ``age_months`` is the pin's age in whole months (``None`` when the pin
    could not be parsed). ``message`` is a one-line human-readable summary for
    ``zerorun staleness`` and the ``--list`` staleness column.
    """

    model: str
    pin: Optional[str]
    status: str  # "fresh" | "stale" | "unknown"
    age_months: Optional[int]
    message: str

    @property
    def is_stale(self) -> bool:
        """True only when ``status == "stale"`` (a positive drift alert)."""
        return self.status == "stale"


def parse_pin(pin: Optional[str]) -> Optional[tuple[str, int, int]]:
    """Parse a ``staleness_pin`` into ``(model, year, month)``.

    Returns ``None`` when the pin is missing/None or does not match the
    ``<model>@<YYYY-MM>`` shape. A failed parse is the signal that produces an
    ``unknown`` report — never a guess.
    """
    if not pin or not isinstance(pin, str):
        return None
    m = _PIN_RE.match(pin.strip())
    if not m:
        return None
    return m.group("model"), int(m.group("year")), int(m.group("month"))


def check_staleness(
    pin: Optional[str],
    *,
    now: Optional[date] = None,
    max_age_months: int = DEFAULT_MAX_AGE_MONTHS,
    model: str = "",
) -> StalenessReport:
    """Score one ``staleness_pin`` against ``now`` and return a report.

    Parameters
    ----------
    pin:
        The ``staleness_pin`` string (``"<model>@<YYYY-MM>"``) or ``None``.
    now:
        The reference date. Defaults to :func:`date.today`; tests inject a
        fixed date so the age math is deterministic without a clock.
    max_age_months:
        A pin older than this many months is ``stale``. Defaults to
        :data:`DEFAULT_MAX_AGE_MONTHS` (6).
    model:
        Optional display name (falls back to the pin's model half or ``"?"``).

    The age is whole months: ``(now.year - pin_year) * 12 + (now.month -
    pin_month)``. A pin dated in the future (negative age) is ``fresh`` — the
    maintainer anchored to an upcoming release; it cannot be stale yet.
    """
    ref = now if now is not None else date.today()
    parsed = parse_pin(pin)
    if parsed is None:
        return StalenessReport(
            model=model or "?",
            pin=pin,
            status="unknown",
            age_months=None,
            message="no parseable staleness_pin — freshness unknown",
        )
    pin_model, year, month = parsed
    age = (ref.year - year) * 12 + (ref.month - month)
    label = model or pin_model
    if age <= max_age_months:
        return StalenessReport(
            model=label,
            pin=pin,
            status="fresh",
            age_months=age,
            message=f"fresh — pin {pin} is {age} month(s) old (<= {max_age_months})",
        )
    return StalenessReport(
        model=label,
        pin=pin,
        status="stale",
        age_months=age,
        message=(
            f"stale — pin {pin} is {age} months old "
            f"(> {max_age_months}); an upstream chat-template patch may postdate this pin"
        ),
    )


def staleness_report_for_entries(
    entries: "dict[str, ModelEntry]",
    *,
    now: Optional[date] = None,
    max_age_months: int = DEFAULT_MAX_AGE_MONTHS,
) -> "dict[str, StalenessReport]":
    """A staleness report per registry key — backs ``zerorun staleness``.

    Pure: takes the already-loaded registry dict (the schema module owns the
    I/O). Models without a ``staleness_pin`` map to an ``unknown`` report so
    the CLI can render them honestly alongside the fresh/stale ones.
    """
    return {
        key: check_staleness(
            entry.staleness_pin,
            now=now,
            max_age_months=max_age_months,
            model=entry.model,
        )
        for key, entry in entries.items()
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from .registry.schema import load_registry_dir

    for key, entry in sorted(load_registry_dir("registry").items()):
        print(check_staleness(entry.staleness_pin, model=entry.model))
