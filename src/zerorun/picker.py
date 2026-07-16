"""Picker — score a GPU probe against a model's quant variants, rank, and pick.

The picker is the m2 core: a **pure** function ``(GPUProbe, ModelEntry) ->
ranked Picks`` with no I/O, so it unit-tests without any hardware. Given a
detected GPU and a versioned model-config entry, it scores every
:class:`~zerorun.registry.schema.QuantConfig` by three feasibility axes
(VRAM fit, compute-capability/arch floor, GPU presence) and then ranks the
feasible variants by a **documented perf-vs-accuracy heuristic** (see
:func:`score_quant`).

The output :class:`Pick` bundles everything the launcher (polish stage) needs to
spawn the backend subprocess — the resolved quant, backend, chat-template
reference, Hugging Face model id, and launch args — plus a human-readable score
breakdown so ``zerorun <model> --dry-run`` and ``zerorun --list`` can show *why*
a pick was made.

Design notes
------------
* **Pure.** No filesystem, no subprocess, no network. Takes already-loaded
  :class:`~zerorun.registry.schema.ModelEntry` objects (the schema module owns
  YAML loading). The CLI iterates models for ``--list``; the picker scores one
  model at a time per the plan's signature ``(GPUProbe, model_entry) -> Picks``.
* **Honest about infeasibility.** When no variant fits the card (e.g. a 24 GB
  card against a 32 GB-floor quant), :func:`best_pick` returns ``None`` and
  :func:`pick` still returns the ranked infeasible block sorted by *closeness*
  to feasible, so the CLI can tell the user exactly how much VRAM/arch they are
  short — the same honesty the analysis report demands.
* **Override-friendly.** :func:`select` honors ``--backend`` / ``--quant``
  overrides by filtering the ranked list before returning the best feasible
  match, so a user who disagrees with the heuristic can force a path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .probe import GPUProbe
from .registry.schema import ModelEntry, QuantConfig

__all__ = [
    "Pick",
    "accuracy_score",
    "backend_throughput",
    "vram_safety",
    "is_feasible",
    "feasibility_reasons",
    "score_quant",
    "pick",
    "best_pick",
    "best_pick_per_model",
    "select",
]


# ---------------------------------------------------------------------------
# Documented perf-vs-accuracy heuristic constants
# ---------------------------------------------------------------------------
#
# The heuristic is the sum of three non-negative terms, each documented below.
# Accuracy is the dominant axis (range ~0.60–1.00); backend throughput and VRAM
# safety are tiebreakers (combined range 0.00–0.23). This means: among feasible
# variants, the higher-accuracy quant wins unless a faster backend or a much
# safer VRAM fit closes the gap — which is exactly the perf-vs-accuracy tension
# the plan names. See :func:`score_quant` for the assembled formula.

#: Per-format accuracy score — a measure of *preserved precision*. Higher is more
#: accurate. Values are illustrative-ordinal, ordered to match the established
#: GGUF/quant hierarchy (F16 > Q8_0 > Q6_K > Q5 > Q4_K_M > Q3 > Q2) and to place
#: NVFP4 (W4A4: 4-bit weight + 4-bit activation) below Q4_K_M (4-bit weight,
#: higher-precision activation). The registry's shipped formats — Q8_0, Q4_K_M,
#: NVFP4 — therefore rank Q8_0 > Q4_K_M > NVFP4 on accuracy.
_ACCURACY_SCORE: dict[str, float] = {
    "F16": 1.00,
    "BF16": 1.00,
    "FP16": 1.00,
    "F32": 1.00,
    "Q8_0": 0.95,
    "Q8_K": 0.95,
    "INT8": 0.90,
    "FP8": 0.90,
    "Q6_K": 0.88,
    "Q5_K_M": 0.84,
    "Q5_K_S": 0.83,
    "Q5_0": 0.82,
    "IQ4_NL": 0.78,
    "IQ4_XS": 0.77,
    "Q4_K_M": 0.80,
    "Q4_K_S": 0.79,
    "Q4_0": 0.76,
    "NVFP4": 0.74,  # W4A4 — tensor-core fast, lower precision than Q4_K_M
    "Q3_K_M": 0.72,
    "IQ3_M": 0.71,
    "Q3_K_S": 0.70,
    "IQ3_S": 0.69,
    "IQ3_XXS": 0.66,
    "Ternary": 0.64,  # 1.58-bit / ternary — frontier, very low precision
    "Q2_K": 0.60,
}

#: Default accuracy for an unrecognized format — mid-low, so a registry author
#: who invents a new format label does not silently get the top slot.
_DEFAULT_ACCURACY: float = 0.72


def accuracy_score(format_name: str) -> float:
    """Preserved-precision score for a quant ``format`` label.

    Falls back to :data:`_DEFAULT_ACCURACY` for formats the table does not list,
    so community-PR'd new formats still score (conservatively) rather than
    silently dominating the ranking.
    """
    return _ACCURACY_SCORE.get(format_name, _DEFAULT_ACCURACY)


def backend_throughput(backend: str, arch_family: str) -> float:
    """Throughput bonus for a backend on a given GPU architecture family.

    vLLM is the production serving backend and shines on Hopper+ where the
    NVFP4 / FP8 / Flash-Attention-4 tensor-core path is available, so it earns
    its largest bonus there. llama.cpp earns a smaller, architecture-agnostic
    bonus — it is the broad-fit path, consistently good across consumer and
    repurposed-enterprise cards. The combined bonus range (0.00–0.15) is
    deliberately smaller than the accuracy spread so accuracy stays the primary
    axis; the backend bonus only breaks near-ties.
    """
    if backend == "vllm":
        if arch_family in ("hopper", "blackwell"):
            return 0.15  # FA-4 / tensor-core path: vLLM's sweet spot
        if arch_family in ("ampere", "ada"):
            return 0.08
        return 0.04  # turing/volta/unknown — vLLM still works, less optimal
    if backend == "llama.cpp":
        return 0.05  # consistent, broad-fit
    return 0.0


def vram_safety(probe_vram_gb: float, min_vram_gb: float) -> float:
    """Safety bonus for VRAM headroom, capped so it never dominates accuracy.

    ``headroom = probe_vram_gb / min_vram_gb`` (>= 1.0 for a feasible variant).
    The bonus is ``0.04`` per unit of headroom above 1.0, capped at ``0.08`` —
    so a quant that *exactly* fits gets 0, and a quant with 3x headroom gets
    the cap. This rewards comfortable fits (lower OOM risk under kv-cache and
    context growth) without letting a tiny quant win purely on headroom.
    """
    if min_vram_gb <= 0:
        return 0.0
    headroom = probe_vram_gb / min_vram_gb
    if headroom <= 1.0:
        return 0.0
    return min(0.04 * (headroom - 1.0), 0.08)


# ---------------------------------------------------------------------------
# Feasibility (hard gate)
# ---------------------------------------------------------------------------

def is_feasible(probe: GPUProbe, quant: QuantConfig) -> bool:
    """Whether a variant can run on the probed GPU at all.

    Three hard gates, all must pass:

    1. A usable NVIDIA GPU was detected (``probe.has_gpu``).
    2. VRAM fits: ``probe.vram_gb >= quant.min_vram_gb``.
    3. Compute-capability floor met: ``probe.compute_cap >=
       quant.compute_cap_min`` (a 0 floor means "any arch" and always passes).
       This is the "arch match" — per ``probe.py``, the coarse arch families
       line up with the registry's ``compute_cap_min`` floors, so the arch
       check *is* the numeric floor check.
    """
    if not probe.has_gpu:
        return False
    if probe.vram_gb < quant.min_vram_gb:
        return False
    if quant.compute_cap_min > 0 and probe.compute_cap < quant.compute_cap_min:
        return False
    return True


def _arch_label(compute_cap_min: float) -> str:
    """Human-friendly arch name for a compute-cap floor (for reason strings)."""
    cc = float(compute_cap_min)
    if cc <= 0:
        return "any"
    if cc < 7.5:
        return "pre-turing"
    if cc < 8.0:
        return "turing"
    if cc < 8.9:
        return "ampere"
    if cc < 9.0:
        return "ada"
    if cc < 10.0:
        return "hopper"
    return "blackwell"


def feasibility_reasons(probe: GPUProbe, quant: QuantConfig) -> list[str]:
    """Human-readable list of why a variant is or is not feasible.

    Empty reasons list means "feasible, no blockers" (the caller adds the score
    breakdown). Each blocker names the exact shortfall so the CLI can say "you
    are 16 GB short" rather than just "won't fit".
    """
    reasons: list[str] = []
    if not probe.has_gpu:
        reasons.append("no NVIDIA GPU detected")
        return reasons
    if probe.vram_gb < quant.min_vram_gb:
        short = quant.min_vram_gb - probe.vram_gb
        reasons.append(
            f"VRAM short: have {probe.vram_gb:.1f} GB, need {quant.min_vram_gb:.0f} GB "
            f"(-{short:.1f} GB)"
        )
    if quant.compute_cap_min > 0 and probe.compute_cap < quant.compute_cap_min:
        reasons.append(
            f"arch short: compute-cap {probe.compute_cap} < required "
            f"{quant.compute_cap_min} ({_arch_label(quant.compute_cap_min)})"
        )
    return reasons


def _infeasible_closeness(probe: GPUProbe, quant: QuantConfig) -> float:
    """Closeness-to-feasible score for ranking the infeasible block.

    Higher (less negative) = closer to runnable, so the infeasible block sorts
    with the closest miss first — the user sees the variant they *almost* meet
    before the one that is hopelessly out of reach. Normalized so VRAM and
    compute-cap shortfalls are on comparable scales.
    """
    if not probe.has_gpu:
        return -1.0
    score = 0.0
    if probe.vram_gb < quant.min_vram_gb and quant.min_vram_gb > 0:
        # e.g. 16 GB short on a 32 GB floor -> -0.5
        score += (probe.vram_gb - quant.min_vram_gb) / quant.min_vram_gb
    if quant.compute_cap_min > 0 and probe.compute_cap < quant.compute_cap_min:
        score += probe.compute_cap - quant.compute_cap_min
    return score


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_quant(probe: GPUProbe, quant: QuantConfig) -> tuple[float, list[str]]:
    """Score one variant against the probe and return ``(score, reasons)``.

    For a **feasible** variant, the documented heuristic is::

        score = accuracy_score(format) + backend_throughput(backend, arch)
                + vram_safety(vram_gb, min_vram_gb)

    Accuracy (0.60–1.00) is the primary axis; backend throughput (0.00–0.15)
    and VRAM safety (0.00–0.08) are tiebreakers that only close near-ties.
    The reasons list carries the full breakdown so ``--dry-run`` can show the
    arithmetic.

    For an **infeasible** variant, the score is :func:`_infeasible_closeness`
    (negative; closer to 0 = closer to feasible) and is used only to order the
    infeasible block for diagnostics — it is never compared against a feasible
    score (see :func:`pick`).
    """
    feasible = is_feasible(probe, quant)
    reasons = feasibility_reasons(probe, quant)
    if not feasible:
        # reasons already describe the shortfall; closeness orders the block.
        reasons.append("infeasible — listed for diagnostics only")
        return _infeasible_closeness(probe, quant), reasons

    acc = accuracy_score(quant.format)
    bt = backend_throughput(quant.backend, probe.arch_family)
    vs = vram_safety(probe.vram_gb, quant.min_vram_gb)
    total = acc + bt + vs
    reasons.append(
        f"score: accuracy {acc:.2f} + backend {bt:.2f} "
        f"+ vram-safety {vs:.3f} = {total:.3f}"
    )
    return total, reasons


# ---------------------------------------------------------------------------
# The Pick result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pick:
    """A ranked pick: the chosen quant/backend plus everything the launcher needs.

    Carries the resolved :class:`QuantConfig`, the backend name, the
    chat-template reference, the Hugging Face model id, and the launch args the
    backend subprocess will consume — plus the heuristic score and a
    human-readable reason breakdown. ``feasible`` is False for diagnostic-only
    entries (the variants the GPU cannot run), which :func:`pick` still returns
    so the CLI can explain the shortfall.
    """

    registry_key: str
    model: str
    hf_id: str
    quant: QuantConfig
    backend: str
    chat_template_ref: str
    args: tuple[str, ...]
    score: float
    feasible: bool
    reasons: tuple[str, ...] = ()

    @property
    def launch_args(self) -> list[str]:
        """Backend CLI args as a mutable list (the launcher appends the model)."""
        return list(self.args)

    @property
    def format(self) -> str:
        """The quant format label of this pick (convenience)."""
        return self.quant.format

    def summary(self) -> str:
        """One-line summary for ``--dry-run`` / ``--list`` output.

        ``[ok] deepseek-v4-flash-dspark @ NVFP4/vllm  score 0.890``
        or, for a diagnostic-only infeasible entry, ``[no-fit] ...``.
        """
        tag = "ok" if self.feasible else "no-fit"
        return (
            f"[{tag}] {self.model} @ {self.quant.format}/{self.backend}  "
            f"score {self.score:.3f}"
        )


# ---------------------------------------------------------------------------
# Ranking entry points
# ---------------------------------------------------------------------------

def pick(probe: GPUProbe, entry: ModelEntry) -> list[Pick]:
    """Score and rank every variant of ``entry`` against ``probe``.

    Returns a list ordered: **feasible variants first** (highest heuristic
    score first), then **infeasible variants** (closest-to-feasible first, so
    the user sees the nearest miss before the hopeless ones). An empty list is
    impossible (a model entry always has >= 1 quant), but every entry may be
    infeasible when the GPU is absent or too small — in that case
    :func:`best_pick` returns ``None`` and the CLI surfaces the top infeasible
    entry's reasons.
    """
    scored: list[Pick] = []
    for q in entry.quants:
        score, reasons = score_quant(probe, q)
        scored.append(
            Pick(
                registry_key=entry.registry_key,
                model=entry.model,
                hf_id=entry.hf_id,
                quant=q,
                backend=q.backend,
                chat_template_ref=entry.chat_template_ref,
                args=q.args,
                score=score,
                feasible=is_feasible(probe, q),
                reasons=tuple(reasons),
            )
        )
    # feasible (True) sorts before infeasible (False) under reverse=True;
    # within each block, higher score first (feasible: best heuristic;
    # infeasible: closest-to-runnable).
    scored.sort(key=lambda p: (p.feasible, p.score), reverse=True)
    return scored


def best_pick(probe: GPUProbe, entry: ModelEntry) -> Optional[Pick]:
    """The highest-scoring feasible pick for ``entry``, or ``None`` if none fit.

    This is the default the launcher runs when the user types
    ``zerorun <model>`` with no overrides.
    """
    for p in pick(probe, entry):
        if p.feasible:
            return p
    return None


def best_pick_per_model(
    probe: GPUProbe,
    entries: dict[str, ModelEntry],
) -> dict[str, Optional[Pick]]:
    """Best feasible pick per registry key — backs ``zerorun --list``.

    Pure: takes the already-loaded registry dict (``schema.load_registry_dir``
    owns the I/O). Models with no feasible variant map to ``None`` so the CLI
    can render "no fit" rows honestly alongside the picked ones.
    """
    return {key: best_pick(probe, e) for key, e in entries.items()}


def select(
    probe: GPUProbe,
    entry: ModelEntry,
    backend: Optional[str] = None,
    quant_format: Optional[str] = None,
) -> Optional[Pick]:
    """Best feasible pick honoring optional ``--backend`` / ``--quant`` overrides.

    When ``backend`` and/or ``quant_format`` are given, the search is restricted
    to matching variants — but the result is still only ever a *feasible* pick
    (an override the GPU cannot run returns ``None`` rather than a diagnostic
    entry, so the CLI can say "your override does not fit this GPU" without
    launching a doomed subprocess).

    The ranked order from :func:`pick` is preserved, so within the constrained
    set the heuristic still picks the best variant.
    """
    for p in pick(probe, entry):
        if not p.feasible:
            continue
        if backend is not None and p.backend != backend:
            continue
        if quant_format is not None and p.quant.format != quant_format:
            continue
        return p
    return None


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from pathlib import Path

    from .probe import probe_gpu
    from .registry.schema import load_model

    _probe = probe_gpu()
    _entry = load_model(
        Path(__file__).resolve().parent.parent.parent.parent
        / "registry"
        / "deepseek-v4.yaml"
    )
    for _p in pick(_probe, _entry):
        print(_p.summary())
        for _r in _p.reasons:
            print(f"    {_r}")
