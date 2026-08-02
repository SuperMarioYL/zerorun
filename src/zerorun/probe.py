"""GPU probing for ZeroRun.

Detects the primary NVIDIA GPU's VRAM, compute capability, and architecture
family. pynvml is the primary source (VRAM + name + compute capability); the
``nvidia-smi`` CLI is a graceful fallback for environments where pynvml is
absent or cannot resolve the compute capability. Pure helpers (architecture
mapping, nvidia-smi output parsing) are factored out so the picker (m2) and
CLI (polish) can build on this module and unit-test it without any hardware.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class GPUProbe:
    """Hardware snapshot of the primary GPU.

    The first three fields mirror the plan's core primitive:
    ``{vram_gb, compute_cap, arch_family}``. ``name`` and ``source`` are
    diagnostic metadata surfaced by ``zerorun probe`` so a user can tell
    *how* the card was detected.
    """

    vram_gb: float = 0.0
    compute_cap: float = 0.0
    arch_family: str = "unknown"
    name: str = ""
    source: str = "none"  # "pynvml" | "nvidia-smi" | "pynvml+nvidia-smi" | "none"

    @property
    def has_gpu(self) -> bool:
        """True when a usable NVIDIA GPU was detected (VRAM known).

        v0.2 fix: decoupled from ``compute_cap``. A *partial* probe (VRAM known,
        compute cap unknown — pynvml resolved the card but not its compute
        capability and nvidia-smi was unavailable) now reports ``has_gpu=True``
        so the picker can still score VRAM fit; :func:`zerorun.picker.is_feasible`
        skips the compute-cap floor when ``compute_cap == 0``. The zeroed
        ``source="none"`` probe (no card at all) stays ``False`` because
        ``vram_gb == 0``.
        """
        return self.vram_gb > 0

    @property
    def has_compute_cap(self) -> bool:
        """True when the compute capability is known (not the partial case)."""
        return self.compute_cap > 0

    def meets_compute_cap(self, floor: float) -> bool:
        """Whether this GPU clears a quant's ``compute_cap_min`` floor.

        Returns ``True`` when the compute capability is unknown (``0.0``) —
        the partial-probe graceful-degradation path: an unknown arch is treated
        as "no floor enforced" rather than failing every variant, matching the
        documented intent that the picker still scores VRAM fit without a known
        compute cap.
        """
        if self.compute_cap <= 0:
            return True
        return self.compute_cap >= float(floor)


# Coarse architecture buckets keyed off the CUDA compute capability.
# Order matters: first range (upper bound exclusive) that matches wins.
_ARCH_RANGES: tuple[tuple[float, str], ...] = (
    (7.5, "volta"),     # 7.0 — Volta (V100)
    (8.0, "turing"),    # 7.5 — Turing (RTX 20xx, T4)
    (8.9, "ampere"),    # 8.0–8.8 — Ampere (A100, GA10x, Orin)
    (9.0, "ada"),       # 8.9 — Ada Lovelace (RTX 40xx, L40/L4)
    (10.0, "hopper"),   # 9.0 — Hopper (H100), for Flash-Attention-4
)


def arch_family_from_compute_cap(compute_cap: float) -> str:
    """Map a CUDA compute capability to a coarse architecture family.

    Returns one of ``volta`` / ``turing`` / ``ampere`` / ``ada`` / ``hopper``
    / ``blackwell``, or ``unknown`` for a non-positive capability (no GPU).
    These families line up with the registry's ``compute_cap_min`` floors so
    the picker's "arch match" is really a numeric floor check.
    """
    cc = float(compute_cap)
    if cc <= 0:
        return "unknown"
    for upper, family in _ARCH_RANGES:
        if cc < upper:
            return family
    return "blackwell"  # 10.0+ — Blackwell (B100/B200, RTX 50xx)


# nvidia-smi query: name, total VRAM (MiB), compute capability (major.minor).
_NVIDIA_SMI_QUERY: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=name,memory.total,compute_cap",
    "--format=csv,noheader,nounits",
)

# Parses a single nvidia-smi CSV line, e.g.:
#   "NVIDIA GeForce RTX 4090, 24564, 8.9"
_SMI_LINE_RE = re.compile(
    r"^\s*(?P<name>[^,]+?)\s*,\s*(?P<mem>[\d.]+)\s*,\s*(?P<cc>[\d.]+)\s*$"
)


def parse_nvidia_smi_output(stdout: str) -> Optional[GPUProbe]:
    """Parse ``nvidia-smi --query-gpu=...`` CSV output into a GPUProbe.

    The first parseable GPU line wins (device 0) — the picker scores against a
    single card's resources; multi-GPU tensor-parallel sizing is a launch-time
    concern handled by the m2 launcher. Returns ``None`` when no GPU line is
    parseable (e.g. "No devices were found" or a non-NVIDIA box).
    """
    if not stdout:
        return None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("no devices") or "not found" in lowered:
            continue
        match = _SMI_LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name").strip()
        try:
            mem_mib = float(match.group("mem"))
            cc = float(match.group("cc"))
        except ValueError:
            continue
        if cc <= 0 or mem_mib <= 0:
            continue
        vram_gb = round(mem_mib / 1024.0, 2)
        return GPUProbe(
            vram_gb=vram_gb,
            compute_cap=round(cc, 2),
            arch_family=arch_family_from_compute_cap(cc),
            name=name,
            source="nvidia-smi",
        )
    return None


def _probe_via_pynvml() -> Optional[GPUProbe]:
    """Probe via pynvml.

    Returns a complete GPUProbe (VRAM + compute cap) when pynvml can resolve
    everything. If pynvml loads but cannot resolve the compute capability, a
    *partial* probe (VRAM known, ``compute_cap == 0``) is returned so the
    orchestrator can merge the missing field from nvidia-smi. Returns ``None``
    when pynvml is unavailable or NVML fails to initialize.
    """
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        count = pynvml.nvmlDeviceGetCount()
        if count < 1:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = round(float(mem.total) / (1024.0 ** 3), 2)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "ignore")
        cc = 0.0
        getter = getattr(pynvml, "nvmlDeviceGetCudaComputeCapability", None)
        if getter is not None:
            try:
                caps = getter(handle)
                if caps and len(caps) >= 2:
                    cc = float(caps[0]) + float(caps[1]) / 10.0
            except Exception:
                cc = 0.0
        if cc <= 0:
            # pynvml resolved VRAM/name but not compute cap — partial probe;
            # the orchestrator fills compute_cap from nvidia-smi (merge path).
            return GPUProbe(
                vram_gb=vram_gb,
                compute_cap=0.0,
                arch_family="unknown",
                name=str(name),
                source="pynvml",
            )
        return GPUProbe(
            vram_gb=vram_gb,
            compute_cap=round(cc, 2),
            arch_family=arch_family_from_compute_cap(cc),
            name=str(name),
            source="pynvml",
        )
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _probe_via_nvidia_smi() -> Optional[GPUProbe]:
    """Probe via the ``nvidia-smi`` CLI subprocess.

    Returns a complete GPUProbe, or ``None`` when nvidia-smi is not installed
    or reports no GPU. Never raises.
    """
    try:
        proc = subprocess.run(
            list(_NVIDIA_SMI_QUERY),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0:
        return None
    return parse_nvidia_smi_output(proc.stdout)


def probe_gpu() -> GPUProbe:
    """Detect the primary NVIDIA GPU and return a GPUProbe.

    Strategy (per the plan): pynvml first, nvidia-smi fallback. If pynvml
    resolves VRAM but not the compute capability, the compute cap is merged
    in from nvidia-smi while pynvml's VRAM is retained (``source`` becomes
    ``"pynvml+nvidia-smi"``). When neither source finds a usable GPU, a
    zeroed ``source="none"`` probe is returned so the picker can cleanly
    report that no registry model fits the (absent) hardware.
    """
    pyn = _probe_via_pynvml()
    if pyn is not None and pyn.compute_cap > 0:
        return pyn

    smi = _probe_via_nvidia_smi()

    # pynvml had VRAM but lacked compute cap -> fill from nvidia-smi.
    if (
        pyn is not None
        and pyn.vram_gb > 0
        and smi is not None
        and smi.compute_cap > 0
    ):
        return replace(
            pyn,
            compute_cap=smi.compute_cap,
            arch_family=arch_family_from_compute_cap(smi.compute_cap),
            source="pynvml+nvidia-smi",
        )

    if smi is not None and smi.has_gpu and smi.has_compute_cap:
        return smi

    # pynvml gave at least VRAM; keep the partial so the picker can still
    # score VRAM fit even without a known compute capability.
    if pyn is not None and pyn.vram_gb > 0:
        return pyn

    return GPUProbe(
        vram_gb=0.0,
        compute_cap=0.0,
        arch_family="unknown",
        name="no-nvidia-gpu",
        source="none",
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    probe = probe_gpu()
    print(f"GPU: {probe.name}")
    print(f"VRAM: {probe.vram_gb} GB")
    print(f"Compute capability: {probe.compute_cap}")
    print(f"Architecture family: {probe.arch_family}")
    print(f"Source: {probe.source}")
