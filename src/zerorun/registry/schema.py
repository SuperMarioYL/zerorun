"""Versioned model-config registry schema for ZeroRun.

The registry is **data**, not code: each YAML entry bundles the quant formats,
backends, VRAM/compute-capability floors, launch args, and chat-template
reference that today are scattered across forum posts and maintainer memory.
This module defines the typed data model, validation, and loaders that the
picker (m2) and CLI (polish) build on.

The on-disk shape (one entry per file under ``registry/<alias>.yaml``) is::

    model: deepseek-v4-flash-dspark          # full model name (informational)
    hf_id: deepseek-ai/DeepSeek-V4-flash-dspark  # backend auto-downloads weights
    description: ...                        # optional, one line
    quants:                                 # >=1 variant; v0.1 ships vllm + llama.cpp
      - format: NVFP4
        backend: vllm
        min_vram_gb: 48                     # picker's VRAM-fit floor
        compute_cap_min: 9.0                # picker's arch floor (optional, default 0)
        args: ["--kv-cache-dtype", "fp8"]
    chat_template_ref: deepseek-v4
    staleness_pin: "deepseek-v4@2026-07"    # m3: drift alert vs upstream

The CLI alias the user types (``zerorun deepseek-v4``) is the filename stem,
so a community PR is just "drop a new ``<alias>.yaml``" — no code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

#: Backends ZeroRun ships at v0.1. ExLlamaV3 is m3 roadmap (out of scope).
VALID_BACKENDS: tuple[str, ...] = ("vllm", "llama.cpp")


class RegistryError(ValueError):
    """Raised when a registry YAML entry is missing fields or invalid."""


@dataclass(frozen=True)
class QuantConfig:
    """One quant/backend variant of a model entry.

    The picker (m2) scores each variant against a :class:`GPUProbe` by three
    feasibility axes — VRAM fit (``min_vram_gb``), compute-capability floor
    (``compute_cap_min``), and backend availability — then ranks by a
    documented perf-vs-accuracy heuristic.
    """

    format: str
    backend: str
    min_vram_gb: float
    compute_cap_min: float = 0.0
    args: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuantConfig":
        """Validate and build a QuantConfig from a parsed YAML mapping."""
        if not isinstance(raw, dict):
            raise RegistryError(
                f"quant entry must be a mapping, got {type(raw).__name__}"
            )

        fmt = raw.get("format")
        if not fmt or not isinstance(fmt, str):
            raise RegistryError("quant entry missing required 'format' string")

        backend = raw.get("backend")
        if not backend or not isinstance(backend, str):
            raise RegistryError(
                f"quant '{fmt}' missing required 'backend' string"
            )
        if backend not in VALID_BACKENDS:
            raise RegistryError(
                f"quant '{fmt}' has unknown backend '{backend}'; "
                f"expected one of {VALID_BACKENDS} (ExLlamaV3 is m3 roadmap)"
            )

        vram = raw.get("min_vram_gb")
        if vram is None:
            raise RegistryError(
                f"quant '{fmt}' ({backend}) missing required 'min_vram_gb'"
            )
        try:
            vram_f = float(vram)
        except (TypeError, ValueError):
            raise RegistryError(
                f"quant '{fmt}' ({backend}) 'min_vram_gb' must be a number"
            )
        if vram_f < 0:
            raise RegistryError(
                f"quant '{fmt}' ({backend}) 'min_vram_gb' must be >= 0"
            )

        cc_raw = raw.get("compute_cap_min", 0.0)
        try:
            cc_f = float(cc_raw)
        except (TypeError, ValueError):
            raise RegistryError(
                f"quant '{fmt}' ({backend}) 'compute_cap_min' must be a number"
            )
        if cc_f < 0:
            raise RegistryError(
                f"quant '{fmt}' ({backend}) 'compute_cap_min' must be >= 0"
            )

        args_raw = raw.get("args", [])
        if not isinstance(args_raw, list):
            raise RegistryError(
                f"quant '{fmt}' ({backend}) 'args' must be a list of strings"
            )
        args = tuple(str(a) for a in args_raw)

        return cls(
            format=fmt,
            backend=backend,
            min_vram_gb=vram_f,
            compute_cap_min=cc_f,
            args=args,
        )

    @property
    def launch_args(self) -> list[str]:
        """Backend CLI args as a mutable list (consumed by the launcher)."""
        return list(self.args)


@dataclass(frozen=True)
class ModelEntry:
    """A versioned model-config registry entry (one YAML file).

    ``registry_key`` is the filename stem — the short alias the user types on
    the CLI (``zerorun deepseek-v4``) — distinct from ``model`` (the full
    model name). This lets the launcher resolve "what the user asked for"
    from the alias while keeping the full name for display/HF id lookup.
    """

    model: str
    hf_id: str
    quants: tuple[QuantConfig, ...]
    chat_template_ref: str
    staleness_pin: Optional[str] = None
    description: Optional[str] = None
    registry_key: str = ""

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], registry_key: str = ""
    ) -> "ModelEntry":
        """Validate and build a ModelEntry from a parsed YAML mapping."""
        if not isinstance(raw, dict):
            raise RegistryError(
                f"model entry must be a mapping, got {type(raw).__name__}"
            )

        model = raw.get("model")
        if not model or not isinstance(model, str):
            raise RegistryError("model entry missing required 'model' string")

        hf_id = raw.get("hf_id")
        if not hf_id or not isinstance(hf_id, str):
            raise RegistryError(
                f"model '{model}' missing required 'hf_id' string "
                "(the backend auto-downloads weights from this HF id)"
            )

        quants_raw = raw.get("quants")
        if not isinstance(quants_raw, list) or not quants_raw:
            raise RegistryError(
                f"model '{model}' needs a non-empty 'quants' list "
                "(>=1 quant/backend variant)"
            )
        quants = tuple(QuantConfig.from_dict(q) for q in quants_raw)

        template = raw.get("chat_template_ref")
        if not template or not isinstance(template, str):
            raise RegistryError(
                f"model '{model}' missing required 'chat_template_ref'"
            )

        pin = raw.get("staleness_pin")
        if pin is not None and not isinstance(pin, str):
            raise RegistryError(
                f"model '{model}' 'staleness_pin' must be a string"
            )

        desc = raw.get("description")
        if desc is not None and not isinstance(desc, str):
            raise RegistryError(
                f"model '{model}' 'description' must be a string"
            )

        return cls(
            model=model,
            hf_id=hf_id,
            quants=quants,
            chat_template_ref=template,
            staleness_pin=pin,
            description=desc,
            registry_key=registry_key or model,
        )

    def quants_for_backend(self, backend: str) -> tuple[QuantConfig, ...]:
        """All variants that use a given backend (e.g. all llama.cpp quants)."""
        return tuple(q for q in self.quants if q.backend == backend)

    def fits_vram(self, vram_gb: float) -> tuple[QuantConfig, ...]:
        """All variants whose VRAM floor fits within ``vram_gb``."""
        return tuple(q for q in self.quants if q.min_vram_gb <= float(vram_gb))


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise RegistryError(
            f"{path}: top-level YAML must be a mapping, "
            f"got {type(loaded).__name__}"
        )
    return loaded


def load_model(path: str | Path, registry_key: Optional[str] = None) -> ModelEntry:
    """Load a single model entry from a YAML file.

    ``registry_key`` defaults to the filename stem (the CLI alias).
    """
    p = Path(path)
    key = registry_key if registry_key is not None else p.stem
    return ModelEntry.from_dict(_read_yaml(p), registry_key=key)


def load_registry_dir(registry_dir: str | Path) -> dict[str, ModelEntry]:
    """Load every ``*.y*ml`` entry in a registry directory.

    Returns a dict keyed by filename stem — the short alias the user types on
    the CLI (e.g. ``deepseek-v4``). Duplicate stems (e.g. ``foo.yaml`` and
    ``foo.yml`` together) raise :class:`RegistryError` to fail loud rather
    than silently shadow one entry with another.
    """
    d = Path(registry_dir)
    if not d.exists():
        raise RegistryError(f"registry directory not found: {d}")
    if not d.is_dir():
        raise RegistryError(f"registry path is not a directory: {d}")

    out: dict[str, ModelEntry] = {}
    for p in sorted(d.glob("*.y*ml")):
        if not p.is_file():
            continue
        key = p.stem
        if key in out:
            raise RegistryError(
                f"duplicate registry key '{key}' from {p} "
                f"(already loaded from another file)"
            )
        out[key] = ModelEntry.from_dict(_read_yaml(p), registry_key=key)
    return out


def default_registry_dir() -> Path:
    """Resolve the registry directory ZeroRun loads data from.

    Order of preference:
      1. ``$ZERORUN_REGISTRY_DIR`` (explicit override, e.g. a fork's registry)
      2. a ``registry/`` directory at the repo root (dev workflow)
      3. a ``registry/`` in the current working directory
      4. the packaged registry shipped inside the wheel (installed mode) —
         ``zerorun/_registry/`` as written by ``pyproject.toml``'s wheel
         ``force-include``; resolved via :mod:`importlib.resources` so an
         ``uv tool install zerorun`` / ``pipx install`` user finds the hero
         models without a checkout.

    Returns the first existing candidate; if none exist, returns the repo-root
    candidate (so a later ``load_registry_dir`` raises a clear "not found").
    """
    env = os.environ.get("ZERORUN_REGISTRY_DIR")
    if env:
        return Path(env)

    # schema.py lives at <repo>/src/zerorun/registry/schema.py
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent.parent  # strip registry/ zerorun/ src/
    candidates = [
        repo_root / "registry",  # repo-root /registry  (dev)
        Path.cwd() / "registry",  # cwd /registry
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand

    # Installed mode (uv tool / pipx): the registry was force-included into the
    # wheel as zerorun/_registry/. importlib.resources resolves it against the
    # installed package location without needing a checkout.
    try:
        from importlib.resources import files  # py3.9+
        packaged = files("zerorun") / "_registry"
        if packaged.is_dir():
            return Path(str(packaged))
    except Exception:
        pass

    return candidates[0]


def list_registry_keys(registry_dir: Optional[str | Path] = None) -> list[str]:
    """Sorted registry keys (filename stems) — backs ``zerorun --list``."""
    d = Path(registry_dir) if registry_dir else default_registry_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.y*ml") if p.is_file())
