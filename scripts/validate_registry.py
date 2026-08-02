#!/usr/bin/env python3
"""Validate every ZeroRun registry entry against the typed schema.

Loads each ``registry/*.y*ml`` file through
``zerorun.registry.schema.load_registry_dir`` (which builds ``ModelEntry`` +
``QuantConfig`` via ``from_dict`` and raises ``RegistryError`` on any missing
required field, unknown backend, or non-numeric floor). Prints a per-file
OK/FAIL report and exits non-zero if any entry is invalid.

Usage::

    python scripts/validate_registry.py [--registry-dir DIR]

If ``--registry-dir`` is omitted, resolves the same way the CLI does:
``$ZERORUN_REGISTRY_DIR`` → ``./registry`` → the packaged registry.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_registry_dir(override: str | None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("ZERORUN_REGISTRY_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    repo_root = here.parent  # scripts/ -> repo root
    candidates = [repo_root / "registry", Path.cwd() / "registry"]
    for cand in candidates:
        if cand.is_dir():
            return cand
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    registry_dir: str | None = None
    while argv:
        a = argv.pop(0)
        if a in ("--registry-dir",):
            registry_dir = argv.pop(0) if argv else None
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            print(f"validate_registry: unknown arg {a!r}", file=sys.stderr)
            return 2

    # Make the in-tree package importable when run from a checkout (no install).
    here = Path(__file__).resolve().parent
    src = here.parent / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))

    from zerorun.registry.schema import (  # type: ignore[import-not-found]
        RegistryError,
        list_registry_keys,
        load_registry_dir,
    )

    reg_dir = _resolve_registry_dir(registry_dir)
    if not reg_dir.is_dir():
        print(f"validate_registry: registry dir not found: {reg_dir}",
              file=sys.stderr)
        return 2

    print(f"validating registry: {reg_dir}")
    keys = list_registry_keys(reg_dir)
    if not keys:
        print("  no registry entries found.", file=sys.stderr)
        return 2

    failures = 0
    try:
        entries = load_registry_dir(reg_dir)
    except RegistryError as exc:
        # A directory-level error (duplicate key, not-a-mapping) fails fast.
        print(f"  FAIL directory: {exc}", file=sys.stderr)
        return 1

    for key in keys:
        entry = entries.get(key)
        if entry is None:
            print(f"  FAIL {key}: entry missing after load", file=sys.stderr)
            failures += 1
            continue
        # Re-validate each entry's quants explicitly (from_dict already did, but
        # this surfaces a per-quant message if a future loader is lenient).
        issues: list[str] = []
        if not entry.hf_id:
            issues.append("missing hf_id")
        if not entry.quants:
            issues.append("empty quants list")
        for q in entry.quants:
            if q.min_vram_gb < 0:
                issues.append(f"{q.format}: negative min_vram_gb")
            if q.compute_cap_min < 0:
                issues.append(f"{q.format}: negative compute_cap_min")
        if issues:
            print(f"  FAIL {key}: {'; '.join(issues)}", file=sys.stderr)
            failures += 1
        else:
            n = len(entry.quants)
            pin = entry.staleness_pin or "—"
            print(f"  ok   {key}: {entry.model} ({n} quant(s), pin {pin})")

    if failures:
        print(f"\n{failures} invalid entr(y/ies) — see above.", file=sys.stderr)
        return 1
    print(f"\nall {len(keys)} entr(y/ies) valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
