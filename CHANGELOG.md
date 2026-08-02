# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-03

### Added
- **m4 — Staleness guard.** The registry's `staleness_pin` field
  (`<model>@<YYYY-MM>`) is no longer an inert stub: `zerorun staleness` prints
  a per-model drift report (`fresh` ≤ 6 months / `stale` > 6 months / `unknown`
  for a missing or malformed pin), and `zerorun --list` gains a staleness
  column. The check is pure (no network), so it runs in CI without a clock.
  This closes the staleness half of the m3 roadmap — the product's own wedge
  (a saved GGUF carries no version pin; ZeroRun's pin + drift alert surfaces
  when a config may have postdated an upstream chat-template patch).
- **m5 — Community-PR-able registry flow.** `CONTRIBUTING.md` documents how to
  PR a new `registry/<alias>.yaml` (no Python change), `scripts/validate_registry.py`
  loads every entry through the typed schema and exits non-zero on any invalid
  field, and `.github/workflows/registry-lint.yml` runs the validator + tests on
  PRs touching `registry/`. The moat (curation) is now community-extensible.

### Fixed
- **fix-partial-probe-has-gpu-invariant.** A GPU whose VRAM was detected but
  whose compute capability was unknown (pynvml partial + `nvidia-smi`
  unavailable) was reported as "no NVIDIA GPU detected" for every variant,
  blocking every pick — contradicting `probe_gpu()`'s documented
  graceful-degradation path. `GPUProbe.has_gpu` now returns true whenever VRAM
  is known, and `picker.is_feasible` skips the compute-cap floor when
  `compute_cap == 0` (still enforcing the VRAM floor). The zeroed `--no-probe`
  sentinel is unaffected.

### Roadmap (not in v0.2.0)
- **ExLlamaV3 backend.** Deferred to v0.3 — no live demand (0 open issues) and
  a heavy, moving-target dependency. v0.2 ships vLLM + llama.cpp only.
- **Network-backed upstream-drift detection.** v0.2's staleness guard is
  age-based off the registry pin (offline, testable); a future `huggingface_hub`
  `last_modified` compare is a later enhancement.

## [0.1.0] - 2026-07-17

### Added
- **m1 — GPU probe + registry.** `zerorun probe` detects VRAM, compute
  capability, and arch family (ampere / ada / hopper) via pynvml with an
  `nvidia-smi` parse fallback. A versioned model-config registry (YAML) with a
  schema, seeded with the DeepSeek V4 flash dspark hero-model entry
  (NVFP4 / vLLM + Q4_K_M / llama.cpp). `zerorun deepseek-v4 --dry-run` prints
  the picked config without launching.
- **m2 — Pick + launch.** Picker scores the GPU probe against the registry
  across 4 hero models (DeepSeek V4, Qwen 3.6, GLM 5.2, Kimi K3) × 2 backends
  (vLLM, llama.cpp). `zerorun <model>` resolves the Hugging Face model id,
  picks quant + backend + chat template + args, and launches the backend
  subprocess streaming output with an OpenAI-compatible endpoint.

### Roadmap (not in v0.1.0)
- **m3 — Staleness guard + ExLlamaV3.** Upstream version-pinning with a drift
  alert, ExLlamaV3 as a third backend, and a community-PR flow for the
  registry.
