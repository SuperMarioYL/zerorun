# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-09-17

A fix-led hardening release — four reproduced defects fixed and the version
lockstep extended, with no stack, audience, or distribution changes. No feature
milestone (zero organic issues/PRs at grill time).

### Fixed
- **fix-cli-version-flag** (MED, `src/zerorun/cli.py`). `zerorun --version`
  errored with "No such option: --version" (rc=2) — the CLI could not report
  its own version even though v0.4.0 locked `zerorun.__version__` against the
  VERSION file. A `--version` flag now prints `zerorun <version>` and exits 0,
  sourcing the lockstep-guarded `__version__` attribute.
  `tests/test_cli.py` covers the flag.
- **fix-override-diagnostics** (MED, `src/zerorun/cli.py` `_resolve_pick`).
  An invalid `--backend`/`--quant` override produced a FALSE "does not fit
  this GPU" diagnostic (e.g. `--backend exllama` — a backend the product does
  not ship — was reported as a hardware shortfall, with an unrelated variant's
  reasons attached). Unknown values now exit 2 before the GPU-fit ranking,
  naming the valid backends or the entry's available formats; a REAL variant
  that is infeasible keeps the honest shortfall path (exit 1 + "does not fit
  this GPU" + VRAM-short reasons). `tests/test_cli.py` pins all three
  branches (red-before: 3 failed / 1 passed on the shipped v0.4.0 code).
- **fix-registry-lint-ci-deps** (HIGH, `.github/workflows/registry-lint.yml`).
  The registry-lint workflow had never run green: the only recorded run
  (32620816317, 2026-08-23) failed at the validator step, which ran with no
  dependency install (`import yaml` → ModuleNotFoundError on a clean runner),
  and the pytest step's `pip install -e ".[dev]"` referenced a nonexistent
  extra (dev deps are a PEP 735 `[dependency-groups]` table) with stderr
  silenced, so pytest was never installed. The workflow now mirrors the green
  `ci.yml` flow (`uv sync --dev` + `uv run`), so the m5 community-PR gate
  actually validates registry PRs.
- **quality-pynvml-to-nvidia-ml-py** (LOW-MED, `pyproject.toml`). The
  deprecated `pynvml` PyPI package printed its own FutureWarning to stderr on
  every CLI invocation that probes. The dependency is now `nvidia-ml-py>=12.0`
  (the maintained NVIDIA binding it names as its replacement, shipping the
  same `pynvml` module) — no source change, verified warning-free import.

### Added
- **quality-version-lockstep-site-changelog** (MED,
  `tests/test_version.py`). The v0.4.0 lockstep test guarded only VERSION ==
  pyproject == `__version__`; two release surfaces moved unwatched. The test
  now also asserts `web/site.json`'s `content_version` equals the VERSION
  file (site drift already required a dedicated repair commit once) and that
  `CHANGELOG.md` carries the release's `## [x.y.z]` heading. Both checks skip
  cleanly when their file is absent (sdist layout).

## [0.4.0] - 2026-09-02

A clean bugfix release — a single verified correctness fix to the in-source
version surface. No stack, audience, distribution, or `target_files` changes;
no feature milestone. Built directly into the existing
`workspace/builds/zerorun--zerorun8` (iteration commit, not a re-scaffold).

### Fixed
- **fix-init-version-drift** (MED-HIGH, `src/zerorun/__init__.py:7`). The
  `__version__` attribute was `0.1.0` — two minor versions behind the
  canonical `VERSION` file (0.3.0) and `pyproject.toml` (0.3.0) — because the
  v0.2.0 and v0.3.0 release bumps each touched only `VERSION` +
  `pyproject.toml` + `CHANGELOG.md` and never propagated to
  `__init__.py.__version__`, so `import zerorun; zerorun.__version__` reported
  `0.1.0` against a v0.3.0 release (the package mis-reports its own version).
  `__version__` is now `0.4.0`, and `tests/test_version.py` is a lockstep
  regression asserting the `VERSION` file, the `pyproject.toml` `version`
  field, and `zerorun.__version__` all agree — so the next bump cannot silently
  leave `__version__` stale (the exact recurrence that produced this drift
  across v0.2.0 and v0.3.0).

## [0.3.0] - 2026-08-23

A clean bugfix release — two verified correctness fixes to the dominant launch
path and the m5 registry validator. No stack, audience, distribution, or
`target_files` changes; no feature milestone. Built directly into the existing
`workspace/builds/zerorun--zerorun8` (iteration commit, not a re-scaffold).

### Fixed
- **fix-llamacpp-hf-uri-launch** (HIGH, `src/zerorun/launcher.py:122`). The
  llama.cpp launch argv used `[llama-server, "-m", "hf://<hf_id>"]`, but
  llama-server's `-m`/`--model` expects a *local file path* and the `hf://`
  scheme is unsupported — so every llama.cpp pick (the dominant path: all 4
  hero models on A100-80G, deepseek-v4 + glm-5.2 on H100-80G, and qwen-3.6 on
  every consumer card including the showcase RTX 4090) exited non-zero before
  the OpenAI-compatible endpoint came up. `build_argv` now emits
  `["llama-server", "--hf-repo", "<hf_id>:<format>"]`, where `--hf-repo`
  fetches the matching GGUF from Hugging Face and the `:<format>` suffix
  selects the quant (Q4_K_M / Q8_0) by the pick's own registry `format` label
  — no schema change. The vLLM path (`vllm serve <hf_id>`, already correct)
  is unchanged. `tests/test_launcher.py` asserts `--hf-repo <hf_id>:<format>`
  and the absence of any `-m`/`hf://` value.
- **fix-registry-args-str-coercion** (MED, `src/zerorun/registry/schema.py:111-116`).
  `QuantConfig.from_dict` validated `args` only as a list, then coerced every
  element via `str()` — so `args=[None]`/`[123]`/`[['nested']]`/`[{'k':'v'}]`
  all passed despite the "must be a list of strings" message, undermining the
  m5 community-PR validator contract (a stray null would pass CI and ship a
  broken argv such as a literal `"None"` token forwarded to vllm/llama-server).
  `from_dict` now type-checks each element and raises `RegistryError` for any
  non-string, reusing the existing "list of strings" message.
  `scripts/validate_registry.py` therefore rejects such an entry (exit 1).
  `tests/test_schema.py` covers `args=[None]`/`[123]`/`[['nested']]`/`[{'k':'v'}]`
  rejection plus a clean string list passing, and an integration test verifies
  `validate_registry.py` exits non-zero on a non-string `args` entry.

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
