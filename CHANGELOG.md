# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
