**English** | [简体中文](./README.md)

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
    <img src="./assets/hero-light.svg" width="880" alt="ZeroRun — probe your GPU, pick quant+backend+template, launch in one command">
  </picture>
</div>

<p align="center"><sub>ZeroRun is a one-command launcher: it probes your GPU, auto-picks the best quant + backend + chat template for any frontier Chinese open-weight MoE, and just runs it — collapsing a community-verified "5 hours of config per model swap" into a single command.</sub></p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license"></a>
  <a href="https://github.com/SuperMarioYL/zerorun/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/zerorun?color=blue" alt="latest release"></a>
  <a href="https://github.com/SuperMarioYL/zerorun/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/zerorun/ci.yml?branch=main&label=CI" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/backends-vLLM%20%7C%20llama.cpp-5E5CE6" alt="backends">
</p>

---

> **5 hours → 1 command.** Turn vLLM's flag hell into a single `zerorun <model>` that picks the right config for you.

## <img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Architecture

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
    <img src="./assets/atlas-light.svg" width="880" alt="Architecture: zerorun CLI → probe GPU × versioned config registry → picker Pick → launcher spawns vllm|llama.cpp → OpenAI-compatible endpoint">
  </picture>
</div>

One process, no microservices, no k8s. ZeroRun is a thin Python orchestrator that shells out to the chosen backend as a child subprocess:

- `cli.py` — parses `zerorun <model> [--dry-run] [--backend X] [--quant Y] [--gpu "80:9.0"]`
- `probe.py` — pynvml probes VRAM / compute capability / arch family, with an `nvidia-smi` parse fallback
- `picker.py` — a **pure function** `(GPUProbe, ModelEntry) -> ranked Picks`, no I/O, unit-testable
- `launcher.py` — builds the backend argv, spawns the child, streams output, prints the OpenAI-compatible endpoint
- `registry/*.yaml` — **data, not code**; community-PR-able; the moat lives here

## <img src="https://api.iconify.design/tabler:bulb.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Why

Every time you swap to a new Chinese open-weight MoE — DeepSeek V4 flash dspark, Qwen 3.6 A3B, GLM 5.2, Kimi K3 — you re-solve the same combinatorial puzzle: which quant format (Q4_K_M / Q8 / NVFP4 / ternary), which backend (vLLM / llama.cpp / ExLlamaV3), and which chat template, hand-written and then silently going stale the moment upstream patches it. One redditter reported "only 5 hours configuring VLLM to run deepseek v4 flash dspark" for a single swap, and the swap frequency itself is rising as the labs release near-monthly. ZeroRun introduces a new verb, *zero-run*: one command that probes your GPU + VRAM, picks the best quant + backend + template for the named model, and just runs it with sane defaults. Model-release velocity stops compounding into your personal time-debt.

## <img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Install & Quickstart

```bash
# 1. One-command install (uv preferred; pipx / pip -e . equivalent)
uv tool install zerorun
# or local dev:
git clone https://github.com/SuperMarioYL/zerorun && cd zerorun && uv sync

# 2. Probe your GPU (VRAM / compute-cap / arch family + which models fit)
zerorun probe

# 3. One command to pick the config + launch the backend (first run downloads weights — a wait, not effort)
zerorun deepseek-v4
# → prints the OpenAI-compatible endpoint http://localhost:8000/v1; point any client at it
```

<details><summary>No NVIDIA GPU? Use <code>--gpu</code> to simulate a card and watch the picker (10-second visible result)</summary>

```bash
# Simulate an 80GB Hopper (H100) — the picker picks DeepSeek's Q8_0/llama.cpp
zerorun deepseek-v4 --dry-run --gpu "80:9.0"
# Simulate a 24GB Ada (RTX 4090) — only the smallest Qwen 3.6 A3B fits
zerorun --list --gpu "24:8.9"
```

`--gpu "<vram_gb>:<compute_cap>"` makes the picker demonstrable on any machine; the probe is labelled `(simulated)`.

</details>

## <img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Usage

```bash
# List every hero model + its auto-picked config on your GPU
zerorun --list

# Print the picked config only (quant/backend/template/argv/endpoint) — no launch
zerorun deepseek-v4 --dry-run --gpu "80:9.0"

# Override the picker's choice — but the GPU must fit it, or you get the exact shortfall
zerorun deepseek-v4 --backend vllm --quant NVFP4 --gpu "80:9.0"
zerorun kimi-k3 --backend llama.cpp --quant Q4_K_M --port 8080

# Four hero models, one command each (all supported at v0.1)
zerorun deepseek-v4   # the 5h → 1-command model
zerorun qwen-3.6      # the small A3B — fits a single consumer GPU
zerorun glm-5.2       # the template-staleness story
zerorun kimi-k3       # the heaviest — the multi-backend pick matters most here
```

The picking heuristic is **documented**: accuracy (preserved precision) is the primary axis; backend throughput and VRAM headroom only break near-ties. `--dry-run` prints the arithmetic for every term, and `registry/*.yaml` is readable YAML you can PR — you don't have to trust my picks blind.

<details><summary>The score formula (<code>picker.score_quant</code>)</summary>

```
score = accuracy_score(format)              # 0.60–1.00, primary axis
      + backend_throughput(backend, arch)   # 0.00–0.15, tiebreak
      + vram_safety(vram, min_vram)         # 0.00–0.08, tiebreak
```
- accuracy ordering follows the GGUF hierarchy: `F16 > Q8_0 > Q6_K > Q4_K_M > NVFP4 > Q3 > Q2`; NVFP4 (W4A4) ranks below Q4_K_M.
- vLLM earns its full 0.15 on Hopper+ (the FA-4 / tensor-core path); llama.cpp earns a flat 0.05 broad-fit bonus.
- Infeasible variants score negatively (ranked by "how close to feasible") and are still listed for diagnostics — you see "16 GB short", not a bare "won't run".

</details>

## <img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Demo

The happy path `zerorun probe` → `zerorun deepseek-v4 --dry-run --gpu "80:9.0"` → `zerorun --list` (CI re-renders it on tag via vhs):

![demo](assets/demo.gif)

## <img src="https://api.iconify.design/tabler:adjustments.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Configuration — the versioned model registry

The moat is a **YAML registry**, not code. Each entry bundles what is today scattered across forum posts and maintainer memory into a machine-readable, hardware-aware config:

```yaml
# registry/deepseek-v4.yaml — data, community-PR-able; CLI alias = filename stem
model: deepseek-v4-flash-dspark
hf_id: deepseek-ai/DeepSeek-V4-flash-dspark   # backend auto-downloads weights
quants:
  - format: NVFP4          # quant format
    backend: vllm          # backend
    min_vram_gb: 48        # picker's VRAM floor
    compute_cap_min: 9.0   # picker's arch floor (optional, 0 = any)
    args: ["--kv-cache-dtype", "fp8", "--tensor-parallel-size", "1"]
  - format: Q4_K_M
    backend: llama.cpp
    min_vram_gb: 32
    compute_cap_min: 7.5
    args: ["-c", "8192", "--n-gpu-layers", "999"]
chat_template_ref: deepseek-v4
staleness_pin: "deepseek-v4@2026-07"   # m3: drift alert vs upstream
```

| field | type | default | meaning |
|---|---|---|---|
| `model` | str | — | full name (display) |
| `hf_id` | str | — | Hugging Face model id; backend fetches weights from it |
| `quants[].format` | str | — | quant format (NVFP4 / Q4_K_M / Q8_0 / FP8 …) |
| `quants[].backend` | str | — | `vllm` or `llama.cpp` (ExLlamaV3 is m3) |
| `quants[].min_vram_gb` | float | — | picker's VRAM-fit floor |
| `quants[].compute_cap_min` | float | 0 | compute-cap floor; 0 = any arch |
| `quants[].args` | list[str] | `[]` | per-variant backend argv flags |
| `chat_template_ref` | str | — | chat-template reference |
| `staleness_pin` | str | — | version pin (m3 drift alert) |

Fork your own registry: set `ZERORUN_REGISTRY_DIR=/path/to/your/registry`, or pass `--registry-dir`.

## <img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Roadmap

- [x] **m1 — GPU probe + registry schema + 1 hero model** (DeepSeek V4 dspark; `probe` + `--dry-run`)
- [x] **m2 — pick + launch** (4 hero models × 2 backends; picker scores + launcher spawns the backend + prints the OpenAI endpoint)
- [ ] **m3 — staleness guard + ExLlamaV3** (upstream version pin + drift alert; a third backend ExLlamaV3; community-PR registry flow)

v0.1 ships vLLM + llama.cpp only. No Web/TUI, no quantization itself, no hosted SaaS — ZeroRun picks existing quants, it never produces them.

## How it differs

| capability | ZeroRun | [ollama](https://github.com/ollama/ollama) | [vLLM](https://github.com/vllm-project/vllm) | [llama.cpp](https://github.com/ggerganov/llama.cpp) |
|---|:---:|:---:|:---:|:---:|
| one-command launch | ✓ | ✓ | — | — |
| multi-backend auto-pick (vLLM + llama.cpp) | ✓ | — | — | — |
| stays current on frontier quants (NVFP4 / FP8 / ternary) | ✓ | partial (lags) | ✓ | partial |
| versioned, hardware-aware model-config registry | ✓ | — | — | — |
| chat-template staleness alert (m3) | ✓ | — | — | — |
| sane defaults, works out of the box | ✓ | ✓ | — (expert-gated) | — (per-model hand-tune) |

Honest: ollama does the zero-config job excellently for the formats it supports — its single-backend philosophy *is* its value proposition; multi-backend auto-pick breaks that model and it has no incentive to carry the quant-research maintenance burden. ZeroRun is the frontier-Chinese-MoE-config specialization layer. vLLM has higher throughput and ollama a wider ecosystem — ZeroRun doesn't compete on performance, it competes on the per-swap configuration cost.

## <img src="https://api.iconify.design/tabler:license.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> License & Contributing

MIT, see [LICENSE](./LICENSE). Dev-audience, OSS, no paywalled feature at v0.1. Issues and PRs welcome — adding a quant floor or fixing a chat-template reference is a one-line `registry/*.yaml` change, no Python required.

## Share this

```
ZeroRun — 5h → 1 command: probe your GPU, auto-pick quant+backend+template for frontier Chinese open-weight MoEs, and just run it. OSS, Python, vLLM+llama.cpp. https://github.com/SuperMarioYL/zerorun
```

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
