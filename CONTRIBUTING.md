# Contributing to ZeroRun

ZeroRun's moat is the **versioned model-config registry** — the curated
`registry/<alias>.yaml` entries that bundle the quant formats, backends,
VRAM/compute-capability floors, launch args, and chat-template references a
user would otherwise spend hours re-deriving from forum posts. The CLI alias a
user types (`zerorun deepseek-v4`) is just the filename stem, so **adding or
fixing a model is a no-code PR** — drop a YAML file, no Python change required.

## Adding a new hero model

1. Create `registry/<alias>.yaml` (the alias is the filename stem, lower-case,
   hyphen-separated — e.g. `yi-large.yaml` for `zerorun yi-large`).
2. Fill in the required fields (see `registry/deepseek-v4.yaml` for a commented
   example):

   ```yaml
   model: <full model name>
   hf_id: <org>/<repo>           # the backend auto-downloads weights from here
   description: >
     <one line>
   quants:                        # >= 1 variant; each is a quant+backend pair
     - format: <Q4_K_M | Q8_0 | NVFP4 | FP8 | ...>
       backend: <vllm | llama.cpp>
       min_vram_gb: <number>      # picker's VRAM-fit floor
       compute_cap_min: <number>  # optional arch floor (0 = any arch)
       args: [<backend CLI flags>]
   chat_template_ref: <alias>     # the template the backend should apply
   staleness_pin: "<alias>@<YYYY-MM>"   # the month this config was verified
   ```

3. Validate locally before pushing:

   ```bash
   python scripts/validate_registry.py
   pytest -q
   ```

   `validate_registry.py` loads every entry through the typed schema and exits
   non-zero on any invalid field (missing `hf_id`, unknown backend, non-numeric
   `min_vram_gb`, etc.). The CI workflow (`.github/workflows/registry-lint.yml`)
   runs the same check on every PR touching `registry/`.

## Updating a stale config

If a model's chat template was patched upstream, bump its `staleness_pin` month
to when you re-verified the config (e.g. `glm-5.2@2026-07` → `glm-5.2@2026-09`).
Run `zerorun staleness` to see which entries are flagged stale (pin older than
6 months) — a stale pin means "this saved config may have drifted; check for
an upstream patch".

## Backends

v0.2 ships `vllm` and `llama.cpp`. A `backend:` value outside that set is
rejected by the schema at load time. Adding a new backend is a code change
(extend `VALID_BACKENDS` in `src/zerorun/registry/schema.py`, add its binary +
default port in `src/zerorun/launcher.py`) — open an issue first to discuss
whether it belongs in the launcher.

## Style

- YAML only in `registry/`; no Python edits for model configs.
- Keep `args` to flags the backend actually accepts; comment non-obvious flags.
- One model per file; the filename stem is the CLI alias.

By submitting a PR you agree your contribution is licensed under the project's
Apache 2.0 license (see [LICENSE](./LICENSE)).
