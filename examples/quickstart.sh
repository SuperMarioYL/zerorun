#!/usr/bin/env bash
# ZeroRun — "this is what using it looks like" end to end.
# Simulates an 80GB Hopper (H100) so it runs anywhere; drop --gpu/--dry-run on a
# real GPU to actually launch the backend.
#
# Install note: zerorun is not published to PyPI, so `uv tool install zerorun`
# cannot resolve. Install from the checkout instead — the same path both
# READMEs document.
set -euo pipefail

# 1. install (works from a checkout, or clones one if run from elsewhere)
if [ ! -f pyproject.toml ]; then
  git clone https://github.com/SuperMarioYL/zerorun.git
  cd zerorun
fi
uv venv .venv
uv pip install --python .venv/bin/python -e .

# 2. probe a simulated 80GB Hopper and dry-run a DeepSeek-V4 launch
.venv/bin/zerorun probe --gpu "80:9.0"
.venv/bin/zerorun deepseek-v4 --dry-run --gpu "80:9.0"

# 3. all 4 hero models + their auto-picked configs
.venv/bin/zerorun --list --gpu "80:9.0"
