#!/usr/bin/env bash
# ZeroRun — "this is what using it looks like" in 3 lines.
# Simulates an 80GB Hopper (H100) so it runs anywhere; drop --gpu/--dry-run on a
# real GPU to actually launch the backend.
set -euo pipefail

uv tool install zerorun                       # 1. install (one command)
zerorun deepseek-v4 --dry-run --gpu "80:9.0"  # 2. see the picked config + endpoint
zerorun --list --gpu "80:9.0"                 # 3. all 4 hero models + their picks
