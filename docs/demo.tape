# ZeroRun demo tape (vhs) — the 10-minute happy path, rendered to assets/demo.gif.
#
# Records the core flow with a SIMULATED GPU (no NVIDIA card needed on the CI
# runner): probe a fake 80GB H100, dry-run a DeepSeek-V4 launch to show the
# picked config + endpoint, then list all 4 hero models with their picks.
# `--gpu "<vram>:<compute_cap>"` is labelled (simulated) so the demo is honest.
#
# Render locally:  vhs docs/demo.tape
# CI refreshes it:  .github/workflows/demo.yml (workflow_dispatch, apt vhs).

Output assets/demo.gif
Set FontSize 14
Set Width 1200
Set Height 640
Set Theme "Molokai"
Set Padding 20

# --- 1. probe a simulated 80GB Hopper (H100) --------------------------------
Type "zerorun probe --gpu 80:9.0"
Enter
Sleep 2s

# --- 2. dry-run a DeepSeek-V4 launch: picked config + endpoint, no spawn ------
Type "zerorun deepseek-v4 --dry-run --gpu 80:9.0"
Enter
Sleep 2s

# --- 3. list all 4 hero models + their auto-picked configs --------------------
Sleep 1s
Type "zerorun --list --gpu 80:9.0"
Enter
Sleep 3s
