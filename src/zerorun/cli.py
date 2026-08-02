"""ZeroRun CLI — one command to probe, pick, and launch a frontier MoE.

Surface (per the plan):

* ``zerorun probe``            — print detected GPU + which registry models fit.
* ``zerorun --list``           — list every registry model + its picked config + staleness.
* ``zerorun deepseek-v4``      — probe, pick, launch the backend, print endpoint.
* ``zerorun deepseek-v4 --dry-run``        — print the picked config, no launch.
* ``zerorun deepseek-v4 --backend vllm``  — override the chosen backend.
* ``zerorun deepseek-v4 --quant Q8_0``      — override the chosen quant format.
* ``zerorun deepseek-v4 --gpu "80:9.0"``    — simulate an H100 (no NVIDIA card
  needed — demos the picker logic on any machine; the probe is labelled
  ``(simulated)``).
* ``zerorun staleness``        — per-model staleness report (drift alert vs the
  registry's ``staleness_pin``).

Routing: a single Typer command (no group, no COMMAND slot) inspects the
first positional. ``probe`` and ``staleness`` are reserved verbs (dispatched
to the GPU-only report and the staleness report respectively); ``--list`` lists
the registry; any other token is a registry key launch. Typer collapses a one-command app to a single-command CLI with no
command-name prefix, so ``zerorun deepseek-v4 --dry-run`` parses as a plain
positional + option (a group would mis-route ``--dry-run`` as a subcommand).
Every hero-model alias (``deepseek-v4``, ``qwen-3.6``, ``glm-5.2``,
``kimi-k3``) is a bare positional, so the launch path is the default. Rich
renders all tables/panels so the output matches the hero demo the README
promises.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .picker import Pick, best_pick_per_model, pick, select
from .probe import GPUProbe, arch_family_from_compute_cap, probe_gpu
from .registry.schema import (
    ModelEntry,
    default_registry_dir,
    list_registry_keys,
    load_registry_dir,
)
from .staleness import (
    STALENESS_VERB,
    staleness_report_for_entries,
)

# Re-exported so tests import the launcher through the cli package surface.
from .launcher import LaunchError, launch  # noqa: F401

#: The verb that selects the GPU-only report (reserved — not a model alias).
PROBE_VERB = "probe"

app = typer.Typer(
    name="zerorun",
    help=(
        "ZeroRun — probe your GPU, pick the best quant + backend + chat "
        "template for a frontier Chinese open-weight MoE, and just run it. "
        "5h-per-swap config gauntlet → one command."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

_console = Console()
_err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gpu_spec(spec: str) -> GPUProbe:
    """Parse a ``--gpu "VRAM:compute_cap"`` simulation spec into a GPUProbe.

    Lets the picker be demonstrated on any machine (CI, no-NVIDIA dev box).
    ``80:9.0`` → a simulated 80 GB Hopper card (H100). The probe is labelled
    ``source="simulated"`` so the output is honest about its origin.
    """
    if not spec:
        raise typer.BadParameter(
            "--gpu needs a '<vram_gb>:<compute_cap>' spec, e.g. '80:9.0' (H100)."
        )
    parts = spec.split(":")
    if len(parts) != 2:
        raise typer.BadParameter(
            "--gpu must be '<vram_gb>:<compute_cap>', e.g. '80:9.0' or '24:8.9'."
        )
    try:
        vram = float(parts[0].strip())
        cc = float(parts[1].strip())
    except ValueError:
        raise typer.BadParameter(
            "--gpu values must be numbers, e.g. '80:9.0'."
        )
    if vram <= 0 or cc <= 0:
        raise typer.BadParameter("--gpu values must be positive.")
    return GPUProbe(
        vram_gb=vram,
        compute_cap=cc,
        arch_family=arch_family_from_compute_cap(cc),
        name=f"simulated-{vram:.0f}gb-cc{cc}",
        source="simulated",
    )


def _resolve_probe(
    gpu_spec: Optional[str], no_probe: bool
) -> GPUProbe:
    """Build the GPUProbe for a command honoring --gpu / --no-probe / hardware."""
    if gpu_spec is not None:
        return _parse_gpu_spec(gpu_spec)
    if no_probe:
        return GPUProbe()
    return probe_gpu()


def _load_registry(registry_dir: Optional[str]):
    """Load every registry entry; print a friendly error if it is missing.

    Returns ``(entries_dict, registry_keys)``. On a missing/invalid registry
    the user gets a hint to set ``ZERORUN_REGISTRY_DIR`` rather than a traceback.
    """
    reg_dir = registry_dir or str(default_registry_dir())
    try:
        entries = load_registry_dir(reg_dir)
    except Exception as exc:
        _err_console.print(
            f"[red]error:[/red] could not load registry from {reg_dir!r}: {exc}"
        )
        _err_console.print(
            "hint: set $ZERORUN_REGISTRY_DIR to a registry directory, or run "
            "from a checkout that ships a registry/ folder."
        )
        raise typer.Exit(code=2)
    keys = list_registry_keys(reg_dir)
    return entries, keys


def _render_probe(probe: GPUProbe, entries: dict[str, ModelEntry]) -> None:
    """Print the GPU probe + a per-model fit table (backs `zerorun probe`)."""
    _console.print(
        Panel.fit(
            Text.assemble(
                ("GPU: ", "bold"),
                (probe.name or "—", "cyan"),
                ("   VRAM: ", "bold"),
                (f"{probe.vram_gb:.0f} GB", "cyan"),
                ("   Compute-cap: ", "bold"),
                (f"{probe.compute_cap}", "cyan"),
                ("   Arch: ", "bold"),
                (probe.arch_family, "cyan"),
                ("   (source: ", "dim"),
                (probe.source, "dim"),
                (")", "dim"),
            ),
            title="zerorun probe",
            border_style="blue",
        )
    )

    if not entries:
        _console.print("[dim]no registry entries loaded.[/dim]")
        return

    table = Table(
        title="Registry fit", show_header=True, header_style="bold blue"
    )
    table.add_column("model", style="cyan", no_wrap=True)
    table.add_column("best pick", overflow="fold")
    table.add_column("score", justify="right")
    table.add_column("fit", justify="center")

    ranked = best_pick_per_model(probe, entries)
    for key in sorted(entries):
        entry = entries[key]
        chosen = ranked[key]
        if chosen is not None:
            table.add_row(
                key,
                f"{chosen.format} / {chosen.backend}",
                f"{chosen.score:.3f}",
                "[green]✓[/green]",
            )
        else:
            # No feasible variant — show the nearest miss for honesty.
            ranked_all = pick(probe, entry)
            nearest = ranked_all[0] if ranked_all else None
            if nearest is not None:
                table.add_row(
                    key,
                    f"{nearest.format} / {nearest.backend}",
                    "—",
                    "[red]✗[/red]",
                )
            else:
                table.add_row(key, "—", "—", "[red]✗[/red]")
    _console.print(table)


def _render_list(probe: GPUProbe, entries: dict[str, ModelEntry]) -> None:
    """Print the registry + picked config per model (backs `zerorun --list`)."""
    if not probe.has_gpu:
        _console.print(
            "[yellow]no NVIDIA GPU detected — showing registry entries with no "
            "feasible pick. Run `zerorun probe` for details.[/yellow]\n"
        )

    table = Table(
        title=f"registry ({len(entries)} models)", show_header=True,
        header_style="bold blue",
    )
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("model", overflow="fold")
    table.add_column("pick", overflow="fold")
    table.add_column("args", overflow="fold", max_width=42)
    table.add_column("staleness", justify="center")
    table.add_column("fit", justify="center")

    ranked = best_pick_per_model(probe, entries)
    reports = staleness_report_for_entries(entries)
    for key in sorted(entries):
        entry = entries[key]
        chosen = ranked[key]
        rep = reports[key]
        if rep.status == "stale":
            stale_cell = "[red]stale[/red]"
        elif rep.status == "fresh":
            stale_cell = "[green]fresh[/green]"
        else:
            stale_cell = "[dim]?[/dim]"
        if chosen is not None:
            table.add_row(
                key,
                entry.model,
                f"{chosen.format} / {chosen.backend}",
                " ".join(chosen.launch_args),
                stale_cell,
                "[green]✓[/green]",
            )
        else:
            table.add_row(
                key, entry.model, "no fit on this GPU", "—",
                stale_cell, "[red]✗[/red]",
            )
    _console.print(table)


def _render_staleness(entries: dict[str, ModelEntry]) -> None:
    """Print a per-model staleness report (backs `zerorun staleness`)."""
    _console.print(
        Panel.fit(
            Text.assemble(
                ("staleness guard — ", "bold"),
                ("registry pin drift alert (", "dim"),
                ("fresh", "green"),
                (" <= 6mo / ", "dim"),
                ("stale", "red"),
                (" > 6mo / ", "dim"),
                ("?", "dim"),
                (" unknown)", "dim"),
            ),
            title="zerorun staleness",
            border_style="blue",
        )
    )
    if not entries:
        _console.print("[dim]no registry entries loaded.[/dim]")
        return

    table = Table(
        title="staleness report", show_header=True, header_style="bold blue"
    )
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("model", overflow="fold")
    table.add_column("pin", overflow="fold")
    table.add_column("status", justify="center")
    table.add_column("age", justify="right")
    table.add_column("message", overflow="fold")

    reports = staleness_report_for_entries(entries)
    for key in sorted(entries):
        rep = reports[key]
        if rep.status == "stale":
            status_cell = "[red]stale[/red]"
        elif rep.status == "fresh":
            status_cell = "[green]fresh[/green]"
        else:
            status_cell = "[dim]unknown[/dim]"
        table.add_row(
            key,
            rep.model,
            rep.pin or "—",
            status_cell,
            f"{rep.age_months}mo" if rep.age_months is not None else "—",
            rep.message,
        )
    _console.print(table)
    stale = [k for k, r in reports.items() if r.is_stale]
    if stale:
        _console.print(
            f"[red]{len(stale)} stale:[/red] {', '.join(sorted(stale))} — "
            "an upstream chat-template patch may postdate the pin; "
            "check for a newer registry entry."
        )
    else:
        _console.print("[green]no stale pins.[/green]")


def _resolve_pick(
    probe: GPUProbe,
    entry: ModelEntry,
    backend: Optional[str],
    quant_format: Optional[str],
) -> Pick:
    """Resolve the pick for a launch, honoring overrides + reporting shortfalls.

    Returns the chosen :class:`Pick` or exits with a clear message:
      * override not feasible on this GPU → exit 1 with the shortfall reasons;
      * no feasible variant at all → exit 1 with the nearest miss reasons.
    """
    chosen = select(probe, entry, backend=backend, quant_format=quant_format)
    if chosen is not None:
        return chosen

    ranked = pick(probe, entry)
    nearest = ranked[0] if ranked else None
    _err_console.print(
        f"[red]no feasible pick[/red] for [cyan]{entry.registry_key}[/cyan] "
        f"on this GPU (VRAM {probe.vram_gb:.0f} GB, compute-cap "
        f"{probe.compute_cap}, arch {probe.arch_family})."
    )
    if backend or quant_format:
        _err_console.print(
            f"your override (backend={backend!r}, quant={quant_format!r}) "
            "does not fit this GPU."
        )
    if nearest is not None:
        for reason in nearest.reasons:
            _err_console.print(f"  · {reason}")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Single command — probe / list / launch
# ---------------------------------------------------------------------------

@app.command()
def main(
    ctx: typer.Context,
    model: str = typer.Argument(
        None,
        help=(
            "registry key to launch (e.g. `deepseek-v4`), or `probe` for the "
            "GPU-only report. With no model and no --list, prints help."
        ),
        metavar="MODEL",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="print the picked config (quant/backend/template/argv/endpoint) "
             "without launching the backend.",
    ),
    backend: str = typer.Option(
        None, "--backend", "-b",
        help="override the chosen backend (vllm | llama.cpp).",
    ),
    quant_format: str = typer.Option(
        None, "--quant", "-q",
        help="override the chosen quant format, e.g. Q4_K_M / NVFP4 / Q8_0.",
    ),
    list_models: bool = typer.Option(
        False, "--list",
        help="list every registry model + its auto-picked config, then exit.",
    ),
    registry_dir: str = typer.Option(
        None, "--registry-dir",
        help="override the registry directory (default: $ZERORUN_REGISTRY_DIR "
             "or ./registry).",
    ),
    port: int = typer.Option(
        None, "--port", "-p",
        help="override the backend serving port (default: 8000 vllm / 8080 llama.cpp).",
    ),
    no_probe: bool = typer.Option(
        False, "--no-probe",
        help="skip the GPU probe (use a zeroed probe — every variant is infeasible).",
    ),
    gpu_spec: str = typer.Option(
        None, "--gpu",
        help=(
            "simulate a GPU for demos/tests, format '<vram_gb>:<compute_cap>' "
            "(e.g. '80:9.0' = H100, '24:8.9' = RTX 4090). The probe is labelled "
            "(simulated). Lets the picker run on any machine."
        ),
        metavar="VRAM:CC",
    ),
) -> None:
    """Probe your GPU, pick the best quant+backend+template, and just run it."""
    entries, _keys = _load_registry(registry_dir)

    # --- `zerorun --list` ----------------------------------------------------
    if list_models:
        probe = _resolve_probe(gpu_spec, no_probe)
        if not probe.has_gpu and not no_probe and gpu_spec is None:
            _err_console.print(
                "[yellow]no NVIDIA GPU detected[/yellow] — variants will be "
                "infeasible. Pass --gpu '80:9.0' to simulate one for inspection."
            )
        _render_list(probe, entries)
        raise typer.Exit(code=0)

    # --- `zerorun probe` (reserved verb → GPU-only report) -------------------
    if model == PROBE_VERB:
        probe = _resolve_probe(gpu_spec, no_probe)
        _render_probe(probe, entries)
        raise typer.Exit(code=0)

    # --- `zerorun staleness` (reserved verb → per-model staleness report) ----
    if model == STALENESS_VERB:
        _render_staleness(entries)
        raise typer.Exit(code=0)

    # --- `zerorun <model>` (default launch path) -----------------------------
    if model is None:
        # Nothing to do — print help (matches no_args_is_help behavior).
        _console.print(ctx.get_help())
        raise typer.Exit(code=0)

    probe = _resolve_probe(gpu_spec, no_probe)
    if not probe.has_gpu and not no_probe and gpu_spec is None:
        _err_console.print(
            "[yellow]no NVIDIA GPU detected[/yellow] — variants will be "
            "infeasible. Pass --gpu '80:9.0' to simulate one for inspection."
        )

    if model not in entries:
        avail = ", ".join(sorted(entries)) or "(empty registry)"
        _err_console.print(
            f"[red]unknown model[/red] [cyan]{model}[/cyan]. "
            f"known models: {avail}"
        )
        raise typer.Exit(code=2)

    entry = entries[model]
    chosen = _resolve_pick(probe, entry, backend, quant_format)
    try:
        result = launch(chosen, port=port, dry_run=dry_run)
    except LaunchError as exc:
        _err_console.print(f"[red]launch error:[/red] {exc}")
        raise typer.Exit(code=1)

    if dry_run:
        _console.print(
            "[green]dry-run ok[/green] — re-run without --dry-run to launch."
        )
    elif result.returncode != 0:
        raise typer.Exit(code=result.returncode)


# Entry point shim for `python -m zerorun`.
def _run() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    _run()
