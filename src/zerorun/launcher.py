"""Launcher — build the backend argv, spawn it, stream output, surface endpoint.

The launcher is the polish-stage glue between the picker's resolved
:class:`~zerorun.picker.Pick` and a real running model. Per the plan:

* ``build_argv`` is a **pure** function: it turns a ``Pick`` into the exact
  ``argv`` list the backend subprocess will be invoked with, including the
  resolved Hugging Face model id (so the backend auto-downloads weights) and
  the registry's per-variant launch args. No I/O — unit-testable.
* :func:`launch` either prints that argv + the OpenAI-compatible endpoint
  (``--dry-run``) or spawns the backend as a child subprocess, streams its
  stdout/stderr, and prints the endpoint banner. It launches **exactly one**
  backend child (the plan's scope guard).

Design notes
------------
* **Honest about the launch surface.** Actually running vLLM / llama.cpp needs
  a fitting GPU, the backend binary on ``$PATH``, and a multi-GB weight
  download. That path is exercised live, not unit-tested. The testable surface
  is ``build_argv`` (argv shape) + ``launch(dry_run=True)`` (the printed plan)
  + ``launch`` with an injected fake runner (no real subprocess).
* **Override-friendly.** A ``--port`` override flows straight through to the
  backend; ``--host`` defaults to ``0.0.0.0`` (bind all interfaces so a remote
  client can reach the endpoint) while the *surfaced* endpoint string always
  uses ``localhost`` so the user can copy-paste it.
* **No weight management.** The launcher never downloads weights itself — it
  delegates to the backend's own HF fetch (``vllm serve <hf_id>`` /
  ``llama-server -m hf://<hf_id>``), per the plan's explicit scope boundary.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

from .picker import Pick

__all__ = [
    "BACKEND_BINARIES",
    "DEFAULT_PORTS",
    "default_port",
    "backend_binary",
    "build_argv",
    "endpoint_url",
    "LaunchError",
    "LaunchResult",
    "launch",
]


#: The executable each backend is invoked as. ``llama.cpp`` ships its OpenAI-
#: compatible server as ``llama-server`` (the modern binary name); the launcher
#: resolves availability via :func:`shutil.which` so a missing binary produces
#: a clear "install X" message rather than a silent spawn failure.
BACKEND_BINARIES: dict[str, str] = {
    "vllm": "vllm",
    "llama.cpp": "llama-server",
}

#: Default serving ports per backend (avoid clashing with common dev ports).
#: vLLM's default is 8000; llama-server's is 8080.
DEFAULT_PORTS: dict[str, int] = {
    "vllm": 8000,
    "llama.cpp": 8080,
}


def default_port(backend: str) -> int:
    """The default serving port for a backend."""
    return DEFAULT_PORTS.get(backend, 8000)


def backend_binary(backend: str) -> str:
    """The executable name a backend is invoked as.

    Raises :class:`KeyError` for an unknown backend (the registry schema already
    restricts backends to ``vllm`` / ``llama.cpp`` at load time, so this only
    fires on a programmatic misuse).
    """
    if backend not in BACKEND_BINARIES:
        raise KeyError(f"unknown backend '{backend}'")
    return BACKEND_BINARIES[backend]


def build_argv(
    pick: Pick,
    *,
    port: Optional[int] = None,
    host: str = "0.0.0.0",
) -> list[str]:
    """Build the exact backend ``argv`` list for a resolved ``Pick``.

    Pure: no I/O, no subprocess. The resolved Hugging Face model id
    (``pick.hf_id``) is passed positionally so the backend fetches weights
    itself; the registry's per-variant ``args`` (kv-cache-dtype, context size,
    gpu layers, ...) are appended verbatim; the serving host/port are pinned
    last so they always apply.

    * **vllm** → ``["vllm", "serve", <hf_id>, *args, "--host", <host>,
      "--port", <port>]`` (``vllm serve`` is the modern OpenAI-compatible
      serving subcommand; vLLM downloads weights from the HF id).
    * **llama.cpp** → ``["llama-server", "-m", "hf://<hf_id>", *args,
      "--host", <host>, "--port", <port>]`` (the ``hf://`` scheme makes
      llama-server fetch the repo's weights from HF; the registry's args
      already carry the per-variant ``-c`` / ``--n-gpu-layers`` flags).
    """
    bin_name = backend_binary(pick.backend)
    chosen_port = port if port is not None else default_port(pick.backend)
    port_str = str(chosen_port)
    base_args = list(pick.args)

    if pick.backend == "vllm":
        argv = [bin_name, "serve", pick.hf_id]
        argv.extend(base_args)
        argv.extend(["--host", host, "--port", port_str])
        return argv

    # llama.cpp — weights via the hf:// scheme so llama-server auto-fetches.
    argv = [bin_name, "-m", f"hf://{pick.hf_id}"]
    argv.extend(base_args)
    argv.extend(["--host", host, "--port", port_str])
    return argv


def endpoint_url(*, host: str = "localhost", port: int) -> str:
    """The OpenAI-compatible endpoint string surfaced to the user.

    Always rendered with ``localhost`` (copy-pasteable for the user) even when
    the backend binds ``0.0.0.0`` — the bind host and the *surfaced* host are
    deliberately separate so a remote client can still reach the endpoint
    through the bind interface.
    """
    display_host = host or "localhost"
    return f"http://{display_host}:{port}/v1"


class LaunchError(RuntimeError):
    """Raised when a real launch cannot proceed (missing binary, bad backend)."""


@dataclass(frozen=True)
class LaunchResult:
    """Outcome of a launch attempt — carries the argv + endpoint for display."""

    argv: tuple[str, ...]
    endpoint: str
    backend: str
    model: str
    returncode: int

    @property
    def dry_run(self) -> bool:
        """True when no subprocess was spawned (the plan was printed only)."""
        return self.returncode == -1


def _resolve_binary_or_raise(backend: str) -> str:
    """Resolve the backend executable on ``$PATH`` or raise a clear message."""
    bin_name = backend_binary(backend)
    found = shutil.which(bin_name)
    if found is None:
        install_hint = _install_hint(backend)
        raise LaunchError(
            f"backend binary '{bin_name}' not found on $PATH. {install_hint}"
        )
    return bin_name


def _install_hint(backend: str) -> str:
    """A one-line install pointer per backend (so the error is actionable)."""
    if backend == "vllm":
        return "Install vLLM: `pip install vllm` (needs a compatible CUDA GPU)."
    if backend == "llama.cpp":
        return (
            "Build llama-server from https://github.com/ggerganov/llama.cpp "
            "and put it on $PATH."
        )
    return "Install the backend and ensure its server binary is on $PATH."


#: Signature of an injectable backend runner (so tests never spawn a process).
#: Receives the full argv + endpoint + a console writer; returns the exit code.
Runner = Callable[[list[str], str, TextIO, TextIO], int]


def _default_runner(
    argv: list[str],
    endpoint: str,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Spawn the backend child, stream its output, wait for it.

    The endpoint banner is printed *before* the backend starts streaming so a
    user sees it immediately even during a long weight download. The child runs
    in the foreground (Ctrl-C stops it). If the child exits non-zero, that code
    is surfaced.
    """
    print(
        f"\n  ZeroRun → launching {argv[0]}\n"
        f"  OpenAI-compatible endpoint: {endpoint}\n"
        f"  (first run downloads weights — a wait, not effort)\n",
        file=stdout,
        flush=True,
    )
    try:
        proc = subprocess.Popen(argv)
    except FileNotFoundError as exc:
        print(f"error: failed to spawn '{argv[0]}': {exc}", file=stderr)
        return 127
    except OSError as exc:
        print(f"error: could not start backend: {exc}", file=stderr)
        return 1
    try:
        return proc.wait()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        return 130


def launch(
    pick: Pick,
    *,
    port: Optional[int] = None,
    host: str = "0.0.0.0",
    dry_run: bool = False,
    runner: Optional[Runner] = None,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> LaunchResult:
    """Launch (or plan) the backend for a resolved ``Pick``.

    Parameters
    ----------
    pick:
        The resolved :class:`~zerorun.picker.Pick` to launch.
    port:
        Override the backend's default serving port. ``None`` →
        :func:`default_port`.
    host:
        Bind host for the backend. Defaults to ``0.0.0.0`` (all interfaces);
        the surfaced endpoint is always rendered with ``localhost``.
    dry_run:
        Print the resolved argv + endpoint **without** spawning anything.
        Never touches ``$PATH`` (no binary check) — so ``--dry-run`` works on
        any machine, matching the plan's "prints the picked config without
        launching" contract.
    runner:
        Injectable backend runner (tests pass a fake; production spawns the
        subprocess via :func:`_default_runner`).
    """
    if not pick.feasible:
        raise LaunchError(
            f"cannot launch infeasible pick: {pick.summary()} "
            "(the GPU does not meet this variant's floors; "
            "use `zerorun probe` to see the shortfall)"
        )

    chosen_port = port if port is not None else default_port(pick.backend)
    argv = build_argv(pick, port=chosen_port, host=host)
    endpoint = endpoint_url(host="localhost", port=chosen_port)

    out: TextIO = stdout if stdout is not None else sys.stdout
    err: TextIO = stderr if stderr is not None else sys.stderr

    if dry_run:
        print(f"\n  model:    {pick.model}", file=out)
        print(f"  hf_id:    {pick.hf_id}", file=out)
        print(f"  quant:    {pick.format} ({pick.backend})", file=out)
        print(f"  template: {pick.chat_template_ref}", file=out)
        print(f"  endpoint: {endpoint}", file=out)
        print(f"  argv:     {' '.join(argv)}", file=out)
        print(file=out)
        for reason in pick.reasons:
            print(f"  · {reason}", file=out)
        return LaunchResult(
            argv=tuple(argv),
            endpoint=endpoint,
            backend=pick.backend,
            model=pick.model,
            returncode=-1,
        )

    # Real launch: the binary must actually exist on this machine.
    _resolve_binary_or_raise(pick.backend)
    run = runner if runner is not None else _default_runner
    rc = run(argv, endpoint, out, err)
    return LaunchResult(
        argv=tuple(argv),
        endpoint=endpoint,
        backend=pick.backend,
        model=pick.model,
        returncode=rc,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from .picker import Pick
    from .registry.schema import QuantConfig

    _q = QuantConfig(
        format="Q4_K_M",
        backend="llama.cpp",
        min_vram_gb=32,
        compute_cap_min=7.5,
        args=("-c", "8192", "--n-gpu-layers", "999"),
    )
    _pick = Pick(
        registry_key="deepseek-v4",
        model="deepseek-v4-flash-dspark",
        hf_id="deepseek-ai/DeepSeek-V4-flash-dspark",
        quant=_q,
        backend="llama.cpp",
        chat_template_ref="deepseek-v4",
        args=_q.args,
        score=0.0,
        feasible=True,
    )
    launch(_pick, port=8080, dry_run=True)
