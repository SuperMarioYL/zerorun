"""Unit tests for the launcher's ``build_argv`` (v0.3.0 fix-llamacpp-hf-uri-launch).

The llama.cpp launch path must use llama-server's ``--hf-repo <hf_id>:<format>``
flag, which fetches the matching GGUF from Hugging Face and selects the quant by
the pick's own registry ``format`` label — never the unsupported ``-m hf://``
scheme (llama-server's ``-m``/``--model`` expects a local file, so ``hf://``
made every llama.cpp pick exit non-zero before the OpenAI-compatible endpoint
came up). The vLLM path is unchanged (``vllm serve <hf_id>`` auto-fetches).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make src/ importable when tests run without an editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from zerorun.launcher import (  # noqa: E402
    build_argv,
    default_port,
    endpoint_url,
)
from zerorun.picker import Pick  # noqa: E402
from zerorun.registry.schema import QuantConfig  # noqa: E402


def _llama_pick(
    hf_id: str = "qwen/Qwen3.6-A3B",
    fmt: str = "Q4_K_M",
    args=("-c", "8192", "--n-gpu-layers", "999"),
) -> Pick:
    """A feasible llama.cpp pick mirroring registry/qwen-3.6.yaml's Q4_K_M."""
    q = QuantConfig(
        format=fmt, backend="llama.cpp", min_vram_gb=12,
        compute_cap_min=7.5, args=tuple(args),
    )
    return Pick(
        registry_key="qwen-3.6", model="qwen3.6-a3b", hf_id=hf_id,
        quant=q, backend="llama.cpp", chat_template_ref="qwen-3.6",
        args=q.args, score=0.9, feasible=True,
    )


def _vllm_pick(
    hf_id: str = "deepseek-ai/DeepSeek-V4-flash-dspark",
    fmt: str = "NVFP4",
    args=("--kv-cache-dtype", "fp8"),
) -> Pick:
    """A feasible vLLM pick mirroring registry/deepseek-v4.yaml's NVFP4."""
    q = QuantConfig(
        format=fmt, backend="vllm", min_vram_gb=48,
        compute_cap_min=9.0, args=tuple(args),
    )
    return Pick(
        registry_key="deepseek-v4",
        model="deepseek-v4-flash-dspark", hf_id=hf_id,
        quant=q, backend="vllm", chat_template_ref="deepseek-v4",
        args=q.args, score=0.9, feasible=True,
    )


class TestBuildArgvLlamaCpp(unittest.TestCase):
    """v0.3.0 fix-llamacpp-hf-uri-launch: the llama.cpp argv must launch."""

    def test_uses_hf_repo_with_format_suffix(self):
        argv = build_argv(_llama_pick(fmt="Q4_K_M"))
        self.assertIn("--hf-repo", argv)
        idx = argv.index("--hf-repo")
        # the :<format> suffix selects the matching GGUF (Q4_K_M / Q8_0).
        self.assertEqual(argv[idx + 1], "qwen/Qwen3.6-A3B:Q4_K_M")

    def test_never_uses_m_hf_uri(self):
        argv = build_argv(_llama_pick())
        self.assertNotIn("-m", argv)
        self.assertFalse(
            any(a.startswith("hf://") for a in argv),
            f"argv must not carry an hf:// value: {argv}",
        )

    def test_format_suffix_tracks_pick_format(self):
        for fmt in ("Q4_K_M", "Q8_0"):
            argv = build_argv(_llama_pick(fmt=fmt))
            idx = argv.index("--hf-repo")
            self.assertTrue(argv[idx + 1].endswith(f":{fmt}"))

    def test_appends_registry_args_and_host_port(self):
        argv = build_argv(_llama_pick(), port=8080)
        self.assertEqual(argv[0], "llama-server")
        # registry args flow through verbatim.
        self.assertIn("-c", argv)
        self.assertIn("8192", argv)
        self.assertIn("--n-gpu-layers", argv)
        # host/port pinned last so they always apply.
        self.assertEqual(argv[-4:], ["--host", "0.0.0.0", "--port", "8080"])


class TestBuildArgvVllmUnchanged(unittest.TestCase):
    """The vLLM path was already correct (vllm serve <hf_id> auto-fetches)."""

    def test_vllm_serves_hf_id_positionally(self):
        argv = build_argv(_vllm_pick())
        self.assertEqual(
            argv[:3],
            ["vllm", "serve", "deepseek-ai/DeepSeek-V4-flash-dspark"],
        )
        # the vLLM path must not pick up the llama.cpp --hf-repo flag.
        self.assertNotIn("--hf-repo", argv)
        self.assertNotIn("-m", argv)
        self.assertEqual(argv[-4:], ["--host", "0.0.0.0", "--port", "8000"])

    def test_vllm_registry_args_pass_through(self):
        argv = build_argv(_vllm_pick())
        self.assertIn("--kv-cache-dtype", argv)
        self.assertIn("fp8", argv)


class TestEndpointAndPorts(unittest.TestCase):
    def test_endpoint_localhost_default(self):
        self.assertEqual(endpoint_url(port=8080), "http://localhost:8080/v1")

    def test_endpoint_blank_host_falls_back(self):
        self.assertEqual(endpoint_url(host="", port=8000),
                         "http://localhost:8000/v1")

    def test_default_port_per_backend(self):
        self.assertEqual(default_port("llama.cpp"), 8080)
        self.assertEqual(default_port("vllm"), 8000)


if __name__ == "__main__":
    unittest.main()
