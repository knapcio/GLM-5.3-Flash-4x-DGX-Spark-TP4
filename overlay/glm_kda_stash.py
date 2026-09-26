# SPDX-License-Identifier: Apache-2.0
"""GLM_KDA_STASH=1: KDA spec-verify recurrence without per-token fp32 state stores.

Gated overlay for the Tony v11 image (vLLM 0.1.dev20051+g487ecf187). Swaps the spec-verify call of
`fused_recurrent_kda` inside `vllm.models.glm5next.nvidia.kda` for `kda_stash.fused_recurrent_kda_stash`
(see that file for the mechanism; measured ulp-level, not bit-exact, vs the stock kernel). Plain decode and
prefill are untouched.

Align-mode prefix caching (on whenever --enable-prefix-caching is set) copies intermediate spec
slots when a verify window crosses a mamba block boundary (block = --block-size tokens). For those
sequences the kernel is told to store full states in every slot, exactly like the stock kernel, so
the copies see the same bytes. The flag is computed on the GPU from `positions` inside the graph.

Install: copy glm_kda_stash.py and ../gpu/kda_stash.py next to overlay/sitecustomize.py and append
sitecustomize.snippet. Inert unless GLM_KDA_STASH=1.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys

TARGET = "vllm.models.glm5next.nvidia.kda"
ROUTER = "vllm.model_executor.layers.fused_moe.router.gate_linear"
HOOKS: dict = {}
# sha256[:16] of the text of the functions this relies on, in the qualified image (filled by --check)
EXPECTED = {
    "Glm5NextLinearAttention._forward": None,
    "Glm5NextLinearAttention.forward": None,
}

_state = {"full": None}


def _ab_flag(name: str) -> bool:
    """Per-call gate under the in-boot A/B harness (overlay/glm_ab.py, TEST ONLY); True when it is off."""
    ab = sys.modules.get("glm_ab")
    return True if ab is None else ab.flag(name)


def _fn_hash(obj) -> str:
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]


def install(mod) -> None:
    import torch

    from kda_stash import fused_recurrent_kda_stash

    cls = mod.Glm5NextLinearAttention
    for name, want in EXPECTED.items():
        if want is None:
            continue
        got = _fn_hash(getattr(cls, name.split(".")[1]))
        if got != want and os.environ.get("GLM_KDA_STASH_ALLOW_DRIFT") != "1":
            raise RuntimeError(f"glm-kda-stash: {name} drifted ({got} != {want}); refusing to patch")

    stock = mod.fused_recurrent_kda

    def dispatch(*args, **kw):
        idx = kw.get("ssm_state_indices")
        nacc = kw.get("num_accepted_tokens")
        if (not _ab_flag("GLM_KDA_STASH") or nacc is None or idx is None or idx.ndim != 2 or kw.get("initial_state") is None
                or kw["initial_state"].dtype != torch.float32 or not kw.get("compute_gate")
                or not kw.get("sigmoid_beta")):
            return stock(*args, **kw)
        return fused_recurrent_kda_stash(
            kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"],
            kw["initial_state"], kw["cu_seqlens"], idx, nacc,
            kw["a_log"], kw["g_bias"], lower_bound=kw.get("lower_bound", -5.0),
            out=kw.get("out"), full_mode=_state["full"],
        )

    mod.fused_recurrent_kda = dispatch

    orig_forward = cls.forward
    orig_inner = cls._forward

    def forward(self, hidden_states, positions):
        self._kda_stash_positions = positions
        return orig_forward(self, hidden_states, positions)

    def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
        _state["full"] = None
        if not _ab_flag("GLM_KDA_STASH"):
            return orig_inner(self, qkv_proj_states, g1, beta, core_attn_out)
        try:
            from vllm.forward_context import get_forward_context
            md_all = get_forward_context().attn_metadata
            md = md_all.get(self.prefix) if isinstance(md_all, dict) else None
            cc = self.cache_config
            if (md is not None and md.spec_sequence_masks is not None and md.num_spec_decodes > 0
                    and cc is not None and getattr(cc, "mamba_cache_mode", "none") != "none"):
                bs = int(cc.mamba_block_size or cc.block_size)
                n = md.num_spec_decodes
                pos = self._kda_stash_positions[: md.num_actual_tokens]
                if md.non_spec_token_indx is not None and md.non_spec_token_indx.numel() > 0:
                    pos = pos.index_select(0, md.spec_token_indx)
                qsl = md.spec_query_start_loc[: n + 1].long()
                hi = max(pos.numel() - 1, 0)
                first = pos[qsl[:-1].clamp(0, hi)]
                last = pos[(qsl[1:] - 1).clamp(0, hi)]
                _state["full"] = ((last + 1) // bs != first // bs).to(torch.int32)
            return orig_inner(self, qkv_proj_states, g1, beta, core_attn_out)
        finally:
            _state["full"] = None

    # keep the eager-break decorator semantics of the original (inactive in FULL graph mode)
    _forward.__wrapped__ = orig_inner
    cls.forward = forward
    cls._forward = _forward
    sys.stderr.write("glm-kda-stash: KDA spec-verify uses the replay stash (GLM_KDA_STASH=1)\n")


def install_router(mod) -> None:
    """GLM_ROUTER_FP32OUT=1: on SM12x GateLinear falls to its tier 6 (bf16 F.linear, then .float()), so the
    router logits the fp32 routing (moe_router_dtype=float32) sees are bf16-rounded. cuBLAS can write the
    fp32 accumulator directly (torch.mm out_dtype), which is what GateLinear already does on SM90/SM100.
    Numerics-changing (closer to the reference), one kernel instead of two."""
    import torch
    cls = mod.GateLinear
    orig = cls.forward

    def forward(self, x):
        if (_ab_flag("GLM_ROUTER_FP32OUT") and self.out_dtype == torch.float32 and x.dtype == torch.bfloat16
                and self.weight.dtype == torch.bfloat16 and self.bias is None and x.dim() == 2):
            return torch.mm(x, self.weight.t(), out_dtype=torch.float32), None
        return orig(self, x)
    cls.forward = forward
    sys.stderr.write("glm-kda-stash: GateLinear writes fp32 logits from cuBLAS (GLM_ROUTER_FP32OUT=1)\n")


def register() -> None:
    import importlib.abc
    import importlib.util

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in HOOKS:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                HOOKS[name](module)
            loader.exec_module = exec_module
            return spec

    if os.environ.get("GLM_KDA_STASH") == "1":
        HOOKS[TARGET] = install
    if os.environ.get("GLM_ROUTER_FP32OUT") == "1":
        HOOKS[ROUTER] = install_router
    if HOOKS:
        sys.meta_path.insert(0, _Finder())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--hashes":
    import importlib
    m = importlib.import_module(TARGET)
    for k in EXPECTED:
        print(k, _fn_hash(getattr(m.Glm5NextLinearAttention, k.split(".")[1])))
