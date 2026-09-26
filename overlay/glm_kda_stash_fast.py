# SPDX-License-Identifier: Apache-2.0
"""GLM_KDA_STASH_NOCOPY / GLM_KDA_FLAG_FUSED: two exact trims of the KDA verify-state stash (GLM_KDA_STASH=1).

Tony v11 image (vLLM 0.1.dev20051+g487ecf187), `vllm.models.glm5next.nvidia.kda`. Needs overlay/glm_kda_stash.py
and overlay/kda_stash.py installed (GLM_KDA_STASH=1); installs after them and changes nothing when the stash is off.

GLM_KDA_STASH_NOCOPY=1
  `kda_stash.fused_recurrent_kda_stash` starts with `q, k, v, g, beta = (t.contiguous() for ...)`. In the GLM
  layer q/k/v are views into the merged [T, 6416] projection (token stride 6416; 6144 after the index_select of a
  mixed step) and beta is a [1, T, 16] view with token stride 6416, so four gather copies run per KDA layer per
  verify step (34 layers). This module launches a copy of `kda_verify_stash_kernel` whose only change is the
  q/k/v/g/beta addressing (base + bos * token_stride + head * D; advance by token_stride), the same two-line
  change `glm_kda_nocopy.py` (glm-exact-wins) makes to the stock kernel. It reads the same elements with the same
  arithmetic, tile shapes, warps and stages, so outputs, states and replay records are bit-identical to the stash
  (GPU test: tests/test_glm_kernels_0926_gpu.py).

GLM_KDA_FLAG_FUSED=1 (or =share)
  The stash wrapper decides per verify sequence whether the window crosses a mamba block boundary (align-mode
  prefix caching then copies intermediate slots, which must hold full states). Today that is a chain of small
  torch ops per KDA layer: positions slice, index_select (mixed steps), qsl .long(), two clamps, two gathers, two
  floor-divides, compare, .to(int32). This replaces the chain with ONE Triton kernel that produces the identical
  int32 flags (floor division kept exact for any sign; mixed prefill/decode via spec_token_indx; padded graph
  rows read the same clamped indices as the torch chain). `share` computes it once per forward (in the first KDA
  layer that needs it) and reuses the tensor in the other layers of the same forward: same inputs, same values.
  Credit: the DS4.1 eager-glue principle (ds41 adapter/eager_glue.py, ours) applied to the GLM stash (glm-kernels,
  this overlay); field-lifetime analysis recorded on 2026-09-26.

Knobs are read per call and are switchable in-boot through overlay/glm_ab.py when it is armed.
Stash kernel origin: vLLM `fused_recurrent_gated_delta_rule_fwd_kernel` from flash-linear-attention (Songlin Yang,
Yu Zhang, MIT), stash rewrite in overlay/kda_stash.py (ours).
"""
from __future__ import annotations

import os
import sys

TARGET = "vllm.models.glm5next.nvidia.kda"
_OFF = ("", "0", "off", "false", "no")
_state = {"kernel": None, "flag_kernel": None, "calls": 0, "strided": 0, "flags": 0, "shared_hits": 0,
          "share_key": None, "share_val": None, "first_prefix": None}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-kda-stash-fast: {msg}\n")


def env(name: str, default=None):
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _on(name: str) -> bool:
    return str(env(name, "0")).strip().lower() not in _OFF


def installed(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in _OFF


# ------------------------------------------------------------------------------------------------------
# strides (pure)
# ------------------------------------------------------------------------------------------------------
def token_strides(q, k, v, g, beta):
    """(s_q, s_k, s_v, s_g, s_beta) when the stash kernel can read the tensors in place, else None. Same rule as
    glm_kda_nocopy.token_strides restricted to what the stash kernel supports (KDA gate [1,T,HV,K], scalar beta)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or g is None or beta is None:
        return None
    if g.dim() != 4 or beta.dim() != 3:
        return None
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1 or g.shape[0] != 1 or beta.shape[0] != 1:
        return None
    H, K = k.shape[2], k.shape[3]
    HV, V = v.shape[2], v.shape[3]
    T = k.shape[1]
    if q.shape != k.shape or v.shape[1] != T or g.shape[1] != T or beta.shape[1] != T:
        return None
    if q.stride(3) != 1 or q.stride(2) != K or k.stride(3) != 1 or k.stride(2) != K:
        return None
    if v.stride(3) != 1 or v.stride(2) != V:
        return None
    if tuple(g.shape[2:]) != (HV, K) or g.stride(3) != 1 or g.stride(2) != K:
        return None
    if beta.shape[2] != HV or beta.stride(2) != 1:
        return None
    return (q.stride(1), k.stride(1), v.stride(1), g.stride(1), beta.stride(1))


# ------------------------------------------------------------------------------------------------------
# kernels (built lazily)
# ------------------------------------------------------------------------------------------------------
def _build_kernel():
    from vllm.third_party.flash_linear_attention.ops.op import exp
    from vllm.triton_utils import tl, triton

    @triton.jit(do_not_specialize=["N", "T"])
    def kda_verify_stash_kernel_strided(
        q, k, v, g, beta, o,
        h, h_bits, aux,
        cu_seqlens, ssm_state_indices, num_accepted_tokens, full_mode,
        a_log, g_bias, scale,
        s_q, s_k, s_v, s_g, s_b,
        N: tl.int64,
        T: tl.int64,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        stride_state: tl.constexpr,
        stride_indices_seq: tl.constexpr,
        HAS_FULL: tl.constexpr,
        LOWER_BOUND: tl.constexpr,
        MARKER: tl.constexpr,
    ):
        # Body: overlay/kda_stash.py kda_verify_stash_kernel, only the five input pointers changed (marked).
        i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_n, i_hv = i_nh // HV, i_nh % HV
        i_h = i_hv // (HV // H)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        if T == 0:
            return

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        o_r = tl.arange(0, BV)

        # CHANGED: token strides instead of the contiguous (bos * H + i_h) * K etc.
        p_q = q + bos * s_q + i_h * K + o_k
        p_k = k + bos * s_k + i_h * K + o_k
        p_v = v + bos * s_v + i_hv * V + o_v
        p_beta = beta + bos * s_b + i_hv
        p_gk = g + bos * s_g + i_hv * K + o_k
        b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))
        p_o = o + (bos * HV + i_hv) * V + o_v

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        reg = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        own = i_hv * V * K + (i_v * BV + BV - 1) * K

        slot0 = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
        if slot0 <= 0:
            return
        n_acc = tl.load(num_accepted_tokens + i_n).to(tl.int64)

        slot1 = tl.load(ssm_state_indices + i_n * stride_indices_seq + 1).to(tl.int64)
        mark1 = tl.load(h_bits + slot1 * stride_state + own + BV + 1)
        prev_par = tl.where((mark1 & -2) == MARKER, mark1 & 1, 1)
        new_par = 1 - prev_par

        b_h = tl.zeros([BV, BK], dtype=tl.float32)
        if n_acc > 1:
            slot_last = tl.load(ssm_state_indices + i_n * stride_indices_seq + n_acc - 1).to(tl.int64)
            mark = tl.load(h_bits + slot_last * stride_state + own + BV + 1)
            if (mark & -2) == MARKER:
                b_h += tl.load(h + slot0 * stride_state + reg, mask=mask_h, other=0).to(tl.float32)
                for j in range(1, n_acc):
                    slot_j = tl.load(ssm_state_indices + i_n * stride_indices_seq + j).to(tl.int64)
                    sb = h + slot_j * stride_state
                    ab = aux + (slot_j * HV + i_hv) * 4 * K + prev_par * 2 * K
                    s_kk = tl.load(ab + o_k)
                    s_d = tl.load(ab + K + o_k)
                    s_vv = tl.load(sb + own + o_r)
                    s_bb = tl.load(sb + own + BV)
                    b_h *= s_d[None, :]
                    s_vv -= tl.sum(b_h * s_kk[None, :], 1)
                    s_vv *= s_bb
                    b_h += s_vv[:, None] * s_kk[None, :]
            else:
                b_h += tl.load(h + slot_last * stride_state + reg, mask=mask_h, other=0).to(tl.float32)
        else:
            b_h += tl.load(h + slot0 * stride_state + reg, mask=mask_h, other=0).to(tl.float32)

        if HAS_FULL:
            full_flag = tl.load(full_mode + i_n).to(tl.int32)
        else:
            full_flag = tl.zeros([], dtype=tl.int32)

        for i_t in range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale
            b_gk = tl.load(p_gk).to(tl.float32)
            b_gk += tl.load(g_bias + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
            b_gk = LOWER_BOUND / (1.0 + tl.exp(-(b_a_log * b_gk)))
            b_d = exp(b_gk)
            b_h *= b_d[None, :]
            b_v0 = b_v
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_beta = tl.sigmoid(tl.load(p_beta).to(tl.float32))
            b_v *= b_beta
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

            slot_t = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
            if slot_t > 0:
                if (i_t == 0) | (full_flag != 0):
                    tl.store(h + slot_t * stride_state + reg, b_h, mask=mask_h)
                else:
                    sb = h + slot_t * stride_state
                    if i_v == 0:
                        ab = aux + (slot_t * HV + i_hv) * 4 * K + new_par * 2 * K
                        tl.store(ab + o_k, b_k)
                        tl.store(ab + K + o_k, b_d)
                    tl.store(sb + own + o_r, b_v0)
                    tl.store(sb + own + BV, b_beta)
                    tl.store(h_bits + slot_t * stride_state + own + BV + 1, MARKER + new_par)

            # CHANGED: advance by the token strides
            p_q += s_q
            p_k += s_k
            p_o += HV * V
            p_v += s_v
            p_gk += s_g
            p_beta += s_b

    return kda_verify_stash_kernel_strided


def _build_flag_kernel():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def kda_stash_full_flags_kernel(pos, sel, qsl, out, n, hi, bs, HAS_SEL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        s = tl.load(qsl + i, mask=m, other=0).to(tl.int64)
        e = tl.load(qsl + i + 1, mask=m, other=0).to(tl.int64) - 1
        s = tl.minimum(tl.maximum(s, 0), hi)
        e = tl.minimum(tl.maximum(e, 0), hi)
        if HAS_SEL:
            s = tl.load(sel + s, mask=m, other=0).to(tl.int64)
            e = tl.load(sel + e, mask=m, other=0).to(tl.int64)
        first = tl.load(pos + s, mask=m, other=0).to(tl.int64)
        last = tl.load(pos + e, mask=m, other=0).to(tl.int64) + 1
        # torch // on int64 is floor division; Triton // truncates. bs > 0.
        qf = first // bs
        qf = tl.where((first % bs != 0) & (first < 0), qf - 1, qf)
        ql = last // bs
        ql = tl.where((last % bs != 0) & (last < 0), ql - 1, ql)
        tl.store(out + i, (ql != qf).to(tl.int32), mask=m)

    return kda_stash_full_flags_kernel


def full_flags_torch(positions, num_actual_tokens, spec_token_indx, non_spec_token_indx, spec_query_start_loc, n,
                     bs):
    """The stash wrapper's torch chain (overlay/glm_kda_stash.py `_state["full"]`), verbatim, for tests."""
    import torch
    pos = positions[:num_actual_tokens]
    if non_spec_token_indx is not None and non_spec_token_indx.numel() > 0:
        pos = pos.index_select(0, spec_token_indx)
    qsl = spec_query_start_loc[: n + 1].long()
    hi = max(pos.numel() - 1, 0)
    first = pos[qsl[:-1].clamp(0, hi)]
    last = pos[(qsl[1:] - 1).clamp(0, hi)]
    return ((last + 1) // bs != first // bs).to(torch.int32)


def full_flags_fused(positions, num_actual_tokens, spec_token_indx, non_spec_token_indx, spec_query_start_loc, n,
                     bs):
    import torch
    if _state["flag_kernel"] is None:
        _state["flag_kernel"] = _build_flag_kernel()
    mixed = non_spec_token_indx is not None and non_spec_token_indx.numel() > 0
    numel = spec_token_indx.numel() if mixed else min(num_actual_tokens, positions.numel())
    if numel == 0 and n > 0:
        # no selected token: the kernel would read element 0 of an empty selection. The stash wrapper never gets
        # here (num_spec_decodes > 0 implies spec tokens); keep the torch chain's own behaviour if it ever does.
        return full_flags_torch(positions, num_actual_tokens, spec_token_indx, non_spec_token_indx,
                                spec_query_start_loc, n, bs)
    hi = max(numel - 1, 0)
    out = torch.empty(n, dtype=torch.int32, device=positions.device)
    if n > 0:
        BLOCK = 64 if n <= 64 else 256
        _state["flag_kernel"][(triton_cdiv(n, BLOCK),)](
            positions, spec_token_indx if mixed else positions, spec_query_start_loc, out, n, hi, bs,
            HAS_SEL=mixed, BLOCK=BLOCK, num_warps=1 if BLOCK == 64 else 4)
    _state["flags"] += 1
    return out


def triton_cdiv(a, b):
    return (a + b - 1) // b


def fused_recurrent_kda_stash_strided(q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices,
                                      num_accepted_tokens, a_log, g_bias, lower_bound=-5.0, scale=None, out=None,
                                      full_mode=None, aux=None, _strides=None):
    """kda_stash.fused_recurrent_kda_stash without the five .contiguous() calls (same launch otherwise)."""
    import torch
    from kda_stash import STASH_MARKER, get_aux
    from vllm.utils.math_utils import cdiv, next_power_of_2

    if _state["kernel"] is None:
        _state["kernel"] = _build_kernel()
    s_q, s_k, s_v, s_g, s_b = _strides
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    assert B == 1 and ssm_state_indices.ndim == 2 and ssm_state_indices.stride(1) == 1
    assert K == 128 and V % 8 == 0 and initial_state.dtype == torch.float32
    assert initial_state.stride(3) == 1 and initial_state.stride(2) == K
    assert initial_state.stride(1) == V * K and initial_state.stride(0) >= HV * V * K
    N = len(cu_seqlens) - 1
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NV = cdiv(V, BV)
    if scale is None:
        scale = K ** -0.5
    if out is None:
        o = torch.empty(k.shape, dtype=k.dtype, device=k.device)  # = empty_like of the contiguous copy
    else:
        assert out.shape == k.shape and out.dtype == k.dtype and out.is_contiguous()
        o = out
    if aux is None:
        aux = get_aux(initial_state)
    grid = (1, NV, N * HV)
    _state["kernel"][grid](
        q, k, v, g, beta, o,
        initial_state, initial_state.view(torch.int32), aux,
        cu_seqlens, ssm_state_indices, num_accepted_tokens,
        full_mode if full_mode is not None else cu_seqlens,
        a_log.reshape(-1).contiguous(), g_bias.reshape(-1).contiguous(), scale,
        s_q, s_k, s_v, s_g, s_b,
        N=N, T=T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        stride_state=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        HAS_FULL=full_mode is not None,
        LOWER_BOUND=lower_bound,
        MARKER=STASH_MARKER - (1 << 32) if STASH_MARKER >= (1 << 31) else STASH_MARKER,
        num_warps=1,
        num_stages=3,
    )
    return o, initial_state


# ------------------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------------------
def _stash_on() -> bool:
    ab = sys.modules.get("glm_ab")
    return True if ab is None else ab.flag("GLM_KDA_STASH")


def install(mod) -> None:
    import torch

    gks = sys.modules.get("glm_kda_stash")
    cls = mod.Glm5NextLinearAttention
    if gks is None or not hasattr(cls._forward, "__wrapped__"):
        _log("GLM_KDA_STASH is not installed on this module; nothing to do")
        return
    if getattr(cls, "_glm_kda_stash_fast", False):
        return
    cls._glm_kda_stash_fast = True

    if installed("GLM_KDA_STASH_NOCOPY"):
        prev = mod.fused_recurrent_kda  # the stash dispatcher (possibly wrapped by glm_kda_nocopy)

        def fused_recurrent_kda(*args, **kw):
            _state["calls"] += 1
            if _state["calls"] <= 2 or _state["calls"] % 20000 == 0:
                _log(f"dispatch call {_state['calls']}: strided={_state['strided']} flags={_state['flags']} "
                     f"shared_hits={_state['shared_hits']} nocopy_on={_on('GLM_KDA_STASH_NOCOPY')} "
                     f"nacc={'set' if kw.get('num_accepted_tokens') is not None else 'none'}")
            if args or not _on("GLM_KDA_STASH_NOCOPY") or not _stash_on():
                return prev(*args, **kw)
            idx = kw.get("ssm_state_indices")
            nacc = kw.get("num_accepted_tokens")
            st = kw.get("initial_state")
            if (nacc is None or idx is None or idx.ndim != 2 or st is None or st.dtype != torch.float32
                    or not kw.get("compute_gate") or not kw.get("sigmoid_beta")):
                return prev(*args, **kw)
            strides = token_strides(kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"])
            out = kw.get("out")
            if strides is None or (out is not None and not out.is_contiguous()):
                return prev(*args, **kw)
            _state["strided"] += 1
            return fused_recurrent_kda_stash_strided(
                kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"], st, kw["cu_seqlens"], idx, nacc,
                kw["a_log"], kw["g_bias"], lower_bound=kw.get("lower_bound", -5.0), out=kw.get("out"),
                full_mode=gks._state["full"], _strides=strides)

        fused_recurrent_kda.__wrapped__ = prev
        mod.fused_recurrent_kda = fused_recurrent_kda
        _log("stash verify kernel reads q/k/v/g/beta in place (GLM_KDA_STASH_NOCOPY=1)")

    if installed("GLM_KDA_FLAG_FUSED"):
        orig_inner = cls._forward.__wrapped__  # the image's _forward (glm_kda_stash sets __wrapped__)
        stash_wrapper = cls._forward

        def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
            mode = str(env("GLM_KDA_FLAG_FUSED", "0")).strip().lower()
            if mode in _OFF or not _stash_on():
                return stash_wrapper(self, qkv_proj_states, g1, beta, core_attn_out)
            gks._state["full"] = None
            try:
                from vllm.forward_context import get_forward_context
                md_all = get_forward_context().attn_metadata
                md = md_all.get(self.prefix) if isinstance(md_all, dict) else None
                cc = self.cache_config
                if (md is not None and md.spec_sequence_masks is not None and md.num_spec_decodes > 0
                        and cc is not None and getattr(cc, "mamba_cache_mode", "none") != "none"):
                    bs = int(cc.mamba_block_size or cc.block_size)
                    n = md.num_spec_decodes
                    positions = self._kda_stash_positions
                    args = (positions, md.num_actual_tokens, md.spec_token_indx, md.non_spec_token_indx,
                            md.spec_query_start_loc, n, bs)
                    if mode == "share":
                        key = (id(md), id(positions), n, md.num_actual_tokens, bs)
                        if _state["first_prefix"] is None:
                            _state["first_prefix"] = self.prefix
                        if self.prefix == _state["first_prefix"] or _state["share_key"] != key:
                            _state["share_key"] = key
                            _state["share_val"] = (md, positions, full_flags_fused(*args))
                        else:
                            _state["shared_hits"] += 1
                        gks._state["full"] = _state["share_val"][2]
                    else:
                        gks._state["full"] = full_flags_fused(*args)
                return orig_inner(self, qkv_proj_states, g1, beta, core_attn_out)
            finally:
                gks._state["full"] = None

        _forward.__wrapped__ = orig_inner
        cls._forward = _forward
        _log("stash boundary flags from one Triton kernel (GLM_KDA_FLAG_FUSED=%s)"
             % os.environ.get("GLM_KDA_FLAG_FUSED"))


def register() -> None:
    import importlib.abc
    import importlib.util

    if not (installed("GLM_KDA_STASH_NOCOPY") or installed("GLM_KDA_FLAG_FUSED")):
        return
    if os.environ.get("GLM_KDA_STASH") != "1":
        _log("GLM_KDA_STASH is not 1; GLM_KDA_STASH_NOCOPY / GLM_KDA_FLAG_FUSED have nothing to trim")
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET:
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
                install(module)
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
