# SPDX-License-Identifier: Apache-2.0
"""KDA spec-verify recurrence with a replay stash instead of per-token state stores.

Derived from `fused_recurrent_gated_delta_rule_fwd_kernel` (IS_KDA, COMPUTE_GATE, SAFE_GATE,
SIGMOID_BETA branch) in vLLM's `third_party/flash_linear_attention/ops/fused_recurrent.py`,
which is copied from flash-linear-attention (Songlin Yang, Yu Zhang, MIT).

Why. In a verify step every sequence carries T = k+1 tokens and the stock kernel stores the full
fp32 state after EVERY token (slot i of the request's num_spec+1 state slots), so the next step can
start from slot (num_accepted-1). Per KDA layer, per sequence, per rank that is a 1 MiB read plus
T MiB of writes (16 local heads x 128 x 128 fp32).

What this does instead (same slots, same memory, no new buffers):
  * slot 0 gets the full state after token 0, exactly as today;
  * slots 1..T-1 get a small replay record instead of a state: the fp32 values the recurrence
    consumed for that token (L2-normalised k, the decay exp(gate), raw v, sigmoid(beta)) plus a
    NaN marker word;
  * the next step reads slot 0 and replays tokens 1..a-1 from their records, running the same fp32
    operations on the same fp32 values as the stock loop did.
    MEASURED (GB10, 2026-09-26, bench_kda.py exact): NOT bit-exact. The rebuilt states differ from
    the stock snapshots by ~1 fp32 ulp (1e-8..3e-8 abs), apparently from different instruction selection
    in the replay loop. Verify outputs (bf16) were bit-equal in the printed trials, with at most
    2.3e-10 abs difference seen. A single-loop variant (kda_stash_v2_single_loop.py) also differs from
    the stock kernel, and already at token 0, and is slower, so this v1 is kept.
  * FULL[n] = 1 makes sequence n store full states in every slot, as today. The caller sets it
    when the verify window touches a mamba block boundary (align-mode prefix caching copies
    intermediate slots then). A slot whose marker word is not the NaN marker holds a full state
    and is read directly, so FULL and stash steps can alternate freely.

Record layout (slot j >= 1, head h: a 128 x 128 fp32 region; program i_v owns rows 8*i_v .. 8*i_v+7):
  own row 8*i_v+7: v[8 values of this program], beta at col 8, marker at col 9 (per program, so a
      program only ever reads markers it wrote itself: no intra-launch race);
  aux[slot, head, P, 0:2, :] (P = parity 0/1): normalised k and decay exp(gate) of the head, written
      by program i_v == 0 only, into a side buffer of 4 KiB per slot per head (3 % of the state pool).
      It cannot live inside the slot: a FULL-mode step overwrites whole slots while other programs of
      the head still replay from them. The parity flips every step and lives in the marker, so this
      step's writes (half P_s) never touch the half the others replay from (P_{s-1}).
A record is ~1.7 KiB per head per token (vs a 64 KiB state), so a verify step writes one state
(slot 0) plus T-1 small records instead of T states.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.utils.math_utils import cdiv, next_power_of_2

# A quiet-NaN bit pattern. A live fp32 state is never NaN, so a slot holding a full state can
# never carry it.
STASH_MARKER = 0x7FC51A5E  # even; the low bit carries the record parity


@triton.jit(do_not_specialize=["N", "T"])
def kda_verify_stash_kernel(
    q, k, v, g, beta, o,
    h,          # fp32 state pool [num_slots, HV, V, K]
    h_bits,     # same storage viewed as int32
    aux,        # fp32 [num_slots, HV, 2, 2, K] shared k/decay records
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    full_mode,  # int32 [N] or dummy
    a_log,
    g_bias,
    scale,
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

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_beta = beta + bos * HV + i_hv
    p_gk = g + (bos * HV + i_hv) * K + o_k
    b_a_log = tl.exp(tl.load(a_log + i_h).to(tl.float32))
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    # offsets of this program's rows inside one slot
    reg = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    own = i_hv * V * K + (i_v * BV + BV - 1) * K  # this program's record row

    slot0 = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    if slot0 <= 0:
        return
    n_acc = tl.load(num_accepted_tokens + i_n).to(tl.int64)

    # parity of the previous step's records, from this program's own marker in slot 1
    slot1 = tl.load(ssm_state_indices + i_n * stride_indices_seq + 1).to(tl.int64)
    mark1 = tl.load(h_bits + slot1 * stride_state + own + BV + 1)
    prev_par = tl.where((mark1 & -2) == MARKER, mark1 & 1, 1)
    new_par = 1 - prev_par

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if n_acc > 1:
        slot_last = tl.load(ssm_state_indices + i_n * stride_indices_seq + n_acc - 1).to(tl.int64)
        mark = tl.load(h_bits + slot_last * stride_state + own + BV + 1)
        if (mark & -2) == MARKER:
            # stash mode: slot 0 = state after token 0 of the previous step, replay 1..a-1
            b_h += tl.load(h + slot0 * stride_state + reg, mask=mask_h, other=0).to(tl.float32)
            for j in range(1, n_acc):
                slot_j = tl.load(ssm_state_indices + i_n * stride_indices_seq + j).to(tl.int64)
                sb = h + slot_j * stride_state
                ab = aux + (slot_j * HV + i_hv) * 4 * K + prev_par * 2 * K
                s_k = tl.load(ab + o_k)
                s_d = tl.load(ab + K + o_k)
                s_v = tl.load(sb + own + o_r)
                s_b = tl.load(sb + own + BV)
                b_h *= s_d[None, :]
                s_v -= tl.sum(b_h * s_k[None, :], 1)
                s_v *= s_b
                b_h += s_v[:, None] * s_k[None, :]
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

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_gk += HV * K
        p_beta += HV


_AUX: dict = {}


def get_aux(state: torch.Tensor) -> torch.Tensor:
    """Side buffer for one state pool, allocated once per pool (outside graph capture)."""
    key = (state.data_ptr(), tuple(state.shape))
    t = _AUX.get(key)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("kda_stash: aux buffer must be allocated before CUDA graph capture")
        t = torch.zeros(state.shape[0], state.shape[1], 2, 2, state.shape[-1], device=state.device,
                        dtype=torch.float32)
        _AUX[key] = t
    return t


def fused_recurrent_kda_stash(
    q, k, v, g, beta,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    lower_bound: float = -5.0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    full_mode: torch.Tensor | None = None,
    aux: torch.Tensor | None = None,
):
    """Drop-in for fused_recurrent_kda(..., compute_gate=True, sigmoid_beta=True,
    use_qk_l2norm_in_kernel=True, inplace_final_state=True) on the spec-verify path."""
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    assert B == 1 and ssm_state_indices.ndim == 2 and ssm_state_indices.stride(1) == 1
    assert K == 128 and V % 8 == 0 and initial_state.dtype == torch.float32
    # The pool is a per-layer view into vLLM's shared KV/mamba cache: slots are strided (stride(0) is
    # the cache page, larger than one state), only the per-slot [HV, V, K] block must be dense.
    assert initial_state.stride(3) == 1 and initial_state.stride(2) == K
    assert initial_state.stride(1) == V * K and initial_state.stride(0) >= HV * V * K
    N = len(cu_seqlens) - 1
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)
    NV = cdiv(V, BV)
    if scale is None:
        scale = K ** -0.5
    o = torch.empty_like(k) if out is None else out
    if aux is None:
        aux = get_aux(initial_state)
    grid = (1, NV, N * HV)
    kda_verify_stash_kernel[grid](
        q, k, v, g, beta, o,
        initial_state, initial_state.view(torch.int32), aux,
        cu_seqlens, ssm_state_indices, num_accepted_tokens,
        full_mode if full_mode is not None else cu_seqlens,
        a_log.reshape(-1).contiguous(), g_bias.reshape(-1).contiguous(), scale,
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
