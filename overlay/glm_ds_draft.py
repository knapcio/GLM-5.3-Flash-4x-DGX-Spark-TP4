# SPDX-License-Identifier: Apache-2.0
"""DFlash2 draft-side transfers from DS4.1 (all default OFF; each flag is independent).

GLM_DS_DRAFT_HEAD_FP8=1   (port of our DS4.1 DSV41_DRAFT_HEAD_FP8)
    DFlash2 (compute_candidates) and DSpark (compute_draft_logits, own head) are both covered.
    DFlash2 borrows the target's vocab-parallel lm_head (BF16, 38720 x 4096 = 317 MB per rank at TP4) and
    reads it once per step for the draft rows (num_reqs x K) in compute_candidates. The DRAFT gets an fp8
    copy (e4m3 + one power-of-two exponent per 32x32 block, built once after the drafter loads) and a Triton
    kernel computes its local logits from it: 159 MB instead of 317 MB per step. The target's logits and
    therefore every committed token are untouched (the draft is verified), so greedy output is identical
    and sampled output stays exact in distribution (the walk and the rejection sampler read the same
    cached draft scores). Acceptance can move slightly because the proposals can change (DS4.1 offline:
    3.0882 -> 3.0875 tokens/step). Rows > GLM_DS_DRAFT_HEAD_FP8_MAX_M (64, <= 128) keep the BF16 head.
    Kernel: ours from ds41 adapter/draft_head_fp8.py, unchanged except for the bf16 output.
GLM_DS_TOPK_ONE_GATHER=1
    LogitsProcessor.get_top_k_tokens (DFlash2's vocab-parallel top-k) all-gathers values and ids with two
    collectives. Here the bf16 values' bits (int16, sign-extended) and the int64 ids ride one int64
    all-gather and are unpacked; the second top-k then sees the same bytes. Bit-identical; one collective
    fewer per draft step (41-56 us NCCL, ~17 us RoCE).
GLM_DS_DRAFT_TAU=0.7      (port of our DS4.1 DSV41_DRAFT_TAU; needs "draft_sample_method": "probabilistic")
    Probabilistic DFlash2 drafts are Gumbel-sampled from the selector scores at the request temperature T.
    Dividing the scores by tau before the walk sharpens the proposal to softmax(s / (tau T)). The walk
    stores the scores it sampled from (_selector_scores) and _cache_draft_logits copies those into
    draft_logits, which the rejection sampler reads as q: proposal and ratio test see the same q, so the
    output distribution is unchanged (speculative sampling is exact for any q); only acceptance moves.
    Greedy rows (T = 0) are argmax, unaffected. With the default greedy drafts the flag does nothing.
    DS4.1 measured +1.2 % (offline) and +1.8-2.0 % (fleet, sampled thinking) at 0.7-0.8.
"""
from __future__ import annotations

import os
import sys

HEAD_FP8 = os.environ.get("GLM_DS_DRAFT_HEAD_FP8", "0").strip() not in ("0", "", "off", "false")
HEAD_FP8_MAX_M = min(128, int(os.environ.get("GLM_DS_DRAFT_HEAD_FP8_MAX_M", "64") or 64))
ONE_GATHER = os.environ.get("GLM_DS_TOPK_ONE_GATHER", "0").strip() not in ("0", "", "off", "false")
TAU = float(os.environ.get("GLM_DS_DRAFT_TAU", "1") or 1)
if not 0.05 <= TAU <= 4.0:
    raise ValueError(f"GLM_DS_DRAFT_TAU={TAU} out of range [0.05, 4]")


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------
# fp8 twin + kernel (ds41 adapter/draft_head_fp8.py)
# ------------------------------------------------------------------------------------------
def make_twin(w, chunk_rows: int = 4096):
    """bf16 [N, K] -> (e4m3 [N, K], exponent uint8 [ceil(N/32), K/32]); lossy, draft-only.
    Built in row chunks so the fp32 temporaries stay ~chunk_rows x K x 4 bytes (unified memory)."""
    import torch
    n, k = w.shape
    assert k % 32 == 0, k
    nb = (n + 31) // 32
    q = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
    s = torch.empty((nb, k // 32), dtype=torch.uint8, device=w.device)
    chunk_rows = max(32, chunk_rows - chunk_rows % 32)
    for r0 in range(0, n, chunk_rows):
        r1 = min(n, r0 + chunk_rows)
        rows = r1 - r0
        pad = (-rows) % 32
        wf = torch.nn.functional.pad(w[r0:r1].float(), (0, 0, 0, pad)).view((rows + pad) // 32, 32, k // 32, 32)
        amax = wf.abs().amax(dim=(1, 3)).clamp_min(2.0 ** -126)
        e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
        qq = (wf / torch.exp2(e)[:, None, :, None]).to(torch.float8_e4m3fn)
        q[r0:r1] = qq.view(rows + pad, k)[:rows]
        s[r0 // 32:r0 // 32 + (rows + pad) // 32] = (e + 127).to(torch.uint8)
        del wf, amax, e, qq
    return q, s


def dequant_twin(twin):
    """Reference dequantization (tests): [N, K] fp32."""
    import torch
    q, s = twin
    n, k = q.shape
    e = torch.repeat_interleave(torch.repeat_interleave(s.float() - 127.0, 32, dim=0)[:n], 32, dim=1)
    return q.float() * torch.exp2(e)


_KERNEL = None
_TILES = ((16, 16, 256, 4, 3), (64, 64, 128, 4, 3), (128, 128, 64, 8, 3))  # max M, BM, BK, warps, stages


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _head_fp8_kernel(X, W8, S, Y, M, N, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr):
        pid = tl.program_id(0)
        m = tl.program_id(1) * BM + tl.arange(0, BM)
        n = pid * BN + tl.arange(0, BN)
        nmask = n < N
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + m[:, None] * K + k[None, :], m[:, None] < M, 0.0)
            w8 = tl.load(W8 + n[None, :] * K + k[:, None], nmask[None, :], 0.0)
            e = tl.load(S + (n[None, :] // 32) * (K // 32) + k[:, None] // 32, nmask[None, :], 127)
            w = (w8.to(tl.float32) * tl.exp2(e.to(tl.float32) - 127.0)).to(tl.bfloat16)
            acc += tl.dot(x, w)
        tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & nmask[None, :])

    _KERNEL = _head_fp8_kernel
    return _KERNEL


def head_fp8(x, twin):
    """x [M, K] bf16 (M <= 128) -> bf16 local logits [M, N] (fp32 accumulate, one rounding)."""
    import torch
    import triton
    w8, s = twin
    m, k = x.shape
    n = w8.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _, bm, bk, warps, stages = next(t for t in _TILES if m <= t[0])
    _kernel()[(triton.cdiv(n, 32), triton.cdiv(m, bm))](x.contiguous(), w8, s, y, m, n, k, bm, 32, bk,
                                                        num_warps=warps, num_stages=stages)
    return y


class _Fp8HeadMethod:
    def __init__(self, twin):
        self.twin = twin

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise RuntimeError("glm-ds draft head fp8: unexpected embedding bias")
        return head_fp8(x, self.twin)


class HeadProxy:
    """Stands in for the vocab-parallel lm_head inside LogitsProcessor.get_top_k_tokens/_apply_head: those
    read .quant_method.apply, .shard_indices and .tp_size only."""
    __slots__ = ("quant_method", "shard_indices", "tp_size", "_head")

    def __init__(self, head, twin):
        self.quant_method = _Fp8HeadMethod(twin)
        self.shard_indices = head.shard_indices
        self.tp_size = head.tp_size
        self._head = head


# ------------------------------------------------------------------------------------------
# one packed all-gather for the vocab-parallel top-k
# ------------------------------------------------------------------------------------------
def pack_topk(values, ids):
    """[M, k] values (2-byte float) + [M, k] int64 ids -> [M, 2k] int64 (ids, then value bits)."""
    import torch
    assert values.element_size() == 2 and ids.dtype == torch.int64
    return torch.cat([ids, values.contiguous().view(torch.int16).to(torch.int64)], dim=-1)


def unpack_gathered(g, k: int, tp: int, value_dtype):
    """[M, tp * 2k] int64 (rank-major, each rank's [ids, bits]) -> (values [M, tp*k], ids [M, tp*k])."""
    import torch
    m = g.shape[0]
    g = g.view(m, tp, 2, k)
    ids = g[:, :, 0, :].reshape(m, tp * k)
    values = g[:, :, 1, :].to(torch.int16).view(value_dtype).reshape(m, tp * k)
    return values, ids


def make_get_top_k_tokens(lp_module):
    """Replacement for LogitsProcessor.get_top_k_tokens: the stock body with the two gathers packed into one
    when ONE_GATHER (and a 2-byte value dtype); otherwise identical."""
    import torch
    _topk = lp_module._topk
    gather = lp_module.tensor_model_parallel_all_gather

    def get_top_k_tokens(self, lm_head, hidden_states, k, embedding_bias=None):
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = _topk(logits, k)
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index
        if lm_head.tp_size > 1:
            if ONE_GATHER and values.element_size() == 2:
                g = gather(pack_topk(values, ids), dim=-1)
                values, ids = unpack_gathered(g, values.shape[-1], lm_head.tp_size, values.dtype)
            else:
                values = gather(values, dim=-1)
                ids = gather(ids, dim=-1)
            values, selected = _topk(values, k)
            ids = ids.gather(-1, selected)
        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values

    return get_top_k_tokens


# ------------------------------------------------------------------------------------------
# installs
# ------------------------------------------------------------------------------------------
def install_logits_processor(mod) -> None:
    if not ONE_GATHER:
        return
    cls = mod.LogitsProcessor
    if getattr(cls, "_glm_ds_one_gather", False):
        return
    for sym in ("_topk", "tensor_model_parallel_all_gather"):
        if not hasattr(mod, sym):
            raise RuntimeError(f"glm-ds draft: logits_processor.{sym} is gone; engine drifted")
    cls.get_top_k_tokens = make_get_top_k_tokens(mod)
    cls._glm_ds_one_gather = True
    _log("glm-ds draft: top-k one packed all-gather armed")


def install_dflash2_model(mod) -> None:
    """compute_candidates: use the fp8 twin (if built) for <= MAX_M rows."""
    if not HEAD_FP8:
        return
    import torch
    cls = mod.DFlash2Qwen3ForCausalLM
    if getattr(cls, "_glm_ds_head_fp8", False):
        return
    orig = cls.compute_candidates

    def compute_candidates(self, hidden_states):
        twin = getattr(self, "_glm_ds_head_fp8_twin", None)
        m = hidden_states.shape[0]
        lp = self.candidate_logits_processor
        if (twin is None or not 0 < m <= HEAD_FP8_MAX_M or hidden_states.dtype != torch.bfloat16
                or hidden_states.dim() != 2 or getattr(lp, "head_dtype", None) not in (None, torch.bfloat16)):
            return orig(self, hidden_states)
        return lp.get_top_k_tokens(HeadProxy(self.lm_head, twin), hidden_states,
                                   self.model.candidate_selector.top_k)

    cls.compute_candidates = compute_candidates
    cls._glm_ds_head_fp8 = True
    _log(f"glm-ds draft: fp8 draft head armed (rows <= {HEAD_FP8_MAX_M})")


def build_twin_for(model) -> bool:
    """fp8 twin of the drafter's lm_head (DFlash2: the attached target head; DSpark: its own head)."""
    import torch
    head = getattr(model, "lm_head", None)
    w = getattr(head, "weight", None)
    if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] % 32:
        _log(f"glm-ds draft: no bf16 lm_head on the drafter ({None if w is None else (w.dtype, tuple(w.shape))}); "
             "fp8 head off")
        return False
    if not (hasattr(model, "compute_candidates") or hasattr(model, "compute_draft_logits")):
        _log("glm-ds draft: drafter is neither DFlash2 nor DSpark; fp8 head off")
        return False
    model._glm_ds_head_fp8_twin = make_twin(w.data)
    torch.cuda.empty_cache()  # the chunk temporaries must not linger
    _log(f"glm-ds draft: fp8 twin of the draft LM head {tuple(w.shape)} "
         f"({w.shape[0] * w.shape[1] / 2**20:.0f} MiB e4m3)")
    return True


def install_dflash_speculator(mod) -> None:
    """Build the twin right after the drafter is loaded and the target head attached (before capture)."""
    if not HEAD_FP8:
        return
    cls = mod.DFlashSpeculator
    if getattr(cls, "_glm_ds_head_fp8", False):
        return
    orig = cls.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig(self, target_model, target_attn_layer_names)
        build_twin_for(model)
        return model

    cls.load_draft_model = load_draft_model
    cls._glm_ds_head_fp8 = True


def install_dflash2_speculator(mod) -> None:
    if TAU == 1.0:
        return
    cls = mod.DFlash2Speculator
    if getattr(cls, "_glm_ds_tau", False):
        return
    orig = cls._sample_path
    inv = 1.0 / TAU
    warned = []

    def _sample_path(self, candidate_ids, scores, num_reqs):
        if self.draft_logits is None:  # greedy drafts: argmax is scale-invariant; skip the kernel
            if not warned:
                warned.append(1)
                _log("glm-ds draft: GLM_DS_DRAFT_TAU set but draft_sample_method is greedy; tau has no effect")
            return orig(self, candidate_ids, scores, num_reqs)
        return orig(self, candidate_ids, scores * inv, num_reqs)

    cls._sample_path = _sample_path
    cls._glm_ds_tau = True
    _log(f"glm-ds draft: draft proposal tau={TAU} armed (probabilistic drafts only)")


def install_dspark_model(mod) -> None:
    """DSpark: compute_draft_logits (full draft-vocab logits, gathered) from the fp8 twin for <= MAX_M rows.
    LogitsProcessor.forward -> _get_logits -> _apply_head reads lm_head.quant_method / .tp_size only."""
    if not HEAD_FP8:
        return
    import torch
    cls = mod.Qwen3DSparkForCausalLM
    if getattr(cls, "_glm_ds_head_fp8", False):
        return
    orig = cls.compute_draft_logits

    def compute_draft_logits(self, hidden_states):
        twin = getattr(self, "_glm_ds_head_fp8_twin", None)
        m = hidden_states.shape[0]
        lp = self.logits_processor
        if (twin is None or not 0 < m <= HEAD_FP8_MAX_M or hidden_states.dtype != torch.bfloat16
                or hidden_states.dim() != 2 or getattr(lp, "head_dtype", None) not in (None, torch.bfloat16)
                or getattr(lp, "logits_as_input", False)):
            return orig(self, hidden_states)
        return lp(HeadProxy(self.lm_head, twin), hidden_states)

    cls.compute_draft_logits = compute_draft_logits
    cls._glm_ds_head_fp8 = True
    _log(f"glm-ds draft: fp8 DSpark draft head armed (rows <= {HEAD_FP8_MAX_M})")


def install_dspark_speculator(mod) -> None:
    if not HEAD_FP8:
        return
    cls = mod.DSparkSpeculator
    if getattr(cls, "_glm_ds_head_fp8", False):
        return
    orig = cls.load_draft_model

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = orig(self, target_model, target_attn_layer_names)
        build_twin_for(model)
        return model

    cls.load_draft_model = load_draft_model
    cls._glm_ds_head_fp8 = True
