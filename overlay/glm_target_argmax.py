# SPDX-License-Identifier: Apache-2.0
"""GLM_TARGET_VOCAB_ARGMAX=1: vocab-parallel greedy selection for the TARGET verify rows. Exact.

Tony v11 image, vLLM 0.1.dev20051+g487ecf187, V2 model runner (vllm/v1/worker/gpu).

Stock path, every step and every rank (GPUModelRunner.sample):
    lm_head shard GEMM [M, 38720] -> all-gather -> [M, 154880] bf16 (29.7 MB received per rank at M=128)
    -> RejectionSampler / Sampler on the full row (per-8192-block max, global argmax, resample kernels).
Greedy path here (all ranks, same decision):
    lm_head shard GEMM [M, 38720] (the very same `_apply_head` call) -> local (max, argmax) per row
    -> all-gather of 16 bytes per row -> the rank with the largest max wins, lowest rank on ties
    -> the greedy branch of the rejection kernel, re-implemented on the token ids.

The vocab-parallel argmax is vLLM's own `LogitsProcessor.get_top_tokens`, added by qizixi in vllm#34049
("[Spec Decode] Reduce TP communication for speculative decoding draft token generation", 2026-02-22);
credit to its author and reviewers. vLLM uses it only for drafts. The reduction below is the same
(value/index pairs, argmax over ranks). It adds three things the target needs:
  * tie order. Shards are contiguous vocab ranges in rank order, the local argmax returns the first
    maximal index and the rank reduction returns the first maximal rank, so a tie resolves to the lowest
    token id. The stock rejection kernel does the same (tl.max(return_indices) is tie-break-left per
    8192 block, tl.argmax over blocks is tie-break-left), and so does torch.argmax on the full row.
  * the rejection step. Greedy acceptance only needs the target argmax of each row: accept draft i while
    draft_i == argmax_i (and draft_i >= 0); emit argmax at the first mismatch, or the bonus row's argmax
    when everything was accepted (rejection_sampler_utils._rejection_kernel / _resample_kernel /
    _insert_resampled_kernel, temp == 0 branches).
  * dispatch. The fast path runs only when EVERY request in the batch is plain greedy: temperature == 0,
    no logprobs / logprob_token_ids, nothing that makes Sampler._requires_logits_processing true
    (penalties, logit bias, min_tokens, bad words, top-k/p, min-p), no grammar bitmask this step, no NaN
    counting, the stock (non-synthetic) rejection sampler, and a lm_head whose gathered width equals the
    vocab (no padding, no added vocab, scale 1, no soft cap). Everything is read from the per-request
    CPU state that every TP rank holds identically (scheduler output), never from local tensor values,
    so all ranks take the same branch and the collective sequence stays aligned.
Mixed batches (any sampled row) take the stock path unchanged.

Modes:
  GLM_TARGET_VOCAB_ARGMAX=1      fast path when eligible
  GLM_TARGET_VOCAB_ARGMAX=check  run the fast path AND the stock path on every eligible step, count
                                 mismatches (logged on TP rank 0), return the STOCK result
  GLM_TARGET_VOCAB_ARGMAX_LOCAL=triton|torch   local argmax implementation on CUDA (default triton)
  GLM_TARGET_VOCAB_ARGMAX_LOG_EVERY=2000       stats line cadence (TP rank 0)
The mode is read per call through overlay/glm_ab.py when the in-boot A/B harness is armed (TEST ONLY).

Composes with GLM_VERIFY_CUT (dead drafts are replaced by -1 exactly as its RejectionSampler hook does).
NaN logits: undefined in the stock path as well (the block max there is not NaN-propagating); not handled.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_MODE = os.environ.get("GLM_TARGET_VOCAB_ARGMAX", "0").strip().lower()
ENABLED = _MODE in ("1", "on", "true", "check")
CHECK = _MODE == "check"
LOCAL_IMPL = os.environ.get("GLM_TARGET_VOCAB_ARGMAX_LOCAL", "triton").strip().lower()
LOG_EVERY = int(os.environ.get("GLM_TARGET_VOCAB_ARGMAX_LOG_EVERY", "2000"))
NO_LOGPROBS = -1
PAIR_WIDTH = 4  # (value, index, 0, 0): 16 bytes per row, so a RoCE gather can take it
BLOCK = 8192


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-target-argmax: {msg}\n")


def _mode() -> str:
    """Per-call mode: the in-boot A/B harness (overlay/glm_ab.py, TEST ONLY) can switch it at runtime;
    otherwise the value read at import."""
    ab = sys.modules.get("glm_ab")
    if ab is None or not ab.ACTIVE:
        return "check" if CHECK else ("1" if ENABLED else "0")
    return ab.norm_value("GLM_TARGET_VOCAB_ARGMAX", ab.env("GLM_TARGET_VOCAB_ARGMAX"))


# ---------------------------------------------------------------------------------------------------
# pure tensor pieces (CPU-testable; CUDA uses the Triton kernels below when available)
# ---------------------------------------------------------------------------------------------------
def local_argmax_torch(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First maximal index per row (torch.argmax contract) and its value in fp32 (bf16 -> fp32 is exact)."""
    idx = logits.argmax(dim=-1)
    val = logits.gather(-1, idx.unsqueeze(-1)).squeeze(-1).float()
    return val, idx


def pack_pairs(val: torch.Tensor, global_idx: torch.Tensor) -> torch.Tensor:
    """[M] fp32 values + [M] int64 global ids -> [M, 4] fp32. Ids < 2**24 are exact in fp32."""
    out = torch.zeros(val.shape[0], PAIR_WIDTH, dtype=torch.float32, device=val.device)
    out[:, 0] = val
    out[:, 1] = global_idx.to(torch.float32)
    return out


def reduce_pairs(gathered: torch.Tensor, tp: int) -> torch.Tensor:
    """[M, tp * 4] (rank-major, as all_gather(dim=-1) returns it) -> [M] int64 global argmax.
    torch.argmax over ranks returns the first (= lowest vocab range) rank on ties."""
    g = gathered.view(gathered.shape[0], tp, PAIR_WIDTH)
    rank = g[:, :, 0].argmax(dim=-1, keepdim=True)
    return g[:, :, 1].gather(-1, rank).squeeze(-1).to(torch.int64)


def greedy_verify_torch(target: torch.Tensor, draft: torch.Tensor, cu: torch.Tensor, num_reqs: int,
                        width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy branch of the stock rejection sampler on token ids.

    target [L] int64: argmax of every logits row. draft [L]: input_ids at the logits rows (the draft token
    checked at row j is draft[j + 1], same request). cu [num_reqs + 1]: cu_num_logits.
    Returns sampled [num_reqs, width] int64 (entries past num_sampled are don't-care, as in the stock
    kernel) and num_sampled [num_reqs] int32."""
    dev = target.device
    L = target.numel()
    cu = cu[: num_reqs + 1].to(torch.int64)
    rows = torch.arange(L, device=dev)
    req = torch.searchsorted(cu[1:].contiguous(), rows, right=True)
    start = cu[req]
    end = cu[req + 1]
    local = rows - start
    nxt = torch.cat([draft[1:].to(torch.int64), torch.full((1,), -1, dtype=torch.int64, device=dev)])
    match = (nxt == target) & (nxt >= 0) & (rows < end - 1)
    big = torch.iinfo(torch.int64).max
    miss = torch.where(match, torch.full_like(local, big), local)
    acc = torch.full((num_reqs,), big, dtype=torch.int64, device=dev)
    acc = acc.scatter_reduce(0, req, miss, reduce="amin", include_self=True)
    cols = torch.arange(width, device=dev)
    src = torch.minimum(cu[:-1, None] + cols[None, :], cu[1:, None] - 1)
    sampled = target[src]
    return sampled, (acc + 1).to(torch.int32)


# ---------------------------------------------------------------------------------------------------
# Triton kernels (built lazily; only on CUDA)
# ---------------------------------------------------------------------------------------------------
_K = {}


def _kernels():
    if _K:
        return _K
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _block_argmax_kernel(logits_ptr, logits_stride, val_ptr, idx_ptr, num_blocks, vocab,
                             BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        blk = tl.program_id(1)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(logits_ptr + row * logits_stride + offs, mask=offs < vocab,
                    other=float("-inf")).to(tl.float32)
        v, i = tl.max(x, axis=0, return_indices=True)  # tie-break-left
        tl.store(val_ptr + row * num_blocks + blk, v)
        tl.store(idx_ptr + row * num_blocks + blk, (blk * BLOCK_SIZE + i).to(tl.int64))

    @triton.jit
    def _greedy_verify_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, target_ptr, draft_ptr, cu_ptr):
        # mirrors _rejection_kernel (is_greedy) + _resample_kernel / _insert_resampled_kernel (temp == 0)
        r = tl.program_id(0)
        start = tl.load(cu_ptr + r).to(tl.int64)
        end = tl.load(cu_ptr + r + 1).to(tl.int64)
        n = end - start - 1
        accepted = tl.zeros((), tl.int64)
        verifying = accepted == 0  # i1 tensor (loop-carried)
        for i in range(n):
            if verifying:
                t = tl.load(target_ptr + start + i).to(tl.int64)
                d = tl.load(draft_ptr + start + i + 1).to(tl.int64)
                ok = (t == d) & (d >= 0)
                tl.store(sampled_ptr + r * sampled_stride + i, tl.where(ok, d, t))
                verifying = ok
                accepted += ok.to(tl.int64)
        tl.store(num_sampled_ptr + r, (accepted + 1).to(tl.int32))
        if accepted == n:
            tl.store(sampled_ptr + r * sampled_stride + n, tl.load(target_ptr + start + n).to(tl.int64))

    _K["block_argmax"] = _block_argmax_kernel
    _K["verify"] = _greedy_verify_kernel
    _K["triton"] = triton
    return _K


def local_argmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not logits.is_cuda or LOCAL_IMPL != "triton" or logits.stride(-1) != 1:
        return local_argmax_torch(logits)
    k = _kernels()
    M, V = logits.shape
    nb = k["triton"].cdiv(V, BLOCK)
    vals = torch.empty(M, nb, dtype=torch.float32, device=logits.device)
    idxs = torch.empty(M, nb, dtype=torch.int64, device=logits.device)
    if M > 0:
        k["block_argmax"][(M, nb)](logits, logits.stride(0), vals, idxs, nb, V, BLOCK_SIZE=BLOCK)
    b = vals.argmax(dim=-1, keepdim=True)  # first maximal block
    return vals.gather(-1, b).squeeze(-1), idxs.gather(-1, b).squeeze(-1)


def greedy_verify(target, draft, cu, num_reqs, width):
    if not target.is_cuda:
        return greedy_verify_torch(target, draft, cu, num_reqs, width)
    k = _kernels()
    sampled = torch.empty(num_reqs, width, dtype=torch.int64, device=target.device)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=target.device)
    k["verify"][(num_reqs,)](sampled, sampled.stride(0), num_sampled, target.contiguous(),
                             draft.contiguous(), cu.contiguous(), num_warps=1)
    return sampled, num_sampled


def vocab_parallel_greedy(shard_logits: torch.Tensor, vocab_start: int, tp: int, all_gather) -> torch.Tensor:
    """shard_logits [M, Vs] (this rank's contiguous vocab range starting at vocab_start) -> [M] global argmax.
    all_gather(t [M, 4]) must return [M, tp * 4] in rank order (tensor_model_parallel_all_gather(t, dim=-1))."""
    val, idx = local_argmax(shard_logits)
    pairs = pack_pairs(val, idx + vocab_start)
    if tp == 1:
        return pairs[:, 1].to(torch.int64)
    return reduce_pairs(all_gather(pairs), tp)


# ---------------------------------------------------------------------------------------------------
# dispatch (rank-invariant: numpy request state + static config only)
# ---------------------------------------------------------------------------------------------------
class Stats:
    fast = 0
    full = 0
    checked = 0
    mismatch = 0
    reasons: dict = {}


def _reason(r):
    Stats.reasons[r] = Stats.reasons.get(r, 0) + 1
    return None


def resolve_head(model):
    """(language model, lm_head, logits_processor) of the target, or a string saying why not."""
    m = model
    for _ in range(6):
        if hasattr(m, "lm_head") and hasattr(m, "logits_processor"):
            break
        nxt = getattr(m, "language_model", None)
        if nxt is None and hasattr(m, "get_language_model"):
            try:
                nxt = m.get_language_model()
            except Exception:  # noqa: BLE001
                nxt = None
        if nxt is None or nxt is m:
            return "no lm_head/logits_processor found"
        m = nxt
    else:
        return "no lm_head/logits_processor found"
    lm_head, lp = m.lm_head, m.logits_processor
    if type(lp).__name__ != "LogitsProcessor":
        return f"logits processor is {type(lp).__name__}"
    if lp.logits_as_input or lp.soft_cap is not None or lp.scale != 1.0:
        return "logits_as_input / soft_cap / scale != 1"
    si = getattr(lm_head, "shard_indices", None)
    if si is None or not hasattr(lp, "_apply_head"):
        return "lm_head has no shard_indices"
    if si.num_org_vocab_padding != 0 or si.num_added_elements_padded != 0:
        return "vocab padding or added vocab"
    if int(lm_head.num_embeddings_padded) != int(lp.org_vocab_size):
        return f"gathered width {lm_head.num_embeddings_padded} != vocab {lp.org_vocab_size}"
    if getattr(lm_head, "bias", None) is not None:
        return "lm_head bias"
    return (m, lm_head, lp)


def plan(runner, input_batch, grammar_output):
    """None -> stock path. Otherwise ("reject", width) or ("sampler", 1)."""
    sampler = runner.sampler
    if sampler is None or grammar_output is not None:
        return _reason("grammar" if grammar_output is not None else "no sampler")
    if type(sampler).__name__ != "Sampler" or getattr(sampler, "compute_nans", False):
        return _reason("custom sampler / nan counting")
    head = getattr(runner, "_glm_tva_head", None)
    if head is None:
        head = resolve_head(runner.model)
        runner._glm_tva_head = head
        if isinstance(head, str):
            _log(f"fast path disabled for this runner: {head}")
    if isinstance(head, str):
        return _reason("head")
    idx_np = input_batch.idx_mapping_np
    if idx_np.size == 0 or (idx_np < 0).any():
        return _reason("empty / masked batch")
    st = sampler.sampling_states
    if not np.all(st.temperature.np[idx_np] == 0.0):
        return _reason("sampled rows")
    if sampler._requires_logits_processing(idx_np):
        return _reason("logits processing")
    if st.max_num_logprobs(idx_np) != NO_LOGPROBS:
        return _reason("logprobs")
    if sampler.logprob_token_ids_state.max_num_token_ids(idx_np) > 0:
        return _reason("logprob token ids")
    if input_batch.num_draft_tokens == 0 or runner.rejection_sampler is None:
        return ("sampler", 1)
    rs = runner.rejection_sampler
    if type(rs).__name__ != "RejectionSampler" or rs.synthetic_conditional_rates is not None:
        return _reason("custom / synthetic rejection sampler")
    dl = getattr(runner.speculator, "draft_logits", None) if runner.speculator is not None else None
    if dl is not None and dl.size(-1) < int(head[2].org_vocab_size):
        return _reason("draft vocab narrower than target")  # stock clamps the vocab to the draft's
    return ("reject", int(rs.num_speculative_steps) + 1)


def _dead_rows(input_batch):
    """GLM_VERIFY_CUT's dead-draft mask at the logits rows, or None."""
    mod = sys.modules.get("glm_verify_cut")
    if mod is None or getattr(mod, "S", None) is None or mod.S.dead is None:
        return None
    return mod.S.dead[input_batch.logits_indices]


def fast_sample(runner, hidden_states, input_batch, how):
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    _, lm_head, lp = runner._glm_tva_head
    h = hidden_states[input_batch.logits_indices]
    shard = lp._apply_head(lm_head, h, None)  # identical call to LogitsProcessor._get_logits
    target = vocab_parallel_greedy(shard, int(lm_head.shard_indices.org_vocab_start_index),
                                   int(lm_head.tp_size),
                                   lambda t: tensor_model_parallel_all_gather(t, dim=-1))
    kind, width = how
    num_reqs = input_batch.num_reqs
    if kind == "sampler":
        sampled = target.view(-1, 1)
        num_sampled = input_batch.seq_lens.new_ones(num_reqs)
    else:
        draft = input_batch.input_ids[input_batch.logits_indices]
        dead = _dead_rows(input_batch)
        if dead is not None:
            draft = draft.masked_fill(dead, -1)
        sampled, num_sampled = greedy_verify(target, draft, input_batch.cu_num_logits, num_reqs, width)
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled, input_batch.seq_lens, input_batch.cu_num_logits, input_batch.idx_mapping,
        runner.sampler.req_states.prefill_len.gpu)
    out = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                        num_sampled=num_sampled, num_rejected=num_rejected)
    return out, num_sampled, num_rejected


def _compare(fast, ref, rank0):
    fo, fns, fnr = fast
    ro, rns, rnr = ref
    ok_n = torch.equal(fns.to(torch.int64), rns.to(torch.int64)) and torch.equal(fnr.to(torch.int64),
                                                                                  rnr.to(torch.int64))
    a, b = fo.sampled_token_ids, ro.sampled_token_ids
    w = min(a.shape[1], b.shape[1])
    cols = torch.arange(w, device=a.device)[None, :]
    live = cols < rns.to(torch.int64)[:, None]
    ok_t = bool(((a[:, :w] == b[:, :w]) | ~live).all()) and a.shape == b.shape
    Stats.checked += 1
    if not (ok_n and ok_t):
        Stats.mismatch += 1
        if rank0 and Stats.mismatch <= 20:
            _log(f"MISMATCH #{Stats.mismatch}: num_sampled fast={fns.tolist()} ref={rns.tolist()} "
                 f"tokens fast={a.tolist()} ref={b.tolist()}")


def install(mod) -> None:
    cls = mod.GPUModelRunner
    orig_sample = cls.sample

    def _rank0():
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            return get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            return True

    def sample(self, hidden_states, input_batch, grammar_output):
        mode = _mode()
        if mode == "0":
            return orig_sample(self, hidden_states, input_batch, grammar_output)
        how = plan(self, input_batch, grammar_output)
        if how is None:
            Stats.full += 1
            res = orig_sample(self, hidden_states, input_batch, grammar_output)
        elif mode == "check":
            fast = fast_sample(self, hidden_states, input_batch, how)
            res = orig_sample(self, hidden_states, input_batch, grammar_output)
            _compare(fast, res, _rank0())
            Stats.fast += 1
        else:
            Stats.fast += 1
            res = fast_sample(self, hidden_states, input_batch, how)
        n = Stats.fast + Stats.full
        if LOG_EVERY and n % LOG_EVERY == 0 and _rank0():
            _log(f"steps fast={Stats.fast} full={Stats.full} reasons={Stats.reasons}"
                 + (f" checked={Stats.checked} mismatches={Stats.mismatch}" if mode == "check" else ""))
        return res

    sample.__wrapped__ = orig_sample
    cls.sample = sample
    _log(f"GPUModelRunner.sample hooked (mode={'check' if CHECK else 'fast'}, local={LOCAL_IMPL})")
