# SPDX-License-Identifier: Apache-2.0
"""glm-levers (diagnostics/glm-levers-20260926): small, independently gated decode levers + counters.

Every part is inert unless its variable is set. Registered from overlay/sitecustomize.py (one block).

GLM_LV_CGSTAT=1
    Counts the TARGET model's CUDA-graph dispatch per step (vllm.v1.worker.gpu.model_runner
    dispatch_cg_and_sync_dp): cg mode (FULL / NONE), requests, real tokens, padded graph tokens,
    uniform decode length, max query length, and the wall period to the next dispatch (a proxy for the
    step period of that batch shape). TP rank 0 rewrites GLM_LV_CGSTAT_FILE (default
    /cache/levers_cg.json) every 64 steps. Read-only diagnostics: no collective, no GPU work.
    Question it answers (replicated-layer eligibility review): with FULL_DECODE_ONLY + compile mode 0 there are no
    PIECEWISE graphs, so a step whose requests verify different k (per-request adaptive k) dispatches NONE
    and runs eager.

GLM_LV_SPLIT_FIX=1
    glm_ds_split.eligible() rejects every module whose tp_size != 1. In this image LinearBase stores the
    real TP size on ReplicatedLinear too (linear.py: disable_tp has no effect for replicated layers), so
    the drafter's model.fc (4096 x 20480 BF16, 168 MB read per rank per step) and the 11 DSA
    indexer.wq_b (4096 x 1536 BF16, 138 MB per step) were logged "not replicated" and never split.
    The fix accepts ReplicatedLinear regardless of tp_size; every other check (BF16, unquantized, no
    bias, bit-exact per-row-count verification MIN-reduced over TP) is unchanged.
    Credit: the split is our DS4.1 replicated_split adapter, ported in overlay/glm_ds_split.py.
    GLM_LV_SPLIT_INEXACT=fc (comma list of parts): skip the per-row-count bit-exact check for those parts only.
    Measured 09-26: the drafter fc slice GEMM (4096 x 20480 -> 1024 columns) differs from the full GEMM's
    columns at every row count (other reduction order), so the exact check never admits it. fc feeds only the
    drafter; the target verifies every draft, so a non-bit-identical fc changes proposals only (acceptance is
    measured), never the committed tokens. Target parts (wq_b, qkv_a) keep the exact check.

GLM_LV_ARGMAX_MINTOK=1   (needs GLM_TARGET_VOCAB_ARGMAX=1)
    overlay/glm_target_argmax.py takes the vocab-parallel greedy fast path only when no request needs logits
    processing. A request with min_tokens > 0 and stop ids (sparkDash sends min_tokens = max_tokens +
    ignore_eos; all_stop_token_ids keeps EOS even with ignore_eos) sets LogitBiasState.use_logit_bias, so
    every such step takes the stock path: full-vocab logits all-gather, fp32 copy, bias kernel, rejection
    kernel. When the ONLY logit bias in the batch is min-tokens (no allowed_token_ids, no logit_bias) and
    everything else is plain greedy, this admits the step to the fast path and applies the stock rule on the
    local vocab shard before the local argmax: stop id -> -inf on rows with pos + 1 < min_len
    (sample/logit_bias.py _bias_kernel, same condition). bf16 -> fp32 is exact and -inf is -inf, so the
    argmax (ties -> lowest id) is the stock one. Decision from per-request CPU state only (rank-invariant).

GLM_LV_DRAFT_FP8_KV=1   (needed with a block-FP8 drafter, profiles/lv-ds.env)
    DFlashQwen3Model._build_context_kv_buffers (qwen3_dflash.py) concatenates qkv_proj.weight[q_size:] of every
    draft layer into one BF16 GEMM weight for the context-KV precompute, at the end of load_weights (before
    process_weights_after_loading). With the drafter's own quantization_config (fp8, block 128) that tensor is
    raw F8_E4M3 without its scales. This builds the same buffer from the dequantized rows instead
    (F8_E4M3 value x weight_scale_inv block, rounded once to BF16). BF16 drafters take the stock path.

Scheduler-side levers live in glm_levers_sched.py (LeversScheduler, a SpecProbeScheduler subclass with a
live control file); they run only in the engine-core process, so they are rank-invariant by construction.
"""
from __future__ import annotations

import json
import os
import sys
import time


def _on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "off", "false", "no")


CGSTAT = _on("GLM_LV_CGSTAT")
SPLIT_FIX = _on("GLM_LV_SPLIT_FIX")
SPLIT_INEXACT = {x.strip() for x in os.environ.get("GLM_LV_SPLIT_INEXACT", "").split(",") if x.strip()}
if SPLIT_INEXACT - {"fc"}:
    raise ValueError("GLM_LV_SPLIT_INEXACT may only name drafter parts (fc); target parts stay bit-exact")
ARGMAX_MINTOK = _on("GLM_LV_ARGMAX_MINTOK")
DRAFT_FP8_KV = _on("GLM_LV_DRAFT_FP8_KV")
DFLASH = "vllm.model_executor.models.qwen3_dflash"
CGSTAT_FILE = os.environ.get("GLM_LV_CGSTAT_FILE", "/cache/levers_cg.json")
RUNNER = "vllm.v1.worker.gpu.model_runner"


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-levers: {msg}\n")
    sys.stderr.flush()


# ------------------------------------------------------------------------------------------------
# cg-mode counter
# ------------------------------------------------------------------------------------------------
class CG:
    rank0: bool | None = None
    notes = 0
    last_t: float | None = None
    last_key: str | None = None
    by_key: dict = {}      # key -> [steps, period_sum_s, period_n]
    by_mode: dict = {}     # mode -> [steps, real_tokens, padded_tokens]


def _is_rank0() -> bool:
    if CG.rank0 is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            CG.rank0 = get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            return False
    return CG.rank0


def _dump() -> None:
    doc = {"t": time.time(), "notes": CG.notes, "by_mode": CG.by_mode,
           "by_key": {k: v for k, v in sorted(CG.by_key.items(), key=lambda kv: -kv[1][0])[:400]}}
    tmp = CGSTAT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, CGSTAT_FILE)


def note(desc, num_reqs, num_tokens, uniform, max_query_len, need_eager) -> None:
    if need_eager or num_tokens <= 0:
        return
    import torch
    if torch.cuda.is_current_stream_capturing() or not _is_rank0():
        return
    now = time.perf_counter()
    if CG.last_key is not None and CG.last_t is not None:
        dt = now - CG.last_t
        if dt < 1.0:  # idle gaps are not step periods
            e = CG.by_key[CG.last_key]
            e[1] += dt
            e[2] += 1
    mode = getattr(desc.cg_mode, "name", str(desc.cg_mode))
    padded = int(desc.num_tokens) - int(num_tokens)
    key = f"{mode}|r{num_reqs}|t{num_tokens}|p{padded}|u{uniform}|q{max_query_len}"
    e = CG.by_key.setdefault(key, [0, 0.0, 0])
    e[0] += 1
    m = CG.by_mode.setdefault(mode, [0, 0, 0])
    m[0] += 1
    m[1] += int(num_tokens)
    m[2] += max(padded, 0)
    CG.last_key, CG.last_t = key, now
    CG.notes += 1
    if CG.notes % 64 == 0:
        try:
            _dump()
        except Exception as exc:  # noqa: BLE001
            if CG.notes <= 64:
                _log(f"cgstat dump failed: {exc!r}")


def install_cgstat(mod) -> None:
    orig = mod.dispatch_cg_and_sync_dp
    if getattr(orig, "_glm_lv", False):
        return

    def dispatch_cg_and_sync_dp(*args, **kwargs):
        res = orig(*args, **kwargs)
        try:
            desc = res[0]
            num_reqs, num_tokens, uniform = args[1], args[2], args[3]
            note(desc, num_reqs, num_tokens, uniform, kwargs.get("max_query_len"),
                 bool(kwargs.get("need_eager", False)))
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break a step
            if CG.notes == 0:
                _log(f"cgstat note failed: {exc!r}")
        return res

    dispatch_cg_and_sync_dp._glm_lv = True
    mod.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
    _log(f"cgstat armed (target dispatch, rank 0 -> {CGSTAT_FILE})")


# ------------------------------------------------------------------------------------------------
# split eligibility fix
# ------------------------------------------------------------------------------------------------
def install_split_fix() -> None:
    import glm_ds_split as gs
    if getattr(gs.eligible, "_glm_lv", False):
        return
    orig = gs.eligible

    def eligible(module):
        why = orig(module)
        # "not replicated" is the last check in eligible(): every earlier check passed
        if why == "not replicated" and type(module).__name__ == "ReplicatedLinear" \
                and not getattr(module, "gather_output", False):
            return None
        return why

    eligible._glm_lv = True
    gs.eligible = eligible
    _log("split fix armed: ReplicatedLinear is splittable regardless of LinearBase.tp_size")
    if SPLIT_INEXACT:
        orig_classify, orig_check = gs.classify, gs.check_rows

        def classify(name, module):
            part = orig_classify(name, module)
            if part is not None:
                module._glm_lv_part = part
            return part

        def check_rows(module, w_slice, c0, c1, max_rows, device, seeds=(0, 1)):
            if getattr(module, "_glm_lv_part", None) in SPLIT_INEXACT:
                return [True] * max_rows
            return orig_check(module, w_slice, c0, c1, max_rows, device, seeds)

        gs.classify, gs.check_rows = classify, check_rows
        _log(f"split: bit-exact check skipped for drafter part(s) {sorted(SPLIT_INEXACT)}")


# ------------------------------------------------------------------------------------------------
# min_tokens admission for the vocab-parallel greedy fast path
# ------------------------------------------------------------------------------------------------
class MT:
    steps = 0
    logged = False


def requires_logits_processing_wo_bias(s, idx) -> bool:
    """Sampler._requires_logits_processing (image hash 49aa83c34acadedd) without the logit-bias clause."""
    import numpy as np
    if np.any(s.penalties_state.use_penalty[idx]):
        return True
    if np.any(s.bad_words_state.num_bad_words.np[idx] > 0):
        return True
    st = s.sampling_states
    t = st.temperature.np[idx]
    if np.any((t != 0.0) & (t != 1.0)):
        return True
    if np.any(st.min_p.np[idx] != 0.0):
        return True
    if np.any(st.top_k.np[idx] != st.vocab_size):
        return True
    return bool(np.any(st.top_p.np[idx] != 1.0))


def mintok_only(s, idx) -> bool:
    """Some request uses logit bias, and all of it is min-tokens stop masking."""
    lb = s.logit_bias_state
    return (bool(lb.use_logit_bias[idx].any())
            and not bool((lb.num_allowed_token_ids.np[idx] > 0).any())
            and not bool((lb.num_logit_bias.np[idx] > 0).any()))


def mask_stop_torch(out, start, eidx, pos, min_lens, nstop, stop_ids) -> None:
    """Reference (CPU tests): out[row, stop - start] = -inf where pos + 1 < min_len, stop inside the shard."""
    import torch
    rows_n, width = out.shape
    req = eidx.long()
    n = nstop[req].long()
    active = ((pos.long() + 1) < min_lens[req].long()) & (n > 0)
    loc = stop_ids[req].long() - start
    col = torch.arange(loc.shape[1], device=out.device)
    m = active[:, None] & (col[None, :] < n[:, None]) & (loc >= 0) & (loc < width)
    rows = torch.arange(rows_n, device=out.device)[:, None].expand_as(loc)
    out[rows[m], loc[m]] = float("-inf")


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _mintok_mask_kernel(out_ptr, out_stride, start, width, eidx_ptr, pos_ptr, min_lens_ptr,
                                nstop_ptr, stop_ptr, stop_stride, BLOCK: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            req = tl.load(eidx_ptr + row).to(tl.int64)
            n = tl.load(nstop_ptr + req)
            pos = tl.load(pos_ptr + row)
            min_len = tl.load(min_lens_ptr + req)
            if n > 0 and pos + 1 < min_len:
                b = tl.arange(0, BLOCK)
                m = b < n
                ids = tl.load(stop_ptr + req * stop_stride + b, mask=m, other=0).to(tl.int64)
                loc = ids - start
                m = m & (loc >= 0) & (loc < width)
                tl.store(out_ptr + row * out_stride + loc, float("-inf"), mask=m)

        _KERNEL = _mintok_mask_kernel
    return _KERNEL


def mask_stop(out, start, eidx, pos, min_lens, nstop, stop_ids) -> None:
    if not out.is_cuda:
        return mask_stop_torch(out, start, eidx, pos, min_lens, nstop, stop_ids)
    import triton
    rows = out.shape[0]
    if rows == 0:
        return
    assert out.stride(1) == 1
    _kernel()[(rows,)](out, out.stride(0), int(start), out.shape[1], eidx, pos, min_lens, nstop, stop_ids,
                       stop_ids.stride(0), BLOCK=triton.next_power_of_2(stop_ids.shape[1]))


def install_mintok(_mod) -> None:
    import glm_target_argmax as ta
    if getattr(ta.plan, "_glm_lv", False):
        return
    orig_plan, orig_fast = ta.plan, ta.fast_sample

    def plan(runner, input_batch, grammar_output):
        runner._glm_lv_mintok = False
        s = runner.sampler
        idx = input_batch.idx_mapping_np
        if (s is not None and grammar_output is None and idx.size and not (idx < 0).any()
                and mintok_only(s, idx)):
            s._requires_logits_processing = lambda i, _s=s: requires_logits_processing_wo_bias(_s, i)
            try:
                res = orig_plan(runner, input_batch, grammar_output)
            finally:
                del s._requires_logits_processing
            if res is not None:
                runner._glm_lv_mintok = True
                MT.steps += 1
                if not MT.logged:
                    MT.logged = True
                    _log(f"argmax min_tokens admission engaged ({res[0]})")
            return res
        return orig_plan(runner, input_batch, grammar_output)

    def fast_sample(runner, hidden_states, input_batch, how):
        if not getattr(runner, "_glm_lv_mintok", False):
            return orig_fast(runner, hidden_states, input_batch, how)
        _, lm_head, lp = runner._glm_tva_head
        lb = runner.sampler.logit_bias_state
        start = int(lm_head.shard_indices.org_vocab_start_index)
        pos = input_batch.positions[input_batch.logits_indices]
        eidx = input_batch.expanded_idx_mapping
        orig_apply = lp._apply_head

        def masked(head, h, bias):
            out = orig_apply(head, h, bias)
            mask_stop(out, start, eidx, pos, lb.min_lens.gpu, lb.num_stop_token_ids.gpu, lb.stop_token_ids.gpu)
            return out

        lp._apply_head = masked
        try:
            return orig_fast(runner, hidden_states, input_batch, how)
        finally:
            del lp._apply_head

    plan._glm_lv = True
    ta.plan, ta.fast_sample = plan, fast_sample
    _log("argmax min_tokens admission armed (stop ids masked on the local vocab shard)")


# ------------------------------------------------------------------------------------------------
# block-FP8 drafter: context-KV fused buffer from dequantized weights
# ------------------------------------------------------------------------------------------------
def dequant_block_rows(w, scale_inv, row0: int, block: int = 128):
    """Rows [row0:] of a block-FP8 weight as fp32: q * scale_inv[row_block, col_block]."""
    import torch
    assert row0 % block == 0, row0
    q = w[row0:].to(torch.float32)
    sc = scale_inv[row0 // block:].to(torch.float32)
    sc = sc.repeat_interleave(block, 0)[: q.shape[0]].repeat_interleave(block, 1)[:, : q.shape[1]]
    return q * sc


def install_draft_fp8_kv(mod) -> None:
    import torch
    cls = mod.DFlashQwen3Model
    if getattr(cls._build_context_kv_buffers, "_glm_lv", False):
        return
    orig = cls._build_context_kv_buffers

    def _build_context_kv_buffers(self, layers_attn, has_bias):
        w0 = layers_attn[0].qkv_proj.weight
        if w0.dtype != torch.float8_e4m3fn:
            return orig(self, layers_attn, has_bias)
        dt = self.hidden_norm.weight.dtype
        self._hidden_norm_weight = self.hidden_norm.weight.data
        kv = []
        for a in layers_attn:
            si = getattr(a.qkv_proj, "weight_scale_inv", None)
            if si is None or a.qkv_proj.weight.dtype != torch.float8_e4m3fn:
                raise RuntimeError("glm-levers: drafter qkv_proj is FP8 without block scales; refusing to build "
                                   "the context-KV buffer from raw FP8 bytes")
            kv.append(dequant_block_rows(a.qkv_proj.weight.data, si.data, a.q_size).to(dt))
        self._fused_kv_weight = torch.cat(kv, dim=0).contiguous()
        self._fused_kv_bias = (torch.cat([a.qkv_proj.bias[a.q_size:] for a in layers_attn], dim=0)
                               if has_bias else None)
        self._k_norm_weights = torch.stack([a.k_norm.weight.data for a in layers_attn], dim=0).contiguous()
        _log(f"draft context-KV buffer built from dequantized block-FP8 k/v rows {tuple(self._fused_kv_weight.shape)} "
             f"{self._fused_kv_weight.dtype}")

    _build_context_kv_buffers._glm_lv = True
    cls._build_context_kv_buffers = _build_context_kv_buffers
    _log("draft FP8 context-KV fix armed")


# ------------------------------------------------------------------------------------------------
# registration
# ------------------------------------------------------------------------------------------------
HOOKS: dict = {}


def _runner_hooks(mod) -> None:
    if CGSTAT:
        install_cgstat(mod)
    if ARGMAX_MINTOK:
        install_mintok(mod)


if CGSTAT or ARGMAX_MINTOK:
    HOOKS[RUNNER] = _runner_hooks
if DRAFT_FP8_KV:
    HOOKS[DFLASH] = install_draft_fp8_kv


def register_early() -> None:
    """Parts that patch our own pure-Python overlay modules (no engine import needed)."""
    if SPLIT_FIX:
        install_split_fix()
