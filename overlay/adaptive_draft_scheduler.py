"""Content-adaptive DRAFT length for DFlash2 on top of AdaptiveKScheduler (variant J, 2026-09-10).

AdaptiveKScheduler only trims how many draft tokens each request *verifies*; the drafter
still produces ``num_speculative_tokens`` every step, and that draft forward is the fixed
tax that made k=7 lose on prose (H) while winning on code. This subclass also chooses how
many tokens the drafter *produces* each step, from the same per-request acceptance EMA:

  step K = max over running decode requests of policy.decide(req) in {k_lo, k_hi}

so a lone prose stream drafts+verifies k_lo (3), a lone code stream k_hi (7), and a mixed
batch drafts k_hi while prose requests verify k_lo (base-class trimming). vLLM reads the
per-step K from ``self.dynamic_sd_lookup[batch_size]`` (scheduler.py:1203-1207); we replace
that list with an object whose __getitem__ ignores batch size and returns the policy K.
CUDA-graph families are captured for every K listed in num_speculative_tokens_per_batch_size
(gpu/cudagraph_utils.py:196-216), so the launch table must list k_lo, k_hi (and 5).
Lossless: speculation never changes outputs; only speed.

Use: --scheduler-cls adaptive_draft_scheduler.AdaptiveDraftScheduler with the same
VLLM_ADAPTIVE_K_* env knobs (VLLM_ADAPTIVE_K_LO=3, VLLM_ADAPTIVE_K_HI=7 recommended).
"""
from __future__ import annotations

from adaptive_k_scheduler import AdaptiveKScheduler, _HAVE_VLLM, logger


class PolicyDraftLookup:
    """Stands in for the dense batch_size -> K list built from the launch table."""

    def __init__(self, scheduler, fallback):
        self._s = scheduler
        self._fallback = fallback  # original dense list, used if the policy is off/failed
        self.calls = 0
        self.hist: dict[int, int] = {}
        import os
        cap = os.environ.get("VLLM_ADAPTIVE_DRAFT_CAP", "").strip()
        self._cap = tuple(int(x) for x in cap.split(":")) if cap else None

    def __getitem__(self, batch_size: int) -> int:
        s = self._s
        if not s._ak_cfg.enabled or s._ak_failed:
            return self._fallback[batch_size]
        try:
            k = 0
            for req in s.running:
                if req.is_finished() or getattr(req, "is_prefill_chunk", False):
                    continue
                k = max(k, int(s._ak.decide(req.request_id)))
            if k <= 0:
                k = s._ak_cfg.k_hi
            k = min(k, s._ak_engine_k)
            # Optional cap for larger batches (multi-agent): VLLM_ADAPTIVE_DRAFT_CAP="3:5"
            # means "at batch size >= 3, draft at most 5". Verify cost scales with K x batch.
            if self._cap and batch_size >= self._cap[0]:
                k = min(k, self._cap[1])
        except Exception:  # noqa: BLE001 - never let the policy kill the engine core
            s._ak_failed = True
            logger.exception("adaptive-draft: K selection failed, falling back to the launch table")
            return self._fallback[batch_size]
        self.calls += 1
        self.hist[k] = self.hist.get(k, 0) + 1
        if s._ak_cfg.log_every and self.calls % (s._ak_cfg.log_every * 5) == 0:
            logger.info("adaptive-draft: step-K histogram %s", dict(sorted(self.hist.items())))
        return k

    def __len__(self):
        return len(self._fallback)


if _HAVE_VLLM:

    class AdaptiveDraftScheduler(AdaptiveKScheduler):  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if self.dynamic_sd_lookup is None:
                logger.error(
                    "adaptive-draft: num_speculative_tokens_per_batch_size is not set, so no "
                    "per-K graph families exist; draft length stays fixed at %s", self.num_spec_tokens)
                return
            self.dynamic_sd_lookup = PolicyDraftLookup(self, list(self.dynamic_sd_lookup))
            logger.info(
                "adaptive-draft: AdaptiveDraftScheduler active; per-step draft K in {%s,%s} from "
                "acceptance EMA, batch max; launch table only registers graph families; cap=%s",
                self._ak_cfg.k_lo, self._ak_cfg.k_hi, self.dynamic_sd_lookup._cap)

else:  # policy-only import

    class AdaptiveDraftScheduler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("AdaptiveDraftScheduler needs vLLM")
