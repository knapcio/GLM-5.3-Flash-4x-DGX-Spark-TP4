"""Per-request verify length and a live policy file for drafter A/B runs (GLM-5.3-Flash, vLLM 487ecf187).

Purpose: sweep speculation settings inside ONE boot instead of one boot per setting.

It subclasses our AdaptiveDraftScheduler (overlay/adaptive_draft_scheduler.py, itself on jnardiello's
AdaptiveKScheduler) and adds two things. Both are decided in the single engine-core process, so every
TP rank sees the same batch (rank-invariant by construction).

1. Per-request override through the OpenAI field ``vllm_xargs`` (-> SamplingParams.extra_args):
     {"vllm_xargs": {"spec_k": 3}}   verify exactly 3 drafts for this request (0..num_speculative_tokens)
   Requests without ``spec_k`` keep the adaptive policy. For timing runs send the same spec_k to every
   concurrent request: a uniform k stays on the FULL decode graph of family k (the family must be listed in
   num_speculative_tokens_per_batch_size, see RESULTS.md); mixed k runs the PIECEWISE graph.

2. Live policy file (env ``SPEC_PROBE_CONTROL``, default /cache/spec_policy.json on the head, which is
   the bind-mounted $OVERLAY_REMOTE/cache). Re-read when its mtime changes (checked every 32 schedule
   calls). Keys, all optional:
     {"k_lo": 3, "k_hi": 7, "up": 0.58, "down": 0.42, "alpha": 0.15, "enabled": 1, "default_k": null}
   ``default_k`` (int) forces every request without its own spec_k to that k (policy bypass).
   A bad file is logged and ignored; the previous policy stays.

Lossless: only how many drafts get verified changes, never the committed tokens (greedy output is
identical; sampled output keeps the target distribution).

Use: copy next to adaptive_draft_scheduler.py / adaptive_k_scheduler.py in the overlay dir and pass
``--scheduler-cls spec_probe_scheduler.SpecProbeScheduler``.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

from adaptive_draft_scheduler import AdaptiveDraftScheduler, PolicyDraftLookup
from adaptive_k_scheduler import _HAVE_VLLM, AdaptiveKPolicy, logger

_CONTROL_KEYS = ("k_lo", "k_hi", "up", "down", "alpha", "enabled", "default_k")


def request_override(request) -> int | None:
    """spec_k from the request's extra_args, or None (pure helper, tolerant of missing fields)."""
    sp = getattr(request, "sampling_params", None)
    extra = getattr(sp, "extra_args", None) if sp is not None else None
    if not extra:
        return None
    raw = extra.get("spec_k")
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return None


class ProbeDraftLookup(PolicyDraftLookup):
    """Step draft K = max over running requests of (override or policy)."""

    def __getitem__(self, batch_size: int) -> int:
        # Only sizes the base AsyncScheduler's placeholders; the per-request lengths are then
        # re-sized in SpecProbeScheduler._ak_size_placeholders, and the drafter always drafts
        # num_speculative_tokens. Kept consistent (batch max) for the logs.
        s = self._s
        try:
            k_step = 0
            for req in s.running:
                if req.is_finished() or getattr(req, "is_prefill_chunk", False):
                    continue
                k = request_override(req)
                if k is None:
                    k = s._probe_default_k
                if k is None:
                    k = s._ak.decide(req.request_id) if s._ak_cfg.enabled else s._ak_engine_k
                k_step = max(k_step, min(int(k), s._ak_engine_k))
            return k_step if k_step > 0 else self._fallback[batch_size]
        except Exception:  # noqa: BLE001
            logger.exception("spec-probe: draft lookup failed, using the launch table")
            return self._fallback[batch_size]


if _HAVE_VLLM:

    class SpecProbeScheduler(AdaptiveDraftScheduler):  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._probe_default_k: int | None = None
            self._probe_path = os.environ.get("SPEC_PROBE_CONTROL", "/cache/spec_policy.json")
            self._probe_mtime = None
            self._probe_calls = 0
            self._probe_hist: dict[int, int] = {}
            if isinstance(self.dynamic_sd_lookup, PolicyDraftLookup):
                self.dynamic_sd_lookup = ProbeDraftLookup(self, self.dynamic_sd_lookup._fallback)
            logger.info("spec-probe: active; per-request vllm_xargs.spec_k honoured; control file %s",
                        self._probe_path)

        # -- live policy file ------------------------------------------------
        def _probe_reload(self) -> None:
            try:
                st = os.stat(self._probe_path)
            except FileNotFoundError:
                return
            if st.st_mtime == self._probe_mtime:
                return
            self._probe_mtime = st.st_mtime
            try:
                with open(self._probe_path) as f:
                    ctl = json.load(f)
                upd = {k: ctl[k] for k in _CONTROL_KEYS if k in ctl and k != "default_k"}
                if "enabled" in upd:
                    upd["enabled"] = bool(upd["enabled"])
                for key in ("k_lo", "k_hi"):
                    if key in upd:
                        upd[key] = min(int(upd[key]), self._ak_engine_k)
                cfg = replace(self._ak_cfg, **upd)  # __post_init__ validates
                dk = ctl.get("default_k")
                self._probe_default_k = None if dk is None else max(0, min(int(dk), self._ak_engine_k))
                if cfg != self._ak_cfg:
                    self._ak_cfg = cfg
                    self._ak = AdaptiveKPolicy(cfg)  # fresh EMAs: new policy, new state
                logger.info("spec-probe: policy now %s default_k=%s", cfg.describe(), self._probe_default_k)
            except Exception:  # noqa: BLE001
                logger.exception("spec-probe: bad control file %s ignored", self._probe_path)

        def schedule(self, *args, **kwargs):
            self._probe_calls += 1
            if self._probe_calls % 32 == 1:
                self._probe_reload()
            return super().schedule(*args, **kwargs)

        # -- per-request override after the policy sized the placeholders ------
        def _ak_size_placeholders(self, scheduler_output) -> None:
            if self._ak_cfg.enabled:
                super()._ak_size_placeholders(scheduler_output)
            reqs = self.requests
            for req_id in scheduler_output.num_scheduled_tokens:
                request = reqs.get(req_id)
                if request is None or request.is_finished() or getattr(request, "is_prefill_chunk", False):
                    continue
                if not request.spec_token_ids:
                    continue
                k = request_override(request)
                if k is None:
                    k = self._probe_default_k
                if k is None:
                    continue
                k = min(k, self._ak_engine_k)
                request.spec_token_ids = self._ak_placeholders[k]
                self._probe_hist[k] = self._probe_hist.get(k, 0) + 1

        def _update_after_schedule(self, scheduler_output) -> None:
            # The base returns early when the policy is disabled; overrides must still apply.
            super()._update_after_schedule(scheduler_output)
            if not self._ak_cfg.enabled and not self._ak_failed:
                try:
                    self._ak_size_placeholders(scheduler_output)
                except Exception:  # noqa: BLE001
                    self._ak_failed = True
                    logger.exception("spec-probe: override sizing failed")

else:

    class SpecProbeScheduler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("SpecProbeScheduler needs vLLM")
