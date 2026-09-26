# SPDX-License-Identifier: Apache-2.0
"""LeversScheduler: SpecProbeScheduler + live scheduler-side levers (diagnostics/glm-levers-20260926).

--scheduler-cls glm_levers_sched.LeversScheduler   (profile: SCHEDULER_CLS=glm_levers_sched.LeversScheduler)

Everything here is decided in the single engine-core process and reaches every TP rank through the
scheduler output, so it is rank-invariant by construction. Speculation length never changes committed
tokens (greedy output is the target's argmax; sampled output keeps the target distribution); batch shape
can move greedy text in this engine the way any batching change does.

Inherited (overlay/spec_probe_scheduler.py): per-request vllm_xargs.spec_k and the live policy file
SPEC_PROBE_CONTROL (k_lo, k_hi, up, down, alpha, enabled, default_k). This class reads extra keys from the
same file on every mtime change:

  "mode": "per-request" | "batch-uniform" | "batch-max"
      per-request   (stock AdaptiveDraftScheduler) each request verifies its own k; the step drafts the
                    batch max. A step that mixes k_lo and k_hi requests has no uniform decode length.
      batch-uniform (adaptive_k_scheduler's mode) k_hi only if every request is high, else k_lo; the
                    draft length follows the same rule.
      batch-max     every request verifies the batch max k (what the drafter drafts anyway), so every
                    decode step is uniform and can replay a FULL decode graph; low-k requests pay extra
                    verify rows and may accept more.
  "end_drain": 0|1        glm_prefill_sched.END_DRAIN (jnardiello E29). Needs GLM_END_DRAIN=1 at boot so the
                          schedule() text edit is installed; 0 makes holds() return False = stock.
  "coalesce_ms": 0..5     glm_prefill_sched.COALESCE_S (E29 idle coalescing). Needs GLM_IDLE_COALESCE_MS>0
                          at boot so EngineCoreProc._process_input_queue is wrapped; 0 = stock.
  "seed": 0..1            EMA start value of a new request (VLLM_ADAPTIVE_K_SEED, stock 1.0: every request
                          starts at k_hi and needs ~6 low-acceptance steps to drop to k_lo).
  "signal": "pos"|"cond"  pos (stock): EMA of 1[draft k_lo accepted]. cond: EMA of 1[draft k_lo accepted] over the
                          steps whose draft k_lo-1 was accepted, i.e. P(pos k_lo | pos k_lo-1), which is what the
                          positions beyond k_lo look like (measured 09-26: code 0.84, json 0.93, harness prose 0.56).

Initial values (before any control file): GLM_LV_MODE (default per-request), GLM_LV_END_DRAIN (0),
GLM_LV_COALESCE_MS (0), GLM_LV_SIGNAL (pos); GLM_LV_POLICY='{...}' (JSON without spaces, same keys as the
control file) is applied at scheduler init, since start.sh pins VLLM_ADAPTIVE_K_* after EXTRA_ENV. So a boot with GLM_END_DRAIN=1 GLM_IDLE_COALESCE_MS=4 still starts as stock KSN.

Credits: jnardiello (AdaptiveKScheduler base, E29 end-drain / idle coalescing), our spec_probe_scheduler.
"""
from __future__ import annotations

import json
import os
import sys

import adaptive_k_scheduler as _aks
from adaptive_k_scheduler import _HAVE_VLLM, logger, placeholder_len
from spec_probe_scheduler import ProbeDraftLookup, request_override

MODES = ("per-request", "batch-uniform", "batch-max")
SIGNALS = ("pos", "cond")
LV = {
    "signal": (os.environ.get("GLM_LV_SIGNAL") or "pos").strip(),
    "mode": (os.environ.get("GLM_LV_MODE") or "per-request").strip(),
    "end_drain": os.environ.get("GLM_LV_END_DRAIN", "0").strip() == "1",
    "coalesce_ms": float(os.environ.get("GLM_LV_COALESCE_MS", "0") or 0),
}
if LV["mode"] not in MODES:
    raise ValueError(f"GLM_LV_MODE must be one of {MODES}")
STATS = {"steps": 0, "uniform_steps": 0, "mixed_steps": 0}

_orig_assign = _aks.assign_lengths


def assign_lengths(policy, req_ids, mode, engine_k):
    """adaptive_k_scheduler.assign_lengths with the live mode."""
    req_ids = list(req_ids)
    m = LV["mode"]
    if not req_ids:
        return {}
    if m == "batch-max":
        ks = {r: placeholder_len(policy.decide(r), engine_k) for r in req_ids}
        kmax = max(ks.values())
        out = {r: kmax for r in req_ids}
    elif m == "batch-uniform":
        out = _orig_assign(policy, req_ids, "batch-uniform", engine_k)
    else:
        out = _orig_assign(policy, req_ids, mode, engine_k)
    STATS["steps"] += 1
    STATS["uniform_steps" if len(set(out.values())) <= 1 else "mixed_steps"] += 1
    return out


def _apply_prefill_sched() -> None:
    ps = sys.modules.get("glm_prefill_sched")
    if ps is None:
        return
    ps.END_DRAIN = bool(LV["end_drain"])
    ps.COALESCE_S = max(0.0, min(float(LV["coalesce_ms"]), 5.0)) / 1000.0


class LvPolicy(_aks.AdaptiveKPolicy):
    """AdaptiveKPolicy with the optional conditional signal (LV["signal"] == "cond")."""

    def observe(self, req_id, num_accepted, num_draft):
        if LV["signal"] != "cond":
            return super().observe(req_id, num_accepted, num_draft)
        k = self.cfg.k_lo
        st = self._get(req_id)
        if num_draft < k or num_accepted < k - 1:
            return st.ema  # position k_lo not reached: no information about P(pos k_lo | pos k_lo-1)
        a = self.cfg.alpha
        st.ema = a * (1.0 if num_accepted >= k else 0.0) + (1.0 - a) * st.ema
        self.counters["observations"] += 1
        return st.ema


class LeversDraftLookup(ProbeDraftLookup):
    """Draft K for the step. batch-uniform: k_hi only if every running decode request is high."""

    def __getitem__(self, batch_size: int) -> int:
        s = self._s
        if LV["mode"] != "batch-uniform" or not s._ak_cfg.enabled or s._ak_failed:
            return super().__getitem__(batch_size)
        try:
            ks = []
            for req in s.running:
                if req.is_finished() or getattr(req, "is_prefill_chunk", False):
                    continue
                k = request_override(req)
                if k is None:
                    k = s._probe_default_k
                if k is None:
                    k = s._ak.decide(req.request_id)
                ks.append(min(int(k), s._ak_engine_k))
            if not ks:
                return self._fallback[batch_size]
            return s._ak_cfg.k_hi if all(k >= s._ak_cfg.k_hi for k in ks) else min(ks)
        except Exception:  # noqa: BLE001
            logger.exception("glm-levers: draft lookup failed, using the launch table")
            return self._fallback[batch_size]


if _HAVE_VLLM:
    from spec_probe_scheduler import SpecProbeScheduler

    class LeversScheduler(SpecProbeScheduler):  # type: ignore[misc]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            _aks.assign_lengths = assign_lengths
            self._ak = LvPolicy(self._ak_cfg)
            if isinstance(self.dynamic_sd_lookup, ProbeDraftLookup):
                self.dynamic_sd_lookup = LeversDraftLookup(self, self.dynamic_sd_lookup._fallback)
            _apply_prefill_sched()
            self._lv_calls = 0
            init = os.environ.get("GLM_LV_POLICY", "").strip()
            if init:  # boot-time policy (same keys as the control file); fatal if malformed
                self._lv_apply(json.loads(init))
            logger.info("glm-levers: LeversScheduler active %s (control file %s; prefill_sched %s)",
                        LV, self._probe_path, "loaded" if "glm_prefill_sched" in sys.modules else "absent")

        def _lv_apply(self, ctl: dict) -> None:
            """Apply a policy dict (control file or GLM_LV_POLICY): the SpecProbe keys + this class's keys."""
            from dataclasses import replace
            upd = {k: ctl[k] for k in ("k_lo", "k_hi", "up", "down", "alpha", "enabled", "seed") if k in ctl}
            if "enabled" in upd:
                upd["enabled"] = bool(upd["enabled"])
            for key in ("k_lo", "k_hi"):
                if key in upd:
                    upd[key] = min(int(upd[key]), self._ak_engine_k)
            if "default_k" in ctl:
                dk = ctl["default_k"]
                self._probe_default_k = None if dk is None else max(0, min(int(dk), self._ak_engine_k))
            if "mode" in ctl:
                if ctl["mode"] not in MODES:
                    raise ValueError(f"mode {ctl['mode']!r}")
                LV["mode"] = ctl["mode"]
            if "end_drain" in ctl:
                LV["end_drain"] = bool(int(ctl["end_drain"]))
            if "coalesce_ms" in ctl:
                LV["coalesce_ms"] = float(ctl["coalesce_ms"])
            fresh = False
            if "signal" in ctl:
                if ctl["signal"] not in SIGNALS:
                    raise ValueError(f"signal {ctl['signal']!r}")
                fresh = ctl["signal"] != LV["signal"]
                LV["signal"] = ctl["signal"]
            cfg = replace(self._ak_cfg, **upd)  # __post_init__ validates
            if cfg != self._ak_cfg or fresh:
                self._ak_cfg = cfg
                self._ak = LvPolicy(cfg)  # fresh EMAs: new policy, new state
            _apply_prefill_sched()
            ps = sys.modules.get("glm_prefill_sched")
            logger.info("glm-levers: now %s policy %s default_k=%s (prefill_sched END_DRAIN=%s COALESCE_S=%s) "
                        "stats %s", LV, self._ak_cfg.describe(), self._probe_default_k,
                        getattr(ps, "END_DRAIN", None), getattr(ps, "COALESCE_S", None), STATS)

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
                    self._lv_apply(json.load(f))
            except Exception:  # noqa: BLE001
                logger.exception("glm-levers: bad control file %s ignored", self._probe_path)

        def schedule(self, *args, **kwargs):
            self._lv_calls += 1
            if self._lv_calls % 4 == 2:  # one os.stat every 4 steps (the base re-reads every 32)
                self._probe_reload()
            if self._lv_calls % 2000 == 0:
                logger.info("glm-levers: %s stats %s", LV, STATS)
            return super().schedule(*args, **kwargs)

else:

    class LeversScheduler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("LeversScheduler needs vLLM")
