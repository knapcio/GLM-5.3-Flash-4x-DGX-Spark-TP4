# SPDX-License-Identifier: Apache-2.0
"""GLM_ROCE_PROXY_CPUS: pin RoCEnante's RDMA proxy thread to one performance core. Default OFF. Exact.

Every all-reduce passes through the proxy thread (b12x `_roce_proxy.c`): it spins on the doorbell in pinned
memory and posts the RDMA writes, so its wake-up and posting latency sit on the critical path of each of the
~100 collectives per decode step, and its jitter becomes rank-arrival skew at the next collective. The thread
is created with the process's affinity, so the scheduler may run it on a Cortex-A725 efficiency core (GB10:
10 X925 + 10 A725) or move it between cores. This module pins it, and optionally takes that core away from
the process's other threads.

Knobs (read at proxy start):
  GLM_ROCE_PROXY_CPUS=""|auto|<list>   "" = off (default). auto = the highest-numbered performance core in the
                                       process's affinity. A list ("9" or "9,19") pins to exactly those.
  GLM_ROCE_PROXY_ISOLATE=0|1           1 = remove the proxy core(s) from every other thread of this process
                                       (threads created later inherit the main thread's reduced mask).
  GLM_ROCE_PROXY_BIG="5-9,15-19"       performance cores when /sys cpu_capacity is unavailable.

Mechanism: wrap b12x `_proxy.Proxy.start`; the proxy thread is the task id that appears in /proc/self/task
across the call. Only thread placement changes; the data path and every result are untouched.

In-boot A/B (added for the 2026-09-28 speed screen): with overlay/glm_ab.py armed, GLM_ROCE_PROXY_CPUS is a "raw"
switch read through glm_ab (the union writes "1" into os.environ, so the value is never read from there while
armed). The proxy threads and their original affinity are recorded at start whatever the variant; a glm_ab switch
hook re-applies the runtime variant's value (pin, or restore the original mask) on every rank after each switch.
ISOLATE is not undone by a switch: leave it off in an A/B.
"""
from __future__ import annotations

import os
import sys

ENV = "GLM_ROCE_PROXY_CPUS"
ENV_ISO = "GLM_ROCE_PROXY_ISOLATE"
ENV_BIG = "GLM_ROCE_PROXY_BIG"
ENV_RING = "GLM_ROCE_RING"
TARGETS = {"roce": "b12x.comm.roce._proxy", "roce_ring": "b12x.comm.roce_ring._proxy"}
TARGET = TARGETS["roce"]  # default target; ring mode is selected by target_module()


def target_module(environ=None) -> str:
    """The b12x proxy module to pin: the ring variant under ``GLM_ROCE_RING=1``."""
    env = os.environ if environ is None else environ
    return TARGETS["roce_ring"] if env.get(ENV_RING, "0").strip() == "1" else TARGETS["roce"]


def parse_cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def big_cores(sysfs: str = "/sys/devices/system/cpu", fallback: str | None = None) -> set[int]:
    caps = {}
    try:
        for d in os.listdir(sysfs):
            if d.startswith("cpu") and d[3:].isdigit():
                p = os.path.join(sysfs, d, "cpu_capacity")
                if os.path.exists(p):
                    with open(p) as f:
                        caps[int(d[3:])] = int(f.read().strip())
    except OSError:
        caps = {}
    if caps:
        top = max(caps.values())
        return {c for c, v in caps.items() if v == top}
    return parse_cpus(fallback or os.environ.get(ENV_BIG, "5-9,15-19"))


def choose(spec: str, affinity: set[int], big: set[int]) -> set[int]:
    """The core set for the proxy thread, or an empty set (= leave it alone)."""
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "no", "false"):
        return set()
    if spec == "auto":
        cand = sorted(affinity & big) or sorted(affinity)
        return {cand[-1]} if cand else set()
    cpus = parse_cpus(spec)
    return cpus & affinity if affinity else cpus


def new_tids(before: set[int], after: set[int]) -> set[int]:
    return set(after) - set(before)


def _tasks() -> set[int]:
    try:
        return {int(t) for t in os.listdir("/proc/self/task")}
    except OSError:
        return set()


def apply(tids: set[int], cpus: set[int], isolate: bool) -> list[str]:
    msgs = []
    for tid in tids:
        os.sched_setaffinity(tid, cpus)
        msgs.append(f"proxy tid {tid} -> cpus {sorted(cpus)}")
    if isolate and cpus:
        for tid in _tasks() - set(tids):
            try:
                cur = os.sched_getaffinity(tid)
                rest = cur - cpus
                if rest and rest != cur:
                    os.sched_setaffinity(tid, rest)
            except OSError:
                pass
        msgs.append(f"other threads of pid {os.getpid()} moved off {sorted(cpus)}")
    return msgs


_STATE = {"tids": {}, "hooked": False, "pinned": None}   # tid -> original affinity


def _spec() -> str:
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return str(ab.env(ENV, "") or "")
    return os.environ.get(ENV, "")


def apply_current(variant=None) -> None:
    """Pin the recorded proxy threads per the current (A/B) value, or give them back their original mask."""
    tids = {t: m for t, m in _STATE["tids"].items() if os.path.exists(f"/proc/self/task/{t}")}
    if not tids:
        return
    cpus = choose(_spec(), os.sched_getaffinity(0), big_cores())
    key = tuple(sorted(cpus))
    if key == _STATE["pinned"]:
        return
    if cpus:
        for m in apply(set(tids), cpus, os.environ.get(ENV_ISO, "0") == "1"):
            sys.stderr.write(f"glm-roce-proxy-pin: {m} (variant {variant})\n")
    else:
        for tid, mask in tids.items():
            os.sched_setaffinity(tid, mask)
        sys.stderr.write(f"glm-roce-proxy-pin: proxy tids {sorted(tids)} back to their original masks "
                         f"(variant {variant})\n")
    _STATE["pinned"] = key


def _install(mod) -> None:
    cls = mod.Proxy
    if getattr(cls, "_glm_pin", False):
        return
    cls._glm_pin = True
    orig = cls.start

    def start(self):
        before = _tasks()
        orig(self)
        tids = new_tids(before, _tasks())
        if not tids:
            sys.stderr.write("glm-roce-proxy-pin: nothing pinned (no new proxy thread)\n")
            return
        for tid in tids:
            try:
                _STATE["tids"][tid] = os.sched_getaffinity(tid)
            except OSError:
                pass
        _STATE["pinned"] = None
        ab = sys.modules.get("glm_ab")
        if ab is not None and getattr(ab, "ACTIVE", False) and not _STATE["hooked"]:
            ab.SWITCH_HOOKS.append(apply_current)
            _STATE["hooked"] = True
        cpus = choose(_spec(), os.sched_getaffinity(0), big_cores())
        if not cpus:
            sys.stderr.write(f"glm-roce-proxy-pin: proxy threads {sorted(tids)} recorded, not pinned "
                             f"(value {_spec()!r})\n")
            _STATE["pinned"] = ()
            return
        apply_current()

    cls.start = start


def register() -> None:
    import importlib.abc
    import importlib.util

    if os.environ.get(ENV, "").strip().lower() in ("", "0", "off", "no", "false"):
        return
    target = target_module()
    if target in sys.modules:
        _install(sys.modules[target])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != target:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _install(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
