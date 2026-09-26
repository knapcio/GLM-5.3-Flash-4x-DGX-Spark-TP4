# SPDX-License-Identifier: Apache-2.0
"""GLM_L2_PREFETCH=1: restricted L2 prefetch of the next dense weights inside a decode/verify step. Exact.

Tony v11 image (vLLM 0.1.dev20051+g487ecf187), `vllm.models.glm5next.nvidia.kda` (the repo's glm5next_kda.py).

Port of the ds41 `adapter/l2_prefetch.py` (ours, 2026-09-24/25; its kernel and the side-stream fork/join), cut
down to two fixed windows instead of the DS learner. The DS AHEAD / MOE / LMHEAD / ENGRAM extensions were flat or
slower on the DS fleet and are not ported.

Window A (GLM_L2_PREFETCH=1): the KDA attention core. `Glm5NextLinearAttention.forward` runs
    in_proj (25 MB) -> f_b / g_b -> _forward (short conv, recurrence: latency-bound, DRAM mostly idle at c1-c4)
    -> o_norm -> o_proj (Marlin W8A16 MXFP8, 2048 x 4096 per rank, ~8.4 MB, ~40 us at M <= 16)
  Right before `_forward` a side stream (waiting on the main stream, i.e. after g_b) issues
  `cp.async.bulk.prefetch.L2` for the first GLM_L2_PREFETCH_MB (default 6) MiB of o_proj's tensors (scales
  first, then the packed weight), so o_proj's first K tiles come from L2.
Window B (GLM_L2_PREFETCH_AR=1, needs A): the post-attention all-reduce (inside o_proj, RoCEnante one-shot:
  1-8 CTAs polling flags, DRAM idle). Right before that collective, a second side branch prefetches
  GLM_L2_PREFETCH_AR_MB (default 4) MiB of what is read next: the layer's hc_ffn_fn (fp32, 1.5 MB), the
  router weight, then the shared expert's (or dense MLP's) gate_up tensors.

Joins: every side branch is joined into the main stream at the end of the target model forward
(Glm5NextModel.forward), long after it finished, so no main-stream kernel ever waits on it. Weights are static
and a prefetch is only a cache hint: every output is bit-identical by construction.

Graph safety: the segment tables and side streams are built on the first eager decode-sized forward (vLLM runs
eager warm-up/dummy forwards before capture); a forward captured before that prefetches nothing. Decode/verify
shapes only: the windows are skipped when the layer sees more than GLM_L2_PREFETCH_MAXTOK tokens (default 32),
so prefill chunks and graph sizes above the cap are untouched. The decision is taken per capture (the token count
of a captured graph is fixed), so a graph never mixes prefetch and no-prefetch.

Knobs (read per call; switchable in-boot through overlay/glm_ab.py when it is armed):
  GLM_L2_PREFETCH=0|1, GLM_L2_PREFETCH_MB (6), GLM_L2_PREFETCH_MAXTOK (32), GLM_L2_PREFETCH_AR=0|1,
  GLM_L2_PREFETCH_AR_MB (4).
"""
from __future__ import annotations

import ctypes as C
import os
import subprocess
import sys
from pathlib import Path

TARGET_KDA = "vllm.models.glm5next.nvidia.kda"
TARGET_MODEL = "vllm.models.glm5next.nvidia.model"
_OFF = ("", "0", "off", "false", "no")

# Kernel: ds41 adapter/l2_prefetch.py (ours), unchanged apart from the symbol name.
_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>
// segs: [n][2] int64 {address, bytes}; one bulk prefetch per <=chunk piece.
__global__ void l2pf(const long long* __restrict__ segs, int n, long long chunk) {
  for (int s = blockIdx.x; s < n; s += gridDim.x) {
    long long a = segs[2 * s], b = segs[2 * s + 1];
    for (long long off = threadIdx.x * chunk; off < b; off += (long long)blockDim.x * chunk) {
      long long sz = b - off < chunk ? b - off : chunk;
      sz &= ~15LL;
      if (sz > 0)
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(a + off), "r"((unsigned)sz) : "memory");
    }
  }
}
extern "C" int glm_l2pf(const void* segs, int n, long long chunk, uintptr_t stream) {
  l2pf<<<n < 8 ? n : 8, 64, 0, (cudaStream_t)stream>>>((const long long*)segs, n, chunk);
  return (int)cudaGetLastError();
}
"""
_CHUNK = 16384
_DEBUG_EVERY = int(os.environ.get("GLM_L2_PREFETCH_LOG_EVERY", "20000") or 0)

_state = {
    "lib": None, "side": {}, "pending": set(), "depth": 0, "armed": None,
    "forks": 0, "tables": 0, "logged": set(), "keepalive": [], "installed": set(),
}


def _log(msg: str) -> None:
    sys.stderr.write(f"glm-l2-prefetch: {msg}\n")


def env(name: str, default=None):
    """os.environ, or the in-boot A/B variant value while overlay/glm_ab.py is armed."""
    ab = sys.modules.get("glm_ab")
    if ab is not None and getattr(ab, "ACTIVE", False):
        return ab.env(name, default)
    return os.environ.get(name, default)


def _on(name: str) -> bool:
    return str(env(name, "0")).strip().lower() not in _OFF


def installed() -> bool:
    return os.environ.get("GLM_L2_PREFETCH", "0").strip().lower() not in _OFF


def _mib(name: str, default: str) -> int:
    return int(float(env(name, default) or default) * (1 << 20))


def _lib():
    if _state["lib"] is None:
        cache = Path(os.environ.get("GLM_L2_PREFETCH_CACHE", "/cache/glm_l2pf"))
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            cache = Path("/tmp/glm_l2pf")
            cache.mkdir(parents=True, exist_ok=True)
        so = cache / "libglm_l2pf_v1.so"
        if not so.exists():
            cu = cache / f"l2pf.{os.getpid()}.cu"
            cu.write_text(_SRC)
            tmp = cache / f"libglm_l2pf_v1.{os.getpid()}.so"
            subprocess.run(["nvcc", "-O3", "-arch=sm_121a", "-shared", "-Xcompiler", "-fPIC", "-o", str(tmp),
                            str(cu)], check=True)
            os.replace(tmp, so)
        lib = C.CDLL(str(so))
        lib.glm_l2pf.argtypes = [C.c_void_p, C.c_int, C.c_longlong, C.c_void_p]
        lib.glm_l2pf.restype = C.c_int
        _state["lib"] = lib
    return _state["lib"]


# -- segment tables ------------------------------------------------------------------------------------

def take(tensors, budget: int):
    """[(ptr, nbytes)] in read order -> the first `budget` bytes, each piece rounded down to 16 B (pure)."""
    out, left = [], budget
    for ptr, nbytes in tensors:
        if left < 16:
            break
        if ptr % 16:
            continue  # cp.async.bulk.prefetch needs 16-byte aligned addresses; a skipped tensor is only a lost hint
        n = min(nbytes, left) & ~15
        if n > 0:
            out.append((ptr, n))
        left -= n
    return out


def module_tensors(mod, min_bytes: int = 4096):
    """The CUDA parameters of one linear-like module in the order a weight-only GEMM consumes them: the small
    ones (scales) first, then the packed weight. Skips non-CUDA and tiny tensors."""
    items = []
    for _name, p in mod.named_parameters(recurse=False):
        if p is None or not p.is_cuda:
            continue
        nbytes = p.numel() * p.element_size()
        if nbytes >= min_bytes and p.is_contiguous():
            items.append((p.data_ptr(), nbytes))
    items.sort(key=lambda it: it[1])
    return items


def _table(segs):
    import torch
    t = torch.tensor([v for s in segs for v in s], dtype=torch.int64, device="cuda")
    _state["keepalive"].append(t)  # captured graphs keep reading this address
    _state["tables"] += 1
    return (t, len(segs), sum(s[1] for s in segs))


def _side():
    import torch
    dev = torch.cuda.current_device()
    s = _state["side"].get(dev)
    if s is None:
        s = torch.cuda.Stream(device=dev)
        _state["side"][dev] = s
    return s


def _fork(item) -> None:
    import torch
    t, n, _nbytes = item
    main = torch.cuda.current_stream()
    side = _side()
    side.wait_stream(main)
    rc = _lib().glm_l2pf(t.data_ptr(), n, _CHUNK, side.cuda_stream)
    if rc:
        raise RuntimeError(f"glm-l2-prefetch: launch failed, cuda error {rc}")
    _state["pending"].add(side)
    _state["forks"] += 1


def join_all() -> None:
    import torch
    if _state["pending"]:
        main = torch.cuda.current_stream()
        for s in _state["pending"]:
            main.wait_stream(s)
        _state["pending"].clear()


# -- plans ---------------------------------------------------------------------------------------------

def _plan_a(attn):
    """Window A table for one KDA layer (built once, eager)."""
    p = attn.__dict__.get("_glm_l2_a")
    if p is None:
        budget = _mib("GLM_L2_PREFETCH_MB", "6")
        segs = take(module_tensors(attn.o_proj), budget)
        p = _table(segs) if segs else False
        attn.__dict__["_glm_l2_a"] = p
        attn.__dict__["_glm_l2_a_budget"] = budget
        if "a" not in _state["logged"]:
            _state["logged"].add("a")
            names = [(n, tuple(x.shape), str(x.dtype)) for n, x in attn.o_proj.named_parameters(recurse=False)]
            _log(f"window A (KDA core -> o_proj) {p[2] / 2**20 if p else 0:.2f} MiB of {names}; budget "
                 f"{budget / 2**20:.2f} MiB, MAXTOK {env('GLM_L2_PREFETCH_MAXTOK', '32')}")
    return p


def _plan_b(attn):
    p = attn.__dict__.get("_glm_l2_b")
    if p is None:
        layer = attn.__dict__.get("_glm_l2_layer")
        segs = []
        if layer is not None:
            budget = _mib("GLM_L2_PREFETCH_AR_MB", "4")
            queue = []
            fn = getattr(layer, "hc_ffn_fn", None)
            if fn is not None and fn.is_cuda and fn.is_contiguous():
                queue.append((fn.data_ptr(), fn.numel() * fn.element_size()))
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)
            if gate is not None:
                queue += module_tensors(gate)
            gu = getattr(getattr(mlp, "shared_experts", None), "gate_up_proj", None)
            if gu is None:
                gu = getattr(mlp, "gate_up_proj", None)
            if gu is not None:
                queue += module_tensors(gu)
            segs = take(queue, budget)
        p = _table(segs) if segs else False
        attn.__dict__["_glm_l2_b"] = p
        if "b" not in _state["logged"]:
            _state["logged"].add("b")
            _log(f"window B (post-attention all-reduce -> hc_ffn/router/shared gate_up) "
                 f"{p[2] / 2**20 if p else 0:.2f} MiB")
    return p


def _link_layers(model) -> None:
    """Give every KDA attention module a reference to its decoder layer (for window B)."""
    for layer in getattr(model, "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "o_proj"):
            attn.__dict__["_glm_l2_layer"] = layer


# -- hooks ---------------------------------------------------------------------------------------------

def install_kda(mod) -> None:
    import torch
    cls = mod.Glm5NextLinearAttention
    if getattr(cls, "_glm_l2_prefetch", False):
        return
    cls._glm_l2_prefetch = True
    orig_forward = cls.forward
    inner = cls._forward

    def _forward(self, *args, **kwargs):
        n = self.__dict__.get("_glm_l2_ntok", 0)
        _state["calls"] = _state.get("calls", 0) + 1
        if _state["calls"] <= 3 or (_DEBUG_EVERY and _state["calls"] % _DEBUG_EVERY == 0):
            _log(f"_forward call {_state['calls']}: ntok={n} depth={_state['depth']} "
                 f"on={_on('GLM_L2_PREFETCH')} capturing={torch.cuda.is_current_stream_capturing()} "
                 f"forks={_state['forks']} tables={_state['tables']}")
        if 0 < n <= int(env("GLM_L2_PREFETCH_MAXTOK", "32")) and _state["depth"] > 0 and _on("GLM_L2_PREFETCH"):
            capturing = torch.cuda.is_current_stream_capturing()
            p = self.__dict__.get("_glm_l2_a")
            if p is None and not capturing:
                p = _plan_a(self)
            if p:
                _fork(p)
            if _on("GLM_L2_PREFETCH_AR"):
                pb = self.__dict__.get("_glm_l2_b")
                if pb is None and not capturing:
                    pb = _plan_b(self)
                _state["armed"] = pb if pb else None
        return inner(self, *args, **kwargs)

    _forward.__wrapped__ = getattr(inner, "__wrapped__", inner)

    def forward(self, hidden_states, positions, *args, **kwargs):
        self.__dict__["_glm_l2_ntok"] = int(hidden_states.size(0))
        try:
            return orig_forward(self, hidden_states, positions, *args, **kwargs)
        finally:
            _state["armed"] = None  # window B is consumed by o_proj's all-reduce or dropped here

    cls._forward = _forward
    cls.forward = forward
    _log("KDA core forks the o_proj prefetch (window A)")


def install_model(mod) -> None:
    cls = mod.Glm5NextModel
    if getattr(cls, "_glm_l2_prefetch", False):
        return
    cls._glm_l2_prefetch = True
    orig = cls.forward

    def forward(self, *args, **kwargs):
        if "linked" not in self.__dict__.get("_glm_l2_flags", ()):
            _link_layers(self)
            self.__dict__["_glm_l2_flags"] = ("linked",)
        _state["depth"] += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _state["depth"] -= 1
            _state["armed"] = None
            if _state["depth"] == 0:
                join_all()

    cls.forward = forward
    _log("target model forward joins the prefetch branches at its end")


def install_roce() -> None:
    """Window B: fork the armed table right before the RoCEnante all-reduce (glm_roce, repos/glm/glm53-rocenante;
    RoCEnante is local-inference-lab/b12x)."""
    try:
        from glm_roce import adapter as ra
    except Exception as exc:  # noqa: BLE001
        _log(f"glm_roce not importable ({exc!r}); window B off")
        return
    cls = ra.GlmRoceAllReduce
    if getattr(cls, "_glm_l2_prefetch", False):
        return
    cls._glm_l2_prefetch = True
    orig = cls.custom_all_reduce

    def custom_all_reduce(self, inp):
        armed = _state["armed"]
        if armed is not None:
            _state["armed"] = None
            if self.should_custom_ar(inp):
                _fork(armed)
        return orig(self, inp)

    cls.custom_all_reduce = custom_all_reduce
    _log("RoCE all-reduce forks the armed window B")


HOOKS = {TARGET_KDA: install_kda, TARGET_MODEL: install_model}


def register() -> None:
    import importlib.abc
    import importlib.util

    if not installed():
        return

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
            fn = HOOKS[name]

            def exec_module(module, _orig=orig_exec, _fn=fn):
                _orig(module)
                _fn(module)
                if _fn is install_model and os.environ.get("GLM_L2_PREFETCH_AR", "0").strip().lower() not in _OFF:
                    install_roce()
            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
