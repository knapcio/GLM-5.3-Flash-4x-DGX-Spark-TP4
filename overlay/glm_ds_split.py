# SPDX-License-Identifier: Apache-2.0
"""Replicated BF16 linears split by output columns over TP (port of our DS4.1 DSV41_REPLICATED_SPLIT).

A replicated linear makes every TP rank stream the whole weight to compute the same output. In the
GLM-5.3-Flash stack three BF16 replicated weights are read on every decode step (per rank):

  part    module                                        shape (N x K)         per step         path
  qkv_a   MLA self_attn.fused_qkv_a_proj (11 layers)    2048 x 4096 (q 1536 + kv 512)  185 MB   target graph
  wq_b    DSA self_attn.indexer.wq_b (11 layers)        4096 x 1536           138 MB   target graph
  fc      DFlash/DFlash2 drafter model.fc               4096 x 20480 (5 taps) 168 MB   eager, before the draft graph

(the mega conversion keeps q_a/kv_a and the whole indexer BF16: scripts/quantize_nonexpert_mxfp8.py IGNORE).
Together ~490 MB, ~2.1 ms per step at ~235 GB/s. Here each rank computes only its N/tp columns from a
contiguous column slice of the weight and the columns are all-gathered (vLLM's
tensor_model_parallel_all_gather(dim=-1); the RoCE shim routes dim=-1 gathers <= its gather limit
through the one-shot kernel when captured). Saves 3/4 of those bytes per rank minus one gather per call.

GLM_DS_SPLIT="part:max_rows,..."  e.g. "fc:128" (NCCL) or "fc:128,qkv_a:64,wq_b:16" (with RoCE).
    A part is split for calls with 1 <= rows <= max_rows; other row counts (prefill chunks) use the stock
    full-weight GEMM. Row counts are identical on every rank, so the branch is rank-invariant.
GLM_DS_SPLIT_EXACT=1 (default): at load, every rank compares slice GEMM vs the stock GEMM's columns bit for
    bit at every row count 1..max_rows on random inputs, and the per-(layer, rows) verdicts are combined
    with a MIN all-reduce over the TP CPU group: a (layer, rows) is split on all ranks or on none.
    0 = split every eligible layer at every row count <= max_rows without the check. Still rank-consistent
    (each column block is computed by exactly one rank and gathered), but the fp32 reduction order can
    differ from the stock GEMM: quality-gate it like RoCEnante.
GLM_DS_SPLIT_AUDIT=N: the first N eager (non-captured) calls per layer also run the stock GEMM and count
    differing elements (logged). Costs a full GEMM per audited call; for soak tests only.

Credit: the technique is our DS4.1 replicated_split adapter (ds41 adapter/replicated_split.py).
"""
from __future__ import annotations

import os
import sys
import types
import zlib

PART_DEFAULT_MAX = {"fc": 128, "qkv_a": 64, "wq_b": 16}


def parse_spec(spec: str) -> dict:
    """"fc:128,qkv_a" -> {"fc": 128, "qkv_a": 64}. Unknown parts raise (a typo must not silently no-op)."""
    out = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item or item in ("0", "off"):
            continue
        if item in ("1", "all"):
            out.update(PART_DEFAULT_MAX)
            continue
        name, _, mx = item.partition(":")
        name = name.strip()
        if name not in PART_DEFAULT_MAX:
            raise ValueError(f"GLM_DS_SPLIT: unknown part {name!r} (known: {sorted(PART_DEFAULT_MAX)})")
        out[name] = int(mx) if mx.strip() else PART_DEFAULT_MAX[name]
        if not 1 <= out[name] <= 4096:
            raise ValueError(f"GLM_DS_SPLIT: {name} max rows {out[name]} out of range")
    return out


PARTS = parse_spec(os.environ.get("GLM_DS_SPLIT", ""))
EXACT = os.environ.get("GLM_DS_SPLIT_EXACT", "1").strip() != "0"
AUDIT = int(os.environ.get("GLM_DS_SPLIT_AUDIT", "0") or 0)
_LOG_ALL_RANKS = os.environ.get("GLM_DS_SPLIT_LOG_ALL", "0") == "1"


def classify(name: str, module) -> str | None:
    """Which part (if any) a module is. Name-based, then type-checked."""
    cls = type(module).__name__
    if name.endswith(".fused_qkv_a_proj") and cls == "DeepSeekV2FusedQkvAProjLinear":
        return "qkv_a"
    if name.endswith(".indexer.wq_b") and cls == "ReplicatedLinear":
        return "wq_b"
    if (name == "model.fc" or name.endswith(".model.fc")) and cls == "ReplicatedLinear":
        return "fc"
    return None


def eligible(module) -> str | None:
    """None when the layer can be split; otherwise the reason it cannot."""
    import torch
    qm = getattr(module, "quant_method", None)
    if qm is None or type(qm).__name__ != "UnquantizedLinearMethod":
        return f"quantized ({type(qm).__name__})"
    w = getattr(module, "weight", None)
    if w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
        return "weight not 2-D bf16"
    if getattr(module, "bias", None) is not None:
        return "has bias"
    if getattr(module, "_use_min_latency_gemm", False):
        return "min-latency GEMM path"
    if getattr(module, "gather_output", False) or getattr(module, "tp_size", 1) != 1:
        return "not replicated"
    return None


def col_range(n: int, rank: int, world: int) -> tuple[int, int]:
    if n % world:
        raise ValueError(f"N={n} not divisible by TP {world}")
    loc = n // world
    return rank * loc, (rank + 1) * loc


class _Proxy:
    """What UnquantizedLinearMethod.apply reads from the layer: .weight (and nothing else on CUDA)."""
    __slots__ = ("weight",)

    def __init__(self, weight):
        self.weight = weight


def _flat(x):
    return x if x.dim() == 2 else x.reshape(-1, x.shape[-1])


def make_forward(module, part: str, w_slice, ok_rows: frozenset, gather, is_capturing, stats: dict):
    """Per-instance forward: slice GEMM + column all-gather for row counts in ok_rows, stock otherwise."""
    orig_forward = module.forward
    proxy = _Proxy(w_slice)
    qm = module.quant_method
    return_bias = getattr(module, "return_bias", True)

    def forward(self, x):
        rows = x.numel() // x.shape[-1] if x.dim() >= 1 and x.shape[-1] else 0
        if rows not in ok_rows:
            stats["stock"] = stats.get("stock", 0) + 1
            return orig_forward(x)
        xs = _flat(x)
        y = gather(qm.apply(proxy, xs, None), -1)
        if AUDIT and stats.get("audited", 0) < AUDIT and not is_capturing():
            import torch
            ref = qm.apply(self, xs, None)
            stats["audited"] = stats.get("audited", 0) + 1
            stats["audit_diff"] = stats.get("audit_diff", 0) + int((ref != y).sum().item())
        if x.dim() != 2:
            y = y.reshape(*x.shape[:-1], y.shape[-1])
        stats["split"] = stats.get("split", 0) + 1
        return (y, None) if return_bias else y

    return types.MethodType(forward, module)


def check_rows(module, w_slice, c0: int, c1: int, max_rows: int, device, seeds=(0, 1)) -> list[bool]:
    """ok[r-1] = slice GEMM equals stock GEMM columns [c0, c1) bit for bit at r rows, for every seed."""
    import torch
    qm = module.quant_method
    proxy = _Proxy(w_slice)
    k = module.weight.shape[1]
    ok = []
    gen = torch.Generator(device="cpu")
    for r in range(1, max_rows + 1):
        good = True
        for s in seeds:
            gen.manual_seed(1000 * r + s)
            # activation-like: mostly O(1) with a few large channels
            x = torch.randn(r, k, generator=gen) * (1.0 + 9.0 * (torch.rand(1, k, generator=gen) < 0.01))
            x = x.to(device=device, dtype=torch.bfloat16)
            full = qm.apply(module, x, None)[:, c0:c1]
            part = qm.apply(proxy, x, None)
            if not torch.equal(full, part):
                good = False
                break
        ok.append(good)
    return ok


def agree_min(flags, group):
    """Elementwise MIN of an int tensor over the TP CPU group (gloo). group None = single process."""
    if group is None:
        return flags
    import torch.distributed as dist
    dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
    return flags


def split_model(model, parts: dict, *, rank: int, world: int, group, gather, is_capturing,
                exact: bool = True, log=print) -> dict:
    """Install the split on every eligible layer of `model`. Collective over `group` (all ranks must call it
    with the same model structure). Returns a report {part: {"layers": n, "split": n, "rows_ok": [...]}}."""
    import torch
    cands = []
    seen = set()
    for name, mod in model.named_modules():
        part = classify(name, mod)
        if part is None or part not in parts or id(mod) in seen:
            continue
        seen.add(id(mod))
        cands.append((name, part, mod))
    report = {p: {"layers": 0, "split": 0, "skipped": {}} for p in parts}
    if not cands:
        return report
    # the candidate list must be identical on every rank; fail loudly otherwise (a rank that skips a
    # gather its peers make would hang the fleet)
    sig = torch.tensor([len(cands), zlib.crc32("|".join(n for n, _, _ in cands).encode())], dtype=torch.int64)
    sig_min = agree_min(sig.clone(), group)
    sig_max = -agree_min(-sig.clone(), group)
    if not torch.equal(sig_min, sig_max):
        raise RuntimeError("glm-ds-split: TP ranks see different candidate layers; refusing to split")
    max_rows = max(parts.values())
    flags = torch.zeros(len(cands), max_rows, dtype=torch.int32)
    slices = []
    for i, (name, part, mod) in enumerate(cands):
        why = eligible(mod)
        n = mod.weight.shape[0]
        if why is None and n % (world * 8):
            why = f"N={n} not a multiple of 8*TP"
        if why is not None:
            report[part]["skipped"][name] = why
            slices.append(None)
            continue
        c0, c1 = col_range(n, rank, world)
        w_slice = mod.weight.data[c0:c1].contiguous()
        slices.append((w_slice, c0, c1))
        lim = parts[part]
        if exact:
            ok = check_rows(mod, w_slice, c0, c1, lim, mod.weight.device)
        else:
            ok = [True] * lim
        flags[i, :lim] = torch.tensor(ok, dtype=torch.int32)
    # a skipped (ineligible) layer has all-zero flags on every rank that skipped it; eligibility is a
    # property of the checkpoint, so it is the same on every rank, and the MIN makes it so regardless
    flags = agree_min(flags, group)
    extra_mb = 0.0
    for i, (name, part, mod) in enumerate(cands):
        report[part]["layers"] += 1
        if slices[i] is None:
            continue
        rows = frozenset(r + 1 for r in range(parts[part]) if int(flags[i, r]) == 1)
        if not rows:
            report[part]["skipped"][name] = "no row count passed the bit-exact check on all ranks"
            continue
        w_slice = slices[i][0]
        stats = {}
        mod._glm_ds_split = {"part": part, "rows": rows, "stats": stats, "cols": slices[i][1:]}
        mod._glm_ds_w_slice = w_slice  # keep alive
        mod.forward = make_forward(mod, part, w_slice, rows, gather, is_capturing, stats)
        report[part]["split"] += 1
        report[part].setdefault("rows_min_max", [])
        report[part]["rows_min_max"].append((min(rows), max(rows), len(rows)))
        extra_mb += w_slice.numel() * w_slice.element_size() / 2**20
    if rank == 0 or _LOG_ALL_RANKS:
        for p, r in report.items():
            rr = r.get("rows_min_max", [])
            cover = f"rows {min(a for a, _, _ in rr)}..{max(b for _, b, _ in rr)}, " \
                    f"{min(c for _, _, c in rr)}-{max(c for _, _, c in rr)} row counts" if rr else "none"
            log(f"glm-ds-split: {p}: {r['split']}/{r['layers']} layers split ({cover}; "
                f"{'bit-exact checked' if exact else 'UNCHECKED'}); skipped {len(r['skipped'])}"
                + (f" e.g. {next(iter(r['skipped'].items()))}" if r["skipped"] else ""))
        log(f"glm-ds-split: extra slice memory {extra_mb:.1f} MiB per rank")
    return report


def _vllm_env():
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.distributed.parallel_state import get_tp_group
    import torch
    tp = get_tp_group()
    return dict(rank=tp.rank_in_group, world=tp.world_size, group=tp.cpu_group,
                gather=lambda y, dim: tensor_model_parallel_all_gather(y, dim),
                is_capturing=torch.cuda.is_current_stream_capturing)


def install_loader(base_loader_module) -> None:
    """Wrap BaseModelLoader.load_model: after the stock load (and process_weights_after_loading) split the
    model's eligible layers. Runs for the target and for the drafter (loaded through get_model too)."""
    cls = base_loader_module.BaseModelLoader
    if getattr(cls, "_glm_ds_split", False):
        return
    if not PARTS:
        return
    orig = cls.load_model

    def load_model(self, vllm_config, model_config, prefix: str = ""):
        model = orig(self, vllm_config, model_config, prefix)
        env = _vllm_env()
        if env["world"] > 1:
            split_model(model, PARTS, exact=EXACT, log=lambda s: print(s, file=sys.stderr, flush=True), **env)
        return model

    load_model.__wrapped__ = orig
    cls.load_model = load_model
    cls._glm_ds_split = True
    print(f"glm-ds-split: armed parts={PARTS} exact={EXACT} audit={AUDIT}", file=sys.stderr, flush=True)
