#!/usr/bin/env python3
"""Non-expert quantization arms for nvidia/GLM-5.3-Flash-NVFP4 (ModelOpt) on 4x DGX Spark, TP4 vLLM.

nvidia's pack quantizes the routed experts (and the three dense MLPs) to NVFP4 and leaves attention, the shared
experts and lm_head in BF16. Those are read on every decode step. This tool writes ModelOpt MIXED_PRECISION arms
that re-encode them per *group* (a fused vLLM module) in one of: bf16 | mxfp8 | fp8 | nvfp4.

Layout per node (no 190 GB copy per arm):
  <root>/_overlay/<arm>/model-qmix-<arm>.safetensors    the re-encoded tensors (+ BF16-kept ones)   (quant, ~5-8 GB)
  <root>/<arm>/     zero-copy (default): hardlinks to every base shard + the overlay + a sentinel shard, an index that
                    points re-encoded names at the overlay, a MIXED_PRECISION config (assemble, seconds). The base
                    shards still hold BF16 copies; overlay/qmix_patch.py drops them at load (QMIX_FP8_BLOCK=1) and
                    the sentinel makes an unpatched load fail loudly.
  <root>/_common/   optional (strip, ~175 GB/node): base shards with those tensors and the MTP layer removed, for
                    assemble --common, which needs no load filter.

Formats on disk (what vLLM's ModelOpt linear methods in the pinned image load):
  mxfp8  weight F8_E4M3 [N,K], weight_scale U8 [N,K/32] (E8M0, value 2**(s-127))  -> Marlin W8A16
  fp8    weight F8_E4M3 [N,K], weight_scale F32 [1], input_scale F32 [1] (=1, unused under
         VLLM_TEST_FORCE_FP8_MARLIN=1, which makes it Marlin W8A16; without that flag it is W8A8 static and wrong)
  nvfp4  weight U8 [N,K/2] (low nibble = even column), weight_scale F8_E4M3 [N,K/16] (not swizzled),
         weight_scale_2 F32 [1] = amax_group/(448*6); one amax per fused group (vLLM refuses mixed global
         scales in a fused W4A16 layer)                                              -> Marlin W4A16
Subcommands: inventory | proxy | plan | quant | strip | assemble | verify | estimate | selftest
All CPU. Quantization processes fused groups but retains output bytes until writing; peak RAM is not bounded here.
"""
import argparse, collections, json, math, os, re, struct, sys, time

FP8_MAX, E2M1_MAX = 448.0, 6.0
MTP_LAYER = 45
DT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1, "I64": 8, "I32": 4, "BOOL": 1}
P = "model.language_model.layers."

# group kind -> (checkpoint member suffixes, vLLM module suffix used in quantized_layers)
KDA_IN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.b_proj",
          "self_attn.f_a_proj", "self_attn.g_a_proj"]
KINDS = {
    "kda_in":  (KDA_IN, "self_attn.in_proj_qkvbfg_a"),
    "kda_o":   (["self_attn.o_proj"], "self_attn.o_proj"),
    "mla_qa":  (["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"], "self_attn.fused_qkv_a_proj"),
    "mla_qb":  (["self_attn.q_b_proj"], "self_attn.q_b_proj"),
    "mla_kvb": (["self_attn.kv_b_proj"], "self_attn.kv_b_proj"),
    "mla_o":   (["self_attn.o_proj"], "self_attn.o_proj"),
    "sh_gu":   (["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"], "mlp.shared_experts.gate_up_proj"),
    "sh_down": (["mlp.shared_experts.down_proj"], "mlp.shared_experts.down_proj"),
}
# vLLM-side input norm for the activation-aware proxy (None = no per-channel weight available)
GAMMA = {"kda_in": "input_layernorm.weight", "mla_qa": "input_layernorm.weight",
         "mla_qb": "self_attn.q_a_layernorm.weight", "kda_o": "self_attn.o_norm.weight",
         "sh_gu": "post_attention_layernorm.weight"}
IGNORE_BASE = ["model.language_model.embed_tokens", "*.self_attn.f_b_proj", "*.self_attn.g_b_proj",
               "*.self_attn.indexer.wk_weights_proj", "*.self_attn.indexer.wk",
               "*.self_attn.indexer.weights_proj", "*.self_attn.indexer.wq_b",
               "*.mlp.gate", "*.eh_proj", "model.visual.*"]


# ----------------------------------------------------------------------------- safetensors I/O
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    meta = h.pop("__metadata__", None)
    return h, 8 + n, meta


def headers(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            out[f] = read_header(os.path.join(d, f))
    return out


def tensor(path, base, info):
    import torch
    a, b = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = bytearray(f.read(b - a))
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn,
          "F16": torch.float16}[info["dtype"]]
    return torch.frombuffer(raw, dtype=dt).view(*info["shape"]) if info["shape"] else torch.frombuffer(raw, dtype=dt)[0]


def drop_cache(fd, sync=False):
    try:
        if sync:
            os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def write_st(dst, items, meta=None):
    """items: list of (name, dtype, shape, source) where source is bytes or (path, abs_offset, nbytes)."""
    hdr, off = {"__metadata__": meta or {"format": "pt"}}, 0
    for name, dt, shape, src in items:
        n = len(src) if isinstance(src, (bytes, bytearray)) else src[2]
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = dst + ".part"
    with open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(hb)) + hb)
        fo.flush()
        written = 0
        for name, dt, shape, src in items:
            if isinstance(src, (bytes, bytearray)):
                fo.write(src)
                continue
            fo.flush()
            path, pos, left = src
            with open(path, "rb") as fi:
                while left:
                    try:
                        k = os.copy_file_range(fi.fileno(), fo.fileno(), min(left, 256 << 20), pos)
                    except (AttributeError, OSError):
                        fi.seek(pos)
                        k = fo.write(fi.read(min(left, 64 << 20)))
                    if k <= 0:
                        raise IOError(f"short copy from {path}")
                    pos += k
                    left -= k
                    written += k
                    if written > (2 << 30):
                        drop_cache(fo.fileno(), sync=True)
                        drop_cache(fi.fileno())
                        written = 0
                drop_cache(fi.fileno())
        fo.flush()
        drop_cache(fo.fileno(), sync=True)
    os.replace(tmp, dst)


# ----------------------------------------------------------------------------- inventory / groups
def load_cfg(src):
    c = json.load(open(os.path.join(src, "config.json")))
    return c.get("text_config", c)


def groups_of(src_headers, cfg):
    """Return {(layer, kind): [member tensor names]} for layers 0..44, plus ('head','lm_head')."""
    lt = cfg["layer_types"]
    names = {k for _, (h, _, _) in src_headers.items() for k in h}
    g = {}
    for L in range(cfg["num_hidden_layers"]):
        att = "kda" if lt[L] == "linear_attention" else "mla"
        for kind, (members, _) in KINDS.items():
            if not kind.startswith(att) and not kind.startswith("sh_"):
                continue
            full = [f"{P}{L}.{m}.weight" for m in members]
            if all(n in names for n in full):
                g[(L, kind)] = full
    if "lm_head.weight" in names:
        g[("head", "lm_head")] = ["lm_head.weight"]
    return g


def locate(src_headers):
    loc = {}
    for f, (h, base, _) in src_headers.items():
        for k, v in h.items():
            loc[k] = (f, base, v)
    return loc


def per_rank_bytes(kind, shape, fmt, tp=4):
    """Bytes one rank reads per decode step for one member tensor."""
    n, k = shape
    if kind in ("mla_qa", "lm_head_rep"):
        share = 1.0                      # ReplicatedLinear / disable_tp: every rank reads it all
    elif kind == "kda_in" and n == 128:  # f_a / g_a: replicated shards of the fused KDA projection
        share = 1.0
    else:
        share = 1.0 / tp
    el = n * k * share
    if fmt == "bf16":
        return el * 2
    if fmt == "mxfp8":
        return el * (1 + 1 / 32)
    if fmt == "fp8":
        return el
    if fmt == "fp8blk":
        return el * (1 + 4 / 16384)
    if fmt == "nvfp4":
        return el * (0.5 + 1 / 16)
    raise ValueError(fmt)


def decode_read(kind):
    """kv_b_proj is absorbed into BF16 W_UK/W_UV copies at load (mla_attention.py process_weights_after_loading):
    its format does not change decode bytes, only adds error."""
    return kind != "mla_kvb"


# ----------------------------------------------------------------------------- quantizers
def q_mxfp8(w, rows=2048):
    """Per 32-block E8M0 exponent: the better (lower squared error) of ceil(log2(amax/448)) (no clipping) and the
    OCP MX rule floor(log2(amax)) - 8 (may saturate the block max at 448). The OCP rule reproduces weights that
    already sit on an MX grid exactly."""
    import torch
    n, k = w.shape
    qs, ss = [], []
    for i in range(0, n, rows):
        b = w[i:i + rows].float().view(-1, k // 32, 32)
        amax = b.abs().amax(-1).clamp(min=2.0 ** -126)
        best = None
        for e in (torch.ceil(torch.log2(amax / FP8_MAX)), torch.floor(torch.log2(amax)) - 8):
            e = e.clamp(-127, 127)
            q = (b / torch.exp2(e).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            err = ((q.float() * torch.exp2(e).unsqueeze(-1) - b) ** 2).sum(-1)
            if best is None:
                best = [err, q, e]
            else:
                t = err < best[0]
                best = [torch.where(t, err, best[0]), torch.where(t.unsqueeze(-1), q.float(), best[1].float()).to(torch.float8_e4m3fn),
                        torch.where(t, e, best[2])]
        qs.append(best[1].view(-1, k)); ss.append((best[2] + 127).to(torch.uint8))
    return torch.cat(qs).contiguous(), torch.cat(ss).contiguous()


def dq_mxfp8(q, s):
    import torch
    n, k = q.shape
    return (q.float().view(n, k // 32, 32) * torch.exp2(s.float() - 127).unsqueeze(-1)).view(n, k)


def q_fp8_tensor(w, amax=None):
    import torch
    amax = float(w.float().abs().max()) if amax is None else amax
    s = max(amax, 1e-12) / FP8_MAX
    return (w.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), torch.tensor([s], dtype=torch.float32)


def dq_fp8_tensor(q, s):
    return q.float() * float(s.reshape(-1)[0])


def q_fp8_blk(w, block=128):
    """DeepSeek/vLLM Fp8LinearMethod block format: weight F8_E4M3 [N,K], weight_scale_inv F32 [ceil(N/128),
    ceil(K/128)], dequant = q * scale_inv. Scale = block amax / 448, which re-finds the grid of weights that were
    released as block-128 FP8."""
    import torch
    n, k = w.shape
    ob, ib = -(-n // block), -(-k // block)
    pad = torch.zeros(ob * block, ib * block)
    pad[:n, :k] = w.float()
    t = pad.view(ob, block, ib, block)
    sc = t.abs().amax(dim=(1, 3)).clamp(min=1e-12) / FP8_MAX
    q = (t / sc[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(ob * block, ib * block)[:n, :k].contiguous(), sc.float().contiguous()


def dq_fp8_blk(q, sc, block=128):
    n, k = q.shape
    return q.float() * sc.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]


def q_fp8_block(w, block=128):
    import torch
    n, k = w.shape
    ob, ib = -(-n // block), -(-k // block)
    pad = torch.zeros(ob * block, ib * block)
    pad[:n, :k] = w.float()
    t = pad.view(ob, block, ib, block)
    sc = t.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12) / FP8_MAX
    q = (t / sc).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * sc
    return q.view(ob * block, ib * block)[:n, :k]


def _e2m1():
    import torch
    return torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _mids():
    import torch
    return torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])   # E2M1 rounding thresholds (ties -> lower)


def _q_nvfp4_rows(w, gs, search):
    import torch
    E, M = _e2m1(), _mids()
    n, k = w.shape
    blk = w.float().view(n, k // 16, 16)
    bmax = blk.abs().amax(-1).clamp(min=1e-12)
    best_err = best_sf = None
    for c in ((1.0, 0.95, 0.9, 0.85) if search else (1.0,)):
        sf = (bmax * c / E2M1_MAX * gs).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float()
        eff = (sf / gs).clamp(min=1e-30).unsqueeze(-1)
        x = (blk / eff).clamp(-E2M1_MAX, E2M1_MAX)
        err = ((torch.sign(x) * E[torch.bucketize(x.abs(), M)] * eff - blk) ** 2).sum(-1)
        if best_err is None:
            best_err, best_sf = err, sf
        else:
            t = err < best_err
            best_err, best_sf = torch.where(t, err, best_err), torch.where(t, sf, best_sf)
    eff = (best_sf / gs).unsqueeze(-1)
    eff = torch.where(eff > 0, eff, torch.ones_like(eff))
    x = (blk / eff).clamp(-E2M1_MAX, E2M1_MAX)
    idx = torch.bucketize(x.abs(), M).to(torch.uint8)
    codes = (idx | torch.where(x < 0, 8, 0).to(torch.uint8)).view(n, k)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), best_sf.to(torch.float8_e4m3fn)


def q_nvfp4(w, amax, search=True, rows=1024):
    """w [N,K] BF16, amax = the group's global amax. Returns packed U8 [N,K/2], scale F8 [N,K/16], ws2 F32 [1]."""
    import torch
    gs = FP8_MAX * E2M1_MAX / max(amax, 1e-12)          # global scale; ws2 = 1/gs
    ps, ss = [], []
    for i in range(0, w.shape[0], rows):
        p, s = _q_nvfp4_rows(w[i:i + rows], gs, search)
        ps.append(p); ss.append(s)
    return torch.cat(ps).contiguous(), torch.cat(ss).contiguous(), torch.tensor([1.0 / gs], dtype=torch.float32)


def dq_nvfp4(packed, sf, ws2):
    import torch
    E = _e2m1()
    n = packed.shape[0]
    codes = torch.stack([packed & 15, packed >> 4], -1).view(n, -1)
    v = torch.where((codes & 8) > 0, -E[(codes & 7).long()], E[(codes & 7).long()])
    return (v.view(n, -1, 16) * (sf.float() * float(ws2.reshape(-1)[0])).unsqueeze(-1)).view(n, -1)


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


# ----------------------------------------------------------------------------- arms
GUARD_LAYERS = (0, 1, 2, 3, 43, 44)   # first MoE layer and the last two (layer 44 kda_o / sh_down kurtosis 339 / 163)


def plan_for(arm, groups, proxy=None, nv_thresh=None):
    """Return {'L:kind': fmt}. fmt in bf16 | mxfp8 | fp8 | fp8blk | nvfp4."""
    plan = {}
    for (L, kind) in groups:
        key = f"{L}:{kind}"
        if arm == "bf16":
            f = "bf16"
        elif arm == "fp8all":           # the mega arm's module set (kv_b included, qkv_a/head BF16), all MXFP8
            f = "mxfp8" if kind in ("kda_in", "kda_o", "mla_qb", "mla_kvb", "mla_o", "sh_gu", "sh_down") else "bf16"
        elif arm == "nvfp4attn":        # tonyd2wild's module set minus kv_b (no decode bytes: absorbed into BF16)
            f = "nvfp4" if kind in ("kda_in", "kda_o", "mla_qb", "mla_o", "sh_gu", "sh_down") else "bf16"
        elif arm in ("lossless8", "mixed", "mixed-mx"):
            # The released BF16 weights already sit on 8-bit grids: KDA on an MXFP8 grid (re-quantization error
            # 0.05 %), MLA projections and shared experts on a block-128 FP8 grid (0.16 %). Encode each on its
            # own grid; kv_b stays BF16 (absorbed into BF16 W_UK/W_UV at load, no decode bytes to win).
            eight = "mxfp8" if arm == "mixed-mx" else "fp8blk"
            f = {"kda_in": "mxfp8", "kda_o": "mxfp8", "mla_qa": eight, "mla_qb": eight, "mla_o": eight,
                 "sh_gu": eight, "sh_down": eight}.get(kind, "bf16")
            if arm in ("mixed", "mixed-mx") and kind in ("sh_gu", "sh_down") and L not in GUARD_LAYERS:
                f = "nvfp4"         # consult lever 4: shared experts to NVFP4, same FFN class as the routed experts
                if proxy:
                    r = proxy.get(key, {})
                    if (nv_thresh is not None and r.get("sel_nvfp4", 0) > nv_thresh) or r.get("kurt", 0) > 6:
                        f = eight
        else:
            raise SystemExit(f"unknown arm {arm}")
        plan[key] = f
    return plan


def quantized_layers(cfg, plan):
    ql = {}
    lt, mt = cfg["layer_types"], cfg.get("mlp_layer_types") or []
    for L in range(cfg["num_hidden_layers"]):
        p = f"{P}{L}"
        if L < len(mt) and mt[L] == "dense":
            for x in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
                ql[f"{p}.mlp.{x}"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        else:
            ql[f"{p}.mlp.experts"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    algo = {"mxfp8": {"quant_algo": "MXFP8"}, "fp8": {"quant_algo": "FP8"}, "fp8blk": {"quant_algo": "FP8_BLOCK"},
            "nvfp4": {"quant_algo": "W4A16_NVFP4", "group_size": 16}}
    for key, f in plan.items():
        if f == "bf16":
            continue
        L, kind = key.split(":")
        if kind == "lm_head":
            ql["lm_head"] = algo[f]
            continue
        members, fused = KINDS[kind]
        ql[f"{P}{L}.{fused}"] = algo[f]
        for m in members:
            ql[f"{P}{L}.{m}"] = algo[f]
    return dict(sorted(ql.items()))


# ----------------------------------------------------------------------------- commands
def cmd_inventory(a):
    H = headers(a.src)
    cfg = load_cfg(a.src)
    G = groups_of(H, cfg)
    loc = locate(H)
    tot = collections.defaultdict(float)
    for (L, kind), mem in G.items():
        for m in mem:
            sh = loc[m][2]["shape"]
            k = "lm_head_rep" if kind == "lm_head" else kind
            tot[kind] += per_rank_bytes(k if kind != "lm_head" else "x", sh, "bf16") if decode_read(kind) else 0
    s = sum(tot.values())
    for k, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"{k:8s} {v/2**20:8.1f} MiB/rank/step BF16")
    print(f"total    {s/2**20:8.1f} MiB/rank/step = {s/235e9*1e3:.2f} ms at 235 GB/s")


def cmd_proxy(a):
    import torch
    torch.set_num_threads(a.threads)
    H = headers(a.src)
    cfg = load_cfg(a.src)
    G = groups_of(H, cfg)
    loc = locate(H)
    out = json.load(open(a.out)) if (a.out and os.path.exists(a.out)) else {}
    t0 = time.time()
    for (L, kind), mem in sorted(G.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        key = f"{L}:{kind}"
        if key in out or (a.only and kind not in a.only.split(",")):
            continue
        ws = [tensor(os.path.join(a.src, loc[m][0]), loc[m][1], loc[m][2]) for m in mem]
        gamma = None
        if kind in GAMMA and L != "head":
            gn = f"{P}{L}.{GAMMA[kind]}"
            if gn in loc:
                gamma = tensor(os.path.join(a.src, loc[gn][0]), loc[gn][1], loc[gn][2]).float().abs()
        amax = max(float(w.float().abs().max()) for w in ws)
        acc = collections.defaultdict(lambda: [0.0, 0.0])
        for w in ws:
            wf = w.float()
            g = None
            if gamma is not None:
                g = gamma.repeat(wf.shape[1] // gamma.numel()) if wf.shape[1] % gamma.numel() == 0 else None
            recs = {"mxfp8": dq_mxfp8(*q_mxfp8(w)), "fp8t": dq_fp8_tensor(*q_fp8_tensor(w, amax)),
                    "fp8b128": q_fp8_block(w)}
            if w.shape[1] % 16 == 0:
                recs["nvfp4"] = dq_nvfp4(*q_nvfp4(w, amax, search=False))
                recs["nvfp4s"] = dq_nvfp4(*q_nvfp4(w, amax, search=True))
            for fm, r in recs.items():
                d = r - wf
                acc[fm][0] += float((d * d).sum()); acc[fm][1] += float((wf * wf).sum())
                if g is not None:
                    acc[fm + "_aw"][0] += float(((d * g) ** 2).sum()); acc[fm + "_aw"][1] += float(((wf * g) ** 2).sum())
        rec = {fm: math.sqrt(v[0] / max(v[1], 1e-30)) for fm, v in acc.items()}
        rec["numel"] = sum(w.numel() for w in ws)
        rec["kurt"] = float(torch.cat([w.float().flatten() for w in ws]).pow(4).mean() /
                            torch.cat([w.float().flatten() for w in ws]).pow(2).mean() ** 2)
        rec["sel_nvfp4"] = rec.get("nvfp4s_aw", rec.get("nvfp4s"))
        if gamma is not None:
            rec["gamma_cv"] = float(gamma.std() / gamma.mean())
        out[key] = rec
        print(f"{key:14s} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "numel"), flush=True)
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)
    print(f"proxy done {time.time()-t0:.0f} s")


def cmd_plan(a):
    H = headers(a.src)
    cfg = load_cfg(a.src)
    G = groups_of(H, cfg)
    proxy = json.load(open(a.proxy)) if a.proxy else None
    plan = plan_for(a.arm, G, proxy, a.nv_thresh)
    loc = locate(H)
    saved = 0.0
    bytes_ = collections.defaultdict(float)
    for key, f in plan.items():
        L, kind = key.split(":")
        L = int(L) if L != "head" else "head"
        for m in G[(L, kind)]:
            sh = loc[m][2]["shape"]
            k = kind if kind != "lm_head" else "x"
            if decode_read(kind):
                b0, b1 = per_rank_bytes(k, sh, "bf16"), per_rank_bytes(k, sh, f)
                saved += b0 - b1
                bytes_[f] += b1
    json.dump({"arm": a.arm, "plan": plan, "saved_bytes_per_rank_step": saved,
               "counts": collections.Counter(plan.values())}, open(a.out, "w"), indent=1)
    c = collections.Counter(plan.values())
    print(f"{a.arm}: groups {dict(c)}; decode bytes saved per rank per step {saved/2**20:.0f} MiB "
          f"({saved/235e9*1e3:.2f} ms at 235 GB/s); remaining by format "
          + ", ".join(f"{k} {v/2**20:.0f} MiB" for k, v in bytes_.items()))


def union_names(G):
    return {m for mem in G.values() for m in mem}


def cmd_quant(a):
    import torch
    torch.set_num_threads(a.threads)
    H = headers(a.src)
    cfg = load_cfg(a.src)
    G = groups_of(H, cfg)
    loc = locate(H)
    plan = json.load(open(a.plan))["plan"]
    os.makedirs(a.out, exist_ok=True)
    items, stats, t0 = [], collections.Counter(), time.time()
    for key in sorted(plan, key=lambda s: (s.split(":")[0].zfill(4), s)):
        f = plan[key]
        L, kind = key.split(":")
        mem = G[(int(L) if L != "head" else "head", kind)]
        if f == "bf16" and a.skip_bf16:          # zero-copy arms read BF16-kept tensors from the base shards
            continue
        if f == "bf16":
            for m in mem:
                fn, base, info = loc[m]
                a0, a1 = info["data_offsets"]
                items.append((m, info["dtype"], info["shape"], (os.path.join(a.src, fn), base + a0, a1 - a0)))
            stats["bf16"] += len(mem)
            continue
        ws = {m: tensor(os.path.join(a.src, loc[m][0]), loc[m][1], loc[m][2]) for m in mem}
        amax = max(float(w.float().abs().max()) for w in ws.values())
        for m, w in ws.items():
            b = m[: -len(".weight")]
            if f == "mxfp8":
                q, s = q_mxfp8(w)
                items += [(m, "F8_E4M3", q.shape, q.view(torch.uint8).numpy().tobytes()),
                          (b + ".weight_scale", "U8", s.shape, s.numpy().tobytes())]
            elif f == "fp8":
                q, s = q_fp8_tensor(w, amax)
                items += [(m, "F8_E4M3", q.shape, q.view(torch.uint8).numpy().tobytes()),
                          (b + ".weight_scale", "F32", [1], s.numpy().tobytes()),
                          (b + ".input_scale", "F32", [1], torch.ones(1).numpy().tobytes())]
            elif f == "fp8blk":
                q, si = q_fp8_blk(w)
                items += [(m, "F8_E4M3", q.shape, q.view(torch.uint8).numpy().tobytes()),
                          (b + ".weight_scale_inv", "F32", si.shape, si.numpy().tobytes())]
            elif f == "nvfp4":
                p, sf, s2 = q_nvfp4(w, amax, search=not a.no_search)
                items += [(m, "U8", p.shape, p.numpy().tobytes()),
                          (b + ".weight_scale", "F8_E4M3", sf.shape, sf.view(torch.uint8).numpy().tobytes()),
                          (b + ".weight_scale_2", "F32", [1], s2.numpy().tobytes())]
            stats[f] += 1
        if sum(stats.values()) % 50 < len(mem):
            print(f"  {sum(stats.values())} tensors {time.time()-t0:.0f} s", flush=True)
    name = f"model-qmix-{a.arm}.safetensors"
    write_st(os.path.join(a.out, name), items, {"format": "pt", "qmix_arm": a.arm})
    json.dump({"arm": a.arm, "overlay": name, "plan": plan, "stats": stats,
               "src": os.path.abspath(a.src)}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"overlay {name}: {dict(stats)} {os.path.getsize(os.path.join(a.out, name))/2**30:.2f} GiB "
          f"in {time.time()-t0:.0f} s")


def cmd_strip(a):
    H = headers(a.src)
    cfg = load_cfg(a.src)
    U = union_names(groups_of(H, cfg))
    os.makedirs(a.out, exist_ok=True)
    mtp = re.compile(rf"^{re.escape(P)}{MTP_LAYER}\.")
    t0, drop = time.time(), 0
    for f, (h, base, meta) in H.items():
        src, dst = os.path.join(a.src, f), os.path.join(a.out, f)
        if os.path.exists(dst):
            continue
        keep = [(k, v) for k, v in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0])
                if k not in U and not mtp.match(k)]
        drop += len(h) - len(keep)
        if len(keep) == len(h):
            os.link(src, dst)
            print(f"  {f}: linked", flush=True)
            continue
        if not keep:
            print(f"  {f}: empty after strip, skipped", flush=True)
            continue
        items = [(k, v["dtype"], v["shape"], (src, base + v["data_offsets"][0],
                                               v["data_offsets"][1] - v["data_offsets"][0])) for k, v in keep]
        write_st(dst, items, meta)
        print(f"  {f}: kept {len(keep)}/{len(h)}  {time.time()-t0:.0f} s", flush=True)
    for x in os.listdir(a.src):
        s = os.path.join(a.src, x)
        if os.path.isfile(s) and not x.endswith(".safetensors") and x not in ("model.safetensors.index.json",):
            d = os.path.join(a.out, x)
            if not os.path.exists(d):
                os.link(s, d)
    json.dump({"src": os.path.abspath(a.src), "union": sorted(U), "dropped_mtp": True},
              open(os.path.join(a.out, "strip-manifest.json"), "w"))
    print(f"strip done: removed {drop} tensors, {time.time()-t0:.0f} s")


SENTINEL = "qmix.requires_patch"


def cmd_assemble(a):
    """Zero-copy (default): hardlink every base file, add the overlay and a sentinel shard, point the index of each
    re-encoded tensor at the overlay. The base shards still hold BF16 copies of those tensors; overlay/qmix_patch.py
    drops them at load (QMIX_FP8_BLOCK=1). Without the patch the sentinel tensor has no parameter and the load
    fails loudly instead of casting BF16 into FP8 parameters. --common uses stripped shards instead (no patch
    needed for the filter, ~175 GB per node)."""
    import torch
    man = json.load(open(os.path.join(a.overlay, "manifest.json")))
    os.makedirs(a.dst, exist_ok=True)
    src_dir = a.common or a.base
    skip = {"strip-manifest.json", "config.json", "hf_quant_config.json", "model.safetensors.index.json"}
    for x in os.listdir(src_dir):
        s, d = os.path.join(src_dir, x), os.path.join(a.dst, x)
        if x in skip or os.path.exists(d) or not os.path.isfile(s):
            continue
        os.link(s, d)
    s, d = os.path.join(a.overlay, man["overlay"]), os.path.join(a.dst, man["overlay"])
    if not os.path.exists(d):
        os.link(s, d)
    base_dir = json.load(open(os.path.join(a.common, "strip-manifest.json")))["src"] if a.common else a.base
    cfg_full = json.load(open(os.path.join(base_dir, "config.json")))
    cfg = cfg_full.get("text_config", cfg_full)
    plan = man["plan"]
    ov_names = set(read_header(os.path.join(a.dst, man["overlay"]))[0])
    quantized = sorted(n for n in ov_names if n.endswith(".weight") and
                       (n[:-7] + ".weight_scale" in ov_names or n[:-7] + ".weight_scale_inv" in ov_names))
    zero_copy = not a.common
    if zero_copy:
        write_st(os.path.join(a.dst, "model-qmix-sentinel.safetensors"),
                 [(SENTINEL, "F32", [1], torch.ones(1).numpy().tobytes())])
    ql = quantized_layers(cfg, plan)
    ignore = list(IGNORE_BASE)
    if plan.get("head:lm_head", "bf16") == "bf16":
        ignore.insert(0, "lm_head")
    kv = cfg_full["quantization_config"].get("kv_cache_scheme")
    q = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "group_size": 16, "kv_cache_scheme": kv,
         "ignore": ignore, "quantized_layers": ql,
         "producer": {"name": "modelopt", "note": f"nvidia NVFP4 experts + glm-quant-mix arm {man['arm']}"}}
    cfg_full["quantization_config"] = q
    json.dump(cfg_full, open(os.path.join(a.dst, "config.json"), "w"), indent=1)
    hq = {"producer": q["producer"], "quantization": {"quant_algo": "MIXED_PRECISION",
          "kv_cache_quant_algo": "FP8" if kv else None, "group_size": 16, "exclude_modules": ignore,
          "quantized_layers": ql}}
    json.dump(hq, open(os.path.join(a.dst, "hf_quant_config.json"), "w"), indent=1)
    wm = {}
    for f in sorted(os.listdir(a.dst)):
        if f.endswith(".safetensors") and f != man["overlay"]:
            for k in read_header(os.path.join(a.dst, f))[0]:
                wm.setdefault(k, f)
    for k in ov_names:                     # overlay wins every name it carries
        wm[k] = man["overlay"]
    json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(a.dst, f)) for f in set(wm.values()))},
               "weight_map": dict(sorted(wm.items()))}, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"),
              indent=1)
    if a.common:
        lost = [n for n in json.load(open(os.path.join(a.common, "strip-manifest.json")))["union"] if n not in wm]
        if lost:
            raise SystemExit(f"overlay lacks {len(lost)} stripped tensors, e.g. {lost[:3]}")
    man.update({"zero_copy": zero_copy, "quantized_names": quantized, "sentinel": SENTINEL if zero_copy else None,
                "base": base_dir})
    json.dump(man, open(os.path.join(a.dst, "qmix-manifest.json"), "w"), indent=1)
    print(f"assembled {a.dst} ({'zero-copy' if zero_copy else 'stripped'}): {len(wm)} names, "
          f"{len(quantized)} re-encoded weights, quantized_layers {len(ql)}")
    cmd_verify(argparse.Namespace(dst=a.dst, zero_copy=zero_copy, overlay=man["overlay"]))


def cmd_verify(a):
    present = collections.defaultdict(list)
    for f in sorted(os.listdir(a.dst)):
        if f.endswith(".safetensors"):
            for k in read_header(os.path.join(a.dst, f))[0]:
                present[k].append(f)
    idx = json.load(open(os.path.join(a.dst, "model.safetensors.index.json")))["weight_map"]
    zc = getattr(a, "zero_copy", False)
    ov = getattr(a, "overlay", None)
    # zero-copy: a name may exist twice only as (base BF16 copy, overlay); the index must point at the overlay
    dups = [k for k, v in present.items() if len(v) > 1 and not (zc and ov in v and len(v) == 2 and idx.get(k) == ov)]
    missing = [k for k in idx if k not in present]
    wrong = [k for k in idx if k in present and idx[k] not in present[k]]
    orphan = [k for k in present if k.endswith((".weight_scale", ".weight_scale_inv")) and
              k.rsplit(".", 1)[0] + ".weight" not in present]
    print(f"VERIFY dups {len(dups)} | indexed-missing {len(missing)} | index-wrong-file {len(wrong)} | "
          f"orphan scales {len(orphan)} | names {len(present)}")
    if dups or missing or wrong or orphan:
        raise SystemExit(1)


def cmd_estimate(a):
    """DRAM-bound decode model (glm-review estimate.py structure), per arm, from plan files."""
    base_step = {"c1 prose": (59.0, 2.2, 1), "c1 code": (72.0, 5.6, 1), "c4 prose": (None, 2.2, 4),
                 "c16 prose": (None, 2.2, 16)}
    E, k, L_moe, pe = 288, 8, 42, 3 * 4096 * 2048

    def union(rows):
        return max(k, E * (1 - (1 - k / E) ** rows) * 0.75) if rows > 1 else k

    rows = []
    for pf in a.plans:
        p = json.load(open(pf))
        rows.append((p["arm"], p["saved_bytes_per_rank_step"]))
    print("arm | saved MiB/rank/step | c1 prose | c1 code | c4 prose agg | c16 prose agg")
    for arm, saved in rows:
        ms = saved / a.bw * 1e3
        out = []
        for cell, (st, acc, c) in base_step.items():
            if st is None:     # review model for c>1: bytes of union + non-expert, + overhead
                vr = 4
                u = union(c * vr)
                by = u * pe * 4.5 / 8 * L_moe / 4 + 4.5e9 * 2 / 4 + 42 * 3 * 4096 * 2048 * 2 / 4
                st = by / a.bw * 1e3 + 28 + 2 * (c - 1)
            out.append(c * acc * 1000 / (st - ms))
        print(f"{arm:12s} | {saved/2**20:7.0f} | " + " | ".join(f"{x:6.1f}" for x in out))


def cmd_selftest(a):
    import torch
    torch.manual_seed(0)
    for shape in [(8192, 4096), (4096, 16384), (128, 4096), (2048, 4096)]:
        w = (torch.randn(shape) * 0.02).to(torch.bfloat16)
        wf = w.float()
        amax = float(wf.abs().max())
        r = {"mxfp8": rel(dq_mxfp8(*q_mxfp8(w)), wf), "fp8t": rel(dq_fp8_tensor(*q_fp8_tensor(w)), wf),
             "fp8b128": rel(q_fp8_block(w), wf), "fp8blk": rel(dq_fp8_blk(*q_fp8_blk(w)), wf), "nvfp4": rel(dq_nvfp4(*q_nvfp4(w, amax, False)), wf),
             "nvfp4s": rel(dq_nvfp4(*q_nvfp4(w, amax, True)), wf)}
        print(shape, " ".join(f"{k} {v:.4f}" for k, v in r.items()))
        assert r["mxfp8"] < 0.04 and r["nvfp4"] < 0.12 and r["nvfp4s"] <= r["nvfp4"] + 1e-6
    try:   # cross-check against vLLM's own dequantizers when run inside the image
        from vllm.model_executor.layers.quantization.utils import mxfp8_utils as M
        w = (torch.randn(256, 512) * 0.02).bfloat16()
        q, s = q_mxfp8(w)
        print("vLLM mxfp8 dequant max diff", float((M.dequant_mxfp8_to_bf16(q, s).float() - dq_mxfp8(q, s)).abs().max()))
    except Exception as e:  # noqa: BLE001
        print("vLLM mxfp8 cross-check skipped:", repr(e)[:120])
    try:
        from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import break_fp4_bytes
        w = (torch.randn(256, 512) * 0.02).bfloat16()
        p, sf, s2 = q_nvfp4(w, float(w.float().abs().max()), True)
        v = break_fp4_bytes(p, torch.float32).view(256, -1, 16)      # vLLM's nibble order and sign rule
        ref = (v * (sf.float() * float(s2[0])).unsqueeze(-1)).view(256, -1)
        print("vLLM nvfp4 unpack max diff", float((ref - dq_nvfp4(p, sf, s2)).abs().max()))
    except Exception as e:  # noqa: BLE001
        print("vLLM nvfp4 cross-check skipped:", repr(e)[:160])
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    x = sp.add_parser("inventory"); x.add_argument("--src", required=True)
    x = sp.add_parser("proxy"); x.add_argument("--src", required=True); x.add_argument("--out")
    x.add_argument("--only", default=""); x.add_argument("--threads", type=int, default=8)
    x = sp.add_parser("plan"); x.add_argument("--src", required=True); x.add_argument("--arm", required=True)
    x.add_argument("--proxy"); x.add_argument("--nv-thresh", type=float); x.add_argument("--out", required=True)
    x = sp.add_parser("quant"); x.add_argument("--src", required=True); x.add_argument("--plan", required=True)
    x.add_argument("--arm", required=True); x.add_argument("--out", required=True)
    x.add_argument("--no-search", action="store_true"); x.add_argument("--threads", type=int, default=8)
    x.add_argument("--skip-bf16", action="store_true", help="omit BF16-kept tensors (zero-copy assembly only)")
    x = sp.add_parser("strip"); x.add_argument("--src", required=True); x.add_argument("--out", required=True)
    x = sp.add_parser("assemble"); x.add_argument("--base"); x.add_argument("--common"); x.add_argument("--overlay", required=True)
    x.add_argument("--dst", required=True)
    x = sp.add_parser("verify"); x.add_argument("--dst", required=True)
    x = sp.add_parser("estimate"); x.add_argument("plans", nargs="+"); x.add_argument("--bw", type=float, default=235e9)
    sp.add_parser("selftest")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
