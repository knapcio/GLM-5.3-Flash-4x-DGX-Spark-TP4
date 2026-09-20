"""Quantise the weights RedHatAI/GLM-5.3-Flash-NVFP4 left in BF16 to NVFP4, shard by shard.

Why: that checkpoint quantises only the routed experts. Everything else is in its `ignore` list, so
18.01 GiB of attention and dense-MLP weights stay BF16 and are read on every decode step (4.5 GiB per
rank at TP4, about 18 ms at 250 GB/s). tonyd2wild measured 67.6 -> 57 ms per step after quantising
them, and +27 % on prose (https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark).

What this does: a pure tensor transform, no GPU and no calibration data, because the scheme is
symmetric static weight-only. For every targeted `X.weight` it writes the four tensors the
compressed-tensors NVFP4 loader expects, mirroring exactly what the routed experts already use in
this checkpoint:

    X.weight_packed        uint8    [out, in/2]     two E2M1 values per byte
    X.weight_scale         fp8_e4m3 [out, in/16]    per-block scale
    X.weight_global_scale  float32  [1]             second-level scale
    X.input_global_scale   float32  [1]             for the dynamic-local activation scheme

and moves those layer names out of `quantization_config.ignore` into a config group.

    python3 quantize_dense_nvfp4.py --src /models/RedHatAI/GLM-5.3-Flash-NVFP4 \
        --dst /models/GLM-5.3-Flash-NVFP4-dense --self-test          # check the math first
    python3 quantize_dense_nvfp4.py --src ... --dst ... --write      # ~200 GB of writes

Not yet validated by a boot: vLLM must accept NVFP4 on these shapes on SM121, and the quality gate
(bench/qeval.py) has to run before this becomes the serving checkpoint. Treat it as a candidate.
"""
import argparse, glob, json, os, re, shutil, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# E2M1: the eight magnitudes NVFP4 can represent, and 6.0 as the per-block maximum.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_MAX = 6.0
FP8_MAX = 448.0
BLOCK = 16

# Quantise 2-D linear weights of the language model's attention and dense MLP. Everything else stays
# as it is: embeddings and the head (output quality), the vision tower (tiny and already excluded
# upstream), norms, biases and the 1-D KDA parameters (A_log, dt_bias) that are not matmul weights.
TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|"
    r"b_proj|g_a_proj|g_b_proj|f_a_proj|f_b_proj|forget_gate\.f_a_proj|forget_gate\.f_b_proj|"
    r"indexer\.(wq_b|wk|weights_proj))"
    r"|mlp\.shared_experts\.(gate_proj|up_proj|down_proj))\.weight$"
)


def quantise_fp8(w: torch.Tensor, block=128):
    """BF16 [out, in] -> (fp8_e4m3 weight, float32 scale [ceil(out/128), ceil(in/128)]).

    Mirrors the block scheme this checkpoint already uses for the layer-45 experts. About 1.5 % of
    relative RMS error against 9 % for NVFP4, and it still halves the bytes read per step."""
    w32 = w.to(torch.float32)
    out_f, in_f = w32.shape
    ob, ib = (out_f + block - 1) // block, (in_f + block - 1) // block
    pad = torch.zeros(ob * block, ib * block, dtype=torch.float32)
    pad[:out_f, :in_f] = w32
    tiles = pad.view(ob, block, ib, block).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=1e-12)
    scale = (amax / FP8_MAX)
    q = (tiles / scale.unsqueeze(-1).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.permute(0, 2, 1, 3).reshape(ob * block, ib * block)[:out_f, :in_f].contiguous()
    return q, scale.to(torch.float32).contiguous()


def dequantise_fp8(q, scale, block=128):
    out_f, in_f = q.shape
    ob, ib = scale.shape
    pad = torch.zeros(ob * block, ib * block, dtype=torch.float32)
    pad[:out_f, :in_f] = q.to(torch.float32)
    tiles = pad.view(ob, block, ib, block).permute(0, 2, 1, 3) * scale.unsqueeze(-1).unsqueeze(-1)
    return tiles.permute(0, 2, 1, 3).reshape(ob * block, ib * block)[:out_f, :in_f]


def quantise(w: torch.Tensor):
    """BF16 [out, in] -> (packed uint8 [out, in/2], scale fp8 [out, in/16], global scale [1])."""
    w32 = w.to(torch.float32)
    out_f, in_f = w32.shape
    assert in_f % BLOCK == 0, f"input dim {in_f} is not a multiple of {BLOCK}"
    amax = w32.abs().amax().clamp(min=1e-12)
    # Second-level scale: put the largest block scale at the top of the fp8 range.
    global_scale = (FP8_MAX * E2M1_MAX) / amax
    blocks = w32.view(out_f, in_f // BLOCK, BLOCK)
    block_amax = blocks.abs().amax(dim=-1).clamp(min=1e-12)
    # Per-block scale search: clipping slightly below the block max lowers the reconstruction error,
    # because it buys resolution for the many small values at the cost of the single largest one.
    best_err = best_scale = None
    for c in (1.0, 0.95, 0.9, 0.85, 0.8):
        cand = ((block_amax * c / E2M1_MAX) * global_scale).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
        eff_c = (cand.to(torch.float32) / global_scale).clamp(min=1e-12)
        n = blocks / eff_c.unsqueeze(-1)
        i = (n.abs().clamp(max=E2M1_MAX).unsqueeze(-1) - E2M1).abs().argmin(dim=-1)
        rec = torch.sign(n) * E2M1[i] * eff_c.unsqueeze(-1)
        err = (rec - blocks).pow(2).sum(dim=-1)
        if best_err is None:
            best_err, best_scale = err, cand.to(torch.float32)
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_scale = torch.where(take, cand.to(torch.float32), best_scale)
    scale_fp8 = best_scale.to(torch.float8_e4m3fn)
    eff = scale_fp8.to(torch.float32) / global_scale          # what the kernel will multiply by
    eff = torch.where(eff > 0, eff, torch.ones_like(eff))
    norm = blocks / eff.unsqueeze(-1)
    # Round each value to the nearest representable E2M1 magnitude, keeping the sign.
    sign = torch.sign(norm)
    mag = norm.abs().clamp(max=E2M1_MAX)
    idx = (mag.unsqueeze(-1) - E2M1.to(mag.device)).abs().argmin(dim=-1)
    codes = idx.to(torch.uint8) | torch.where(sign < 0, torch.tensor(8, dtype=torch.uint8),
                                              torch.tensor(0, dtype=torch.uint8))
    codes = codes.view(out_f, in_f)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, scale_fp8.contiguous(), torch.tensor([global_scale], dtype=torch.float32)


def dequantise(packed, scale_fp8, global_scale):
    """Inverse of quantise(), for the self-test only."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out_f = packed.shape[0]
    codes = torch.stack([lo, hi], dim=-1).view(out_f, -1)
    mag = E2M1.to(codes.device)[(codes & 0x07).long()]
    val = torch.where((codes & 0x08) > 0, -mag, mag)
    eff = scale_fp8.to(torch.float32) / global_scale.item()
    return (val.view(out_f, -1, BLOCK) * eff.unsqueeze(-1)).view(out_f, -1)


def self_test():
    torch.manual_seed(0)
    for shape in [(4096, 4096), (128, 4096), (64, 4096), (1536, 5120)]:
        w = (torch.randn(shape) * 0.02).to(torch.bfloat16)
        p, sc, g = quantise(w)
        rel4 = (dequantise(p, sc, g) - w.to(torch.float32)).norm() / w.to(torch.float32).norm()
        q8, s8 = quantise_fp8(w)
        rel8 = (dequantise_fp8(q8, s8) - w.to(torch.float32)).norm() / w.to(torch.float32).norm()
        print(f"  {str(shape):14} nvfp4 rel-rms {rel4:.4f} ({p.numel()/w.numel():.2f} B/weight)  "
              f"fp8 rel-rms {rel8:.4f} (1.00 B/weight)")
        assert rel4 < 0.12 and rel8 < 0.03, "round trip worse than expected; do not ship this"
    print("  self-test OK. NVFP4 costs about 9 % relative weight error, block-128 FP8 about 2.6 %;")
    print("  that gap is the whole quality question, so run bench/qeval.py before serving either.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--write", action="store_true", help="actually write; default is a dry run")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--scheme", choices=("fp8", "nvfp4"), default="fp8",
                    help="fp8 halves the bytes at 2.6 %% weight error; nvfp4 quarters them at 9 %%")
    a = ap.parse_args()
    if a.self_test:
        print("NVFP4 round trip:")
        self_test()
        if not a.src:
            return
    assert a.src, "--src is required unless you only run --self-test"
    files = sorted(glob.glob(os.path.join(a.src, "*.safetensors")))
    assert files, f"no safetensors in {a.src}"
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    ignore = list(cfg.get("quantization_config", {}).get("ignore", []))
    planned, bytes_in, bytes_out = [], 0, 0
    if a.write:
        assert a.dst, "--write needs --dst"
        os.makedirs(a.dst, exist_ok=True)
    for f in files:
        name = os.path.basename(f)
        keep, new = {}, {}
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k) if (a.write or TARGET.match(k)) else None
                if TARGET.match(k) and t is not None and t.dtype == torch.bfloat16 and t.dim() == 2 \
                        and t.shape[1] % BLOCK == 0:
                    base = k[: -len(".weight")]
                    planned.append(base)
                    bytes_in += t.numel() * 2
                    if a.write and a.scheme == "nvfp4":
                        p, sc, g = quantise(t)
                        new[base + ".weight_packed"] = p
                        new[base + ".weight_scale"] = sc
                        new[base + ".weight_global_scale"] = g
                        new[base + ".input_global_scale"] = g.clone()
                        bytes_out += p.numel() + sc.numel() + 8
                    elif a.write:
                        q8, s8 = quantise_fp8(t)
                        new[base + ".weight"] = q8
                        new[base + ".weight_scale"] = s8
                        bytes_out += q8.numel() + s8.numel() * 4
                    else:
                        bytes_out += (t.numel() // 2 + t.numel() // BLOCK + 8) if a.scheme == "nvfp4" \
                            else (t.numel() + t.numel() // (128 * 128) * 4)
                elif a.write:
                    keep[k] = t
        if a.write:
            keep.update(new)
            save_file(keep, os.path.join(a.dst, name), metadata={"format": "pt"})
            print(f"  wrote {name}: {len(keep)} tensors", flush=True)
    print(f"targets: {len(planned)} weights, {bytes_in/2**30:.2f} GiB BF16 -> "
          f"{bytes_out/2**30:.2f} GiB NVFP4 (saves {(bytes_in-bytes_out)/2**30:.2f} GiB, "
          f"{(bytes_in-bytes_out)/4/2**30:.2f} GiB per rank at TP4)")
    if not a.write:
        print("dry run; pass --write to produce the checkpoint")
        return
    # config.json: take the quantised layers out of `ignore` and give them a group of their own.
    q = cfg.setdefault("quantization_config", {})
    q["ignore"] = [n for n in ignore if n not in set(planned)]
    groups = q.setdefault("config_groups", {})
    if a.scheme == "nvfp4":
        groups["group_dense_nvfp4"] = {
            "targets": sorted(planned),
            "weights": {"num_bits": 4, "type": "float", "symmetric": True, "strategy": "tensor_group",
                        "group_size": BLOCK, "dynamic": False, "observer": "memoryless_minmax",
                        "scale_dtype": "torch.float8_e4m3fn"},
            "input_activations": {"num_bits": 4, "type": "float", "symmetric": True,
                                  "strategy": "tensor_group", "group_size": BLOCK, "dynamic": "local",
                                  "observer": "static_minmax", "scale_dtype": "torch.float8_e4m3fn"},
        }
    else:
        # Same shape of declaration as this checkpoint's layer-45 expert group.
        groups["group_dense_fp8"] = {
            "targets": sorted(planned),
            "weights": {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "block",
                        "block_structure": [128, 128], "dynamic": False,
                        "observer": "memoryless_minmax"},
            "input_activations": {"num_bits": 8, "type": "float", "symmetric": True,
                                  "strategy": "group", "group_size": 128, "dynamic": True},
        }
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=1)
    for extra in ("chat_template.jinja", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "preprocessor_config.json", "README.md"):
        src = os.path.join(a.src, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(a.dst, extra))
    idx = os.path.join(a.src, "model.safetensors.index.json")
    if os.path.exists(idx):
        m = json.load(open(idx))
        wm = {}
        for k, v in m["weight_map"].items():
            base = k[: -len(".weight")] if k.endswith(".weight") else None
            if base in set(planned):
                sufs = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale") \
                    if a.scheme == "nvfp4" else ("weight", "weight_scale")
                for suf in sufs:
                    wm[f"{base}.{suf}"] = v
            else:
                wm[k] = v
        m["weight_map"] = wm
        json.dump(m, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=1)
    print("done; boot it with MODEL_DIR pointing here and run bench/qeval.py before trusting it")


if __name__ == "__main__":
    main()
