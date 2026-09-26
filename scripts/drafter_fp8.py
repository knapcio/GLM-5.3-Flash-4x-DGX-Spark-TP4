#!/usr/bin/env python3
"""Re-encode the incoai GLM-5.3-Flash-DFlash2 drafter's decoder linears as block-128 FP8 (W8A16 on Marlin).

Same weights, same drafter: no training, no other checkpoint. Only the 5 decoder layers' q/k/v/o and
gate/up/down projections are re-encoded, with the exact block-128 grid code the target's lossless8 FP8
layers use (diagnostics/glm-quant-mix/tools/glm_quant_mix.py q_fp8_blk: scale = block amax / 448,
weight F8_E4M3 [N,K] + weight_scale_inv F32 [N/128, K/128]). vLLM builds them with Fp8LinearMethod from
the draft's own quantization_config (models/utils.get_draft_quant_config); with VLLM_TEST_FORCE_FP8_MARLIN=1
(already set by the lossless8 profile) that is Marlin W8A16, BF16 activations.

Kept BF16: model.fc (so glm_ds_split can split it over TP instead), the conv kernel projections and the
candidate selector (built with quant_config=None by the model code), norms, codebooks.

The drafter was trained in BF16 and sits on no FP8 grid, so this is NOT lossless for the drafter: the
per-tensor relative error is printed. Output tokens stay exact because the target verifies every draft;
only acceptance can move. Licence: incoai DFlash2 is CC BY-NC-ND; keep the re-encoded copy local.

usage: drafter_fp8.py SRC_DIR DST_DIR
"""
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm_quant_mix import dq_fp8_blk, q_fp8_blk  # noqa: E402

QUANT = re.compile(r"^layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def main(src, dst):
    os.mkdir(dst)  # refuse existing output; preserve partial conversions
    sd = load_file(os.path.join(src, "model.safetensors"))
    out, errs, nq, bytes_in, bytes_out = {}, [], 0, 0, 0
    for name, w in sd.items():
        if QUANT.match(name):
            q, sc = q_fp8_blk(w)
            rel = float((dq_fp8_blk(q, sc) - w.float()).norm() / w.float().norm())
            errs.append((name, rel))
            out[name] = q
            out[name[:-len("weight")] + "weight_scale_inv"] = sc
            nq += 1
            bytes_in += w.numel() * 2
            bytes_out += q.numel() + sc.numel() * 4
        else:
            out[name] = w.contiguous()
    save_file(out, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["quantization_config"] = {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "ignored_layers": ["model.fc", "fc"],
    }
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for extra in ("README.md",):
        if os.path.exists(os.path.join(src, extra)):
            shutil.copy(os.path.join(src, extra), os.path.join(dst, extra))
    rel = [e for _, e in errs]
    print(f"quantized {nq} tensors: {bytes_in / 2**20:.0f} MiB bf16 -> {bytes_out / 2**20:.0f} MiB fp8+scales")
    print(f"relative Frobenius error: mean {sum(rel) / len(rel) * 100:.2f} %  "
          f"min {min(rel) * 100:.2f} %  max {max(rel) * 100:.2f} %")
    for n, e in errs[:7]:
        print(f"  {n}: {e * 100:.2f} %")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
