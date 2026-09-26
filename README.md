# GLM-5.3-Flash on 4× DGX Spark

Run **GLM-5.3-Flash on four NVIDIA DGX Spark / GB10 nodes** with vLLM TP4, NVFP4 routed experts, 8-bit non-expert weights and DFlash2 speculative decoding. The LVKP-S-L2 profile supports **262k context**, up to **32 concurrent sequences**, and an OpenAI-compatible API for text, tools, reasoning and images. The tested fleet uses a RoCE switch.

This recipe builds on **tonyd2wild's SM121 vLLM image**, **Jacopo Nardiello's scheduler**, **local-inference-lab's b12x / RoCEnante**, **incoai's DFlash2** and the upstream vLLM kernels. See [full credits and component licences](CREDITS.md).

## Current measurements

**Decode throughput, tok/s — sparkDash, 2026-09-26.** 256 output tokens, temperature 0, thinking off. At c2–c16, values are aggregate throughput, with mean per-stream tok/s in parentheses.

| Prompt | c1 tok/s | c2 aggregate | c4 aggregate | c8 aggregate | c16 aggregate |
|---|---:|---:|---:|---:|---:|
| Prose | **70.19** | 107.45 (57.95) | **152.89** (39.39) | 221.70 (28.83) | **312.69** (20.34) |
| Code | 126.44 | 160.34 (81.53) | 185.37 (50.37) | 231.27 (31.77) | 308.26 (21.37) |
| Structured | 161.11 | 140.48 (80.32) | 196.68 (52.98) | 242.16 (32.61) | 323.78 (23.67) |
| JSON | 124.25 | 115.55 (61.73) | 193.62 (51.44) | 303.75 (41.90) | 476.41 (32.25) |

Prose c1 is the median of five runs; prose c4 reports the median of three runs for each metric. Other cells are single runs. These are short-prompt decode measurements. [Results and methodology](docs/results/2026-09-26-l2.md) · [Additional measurements](docs/results/2026-09-26-l2-supplement.md).

**Prefill probe:** **~2.2k input tokens/s at 16k–64k**, measured to the first output token, with three runs per prompt length. [Prefill measurements](docs/results/2026-09-26-prefill.md).

**Quality:** qeval **75/75 at c1 and c4**. The separate teacher-forced KLD evaluation measured **0.02884** against the BF16-attention reference on a private 17-item panel. See [validation](docs/validation.md) for the reference, test scope and reproducibility limits.

## Serving profile

| Component | Setting |
|---|---|
| Target | NVIDIA NVFP4 routed experts; `lossless8` non-expert conversion |
| Non-expert weights | KDA projections on the MXFP8 grid; MLA and shared-expert projections in block-128 FP8; selected tensors retained in BF16 |
| Dense / expert kernels | Marlin W8A16 / NVFP4, with BF16 activations |
| Parallelism | Tensor parallelism across 4 nodes |
| Draft | incoai DFlash2 with FP8-block decoder weights and an adaptive, batch-uniform k=3 or k=7 |
| Context / admission | 262,144 maximum context; up to 32 sequences |
| KV cache | 24 GiB FP8 KV per rank; token capacity depends on the runtime layout |
| CUDA graphs | FULL_DECODE_ONLY, capture sizes through 256 tokens |
| Prefill | 6,912-token chunks |
| Communication | RoCEnante small all-reduce / all-gather; NCCL over both configured RoCE rails |
| API | Loopback on the head, port 8093; served model alias `GLM-5.3-Flash-FP8` |

The `lossless8` conversion chooses an 8-bit representation per tensor to reduce re-encoding error; the name does not imply mathematical losslessness. See [weight preparation](docs/weights.md). The historical API alias `GLM-5.3-Flash-FP8` is kept for client compatibility.

## Reproduce

1. Prepare four ARM64 DGX Sparks with Docker GPU access, working SSH management paths, and a verified RoCE fabric. Configure the real interface names, addresses and GID on your fleet.
2. Follow [installation and image build](docs/install.md), including the pinned base image, RoCEnante and NCCL dependencies. Build on idle nodes.
3. Follow [weight preparation](docs/weights.md). Target and drafter directories must exist at matching paths on every node. Model weights are not distributed in this repository.
4. Copy `.env.example` to `.env` and edit the host, fabric and model paths. The example selects the accepted L2 profile.
5. Start and inspect the service:

```bash
./start.sh serve
./start.sh status
./start.sh logs 0 80
```

The API binds to loopback by default. Use SSH forwarding or an authenticated private-network proxy for remote access; do not expose the model endpoint through a public router.

```bash
curl http://127.0.0.1:8093/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash-FP8","messages":[{"role":"user","content":"What is 19 + 23?"}],"max_tokens":128,"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

Run the [validation procedure](docs/validation.md) on your deployment. These measurements come from the qualified serving stack; a fresh deployment from the public package has not yet been tested on all four nodes.

## What the optimized stack does

- **Batch-uniform adaptive draft:** picks one draft length for the whole running batch, avoiding mixed-k steps that fall back from full CUDA graph replay.
- **KDA verification-state stash and trims:** reuses verification state and removes redundant copies / flag work. FP32-state differences are at the ulp level.
- **Replicated projection splitting:** distributes supported repeated projection work across ranks. Target paths carry numerical checks; the drafter remains subject to target verification.
- **Vocab-parallel greedy selection:** avoids gathering full target logits where the sampling contract permits exact argmax, including compatible min-token requests.
- **L2 prefetch:** overlaps read-only weight prefetch with attention / communication. Step-time measurements are recorded in [history](docs/history.md).
- **Indexer correctness fixes:** carries padded seed stride, speculative ring and hybrid tail-slot mapping fixes together.
- **Faster loading and persistent JIT caches:** uses the slab loader and keeps compilation caches between boots. Measured warm-cache boot: **129 seconds**. First-time compilation takes longer.

See [runtime notes](docs/runtime.md) for enabled switches, source versions and implementation details.

## Further reading

- [Installation](docs/install.md): image build, NCCL, fabric settings and launch.
- [Weight preparation](docs/weights.md): target and drafter conversion, model licences.
- [Validation](docs/validation.md): benchmark commands, quality checks and known limits.
- [History](docs/history.md): earlier results and optimization experiments.
- [Credits](CREDITS.md): upstream authors, projects and component licences.
