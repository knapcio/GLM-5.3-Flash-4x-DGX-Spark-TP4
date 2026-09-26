# GLM-5.3-Flash on 4× DGX Spark

A measured **vLLM TP4** recipe for GLM-5.3-Flash on four NVIDIA DGX Spark / GB10 nodes connected through a RoCE switch. The accepted profile is **LVKP-S-L2**, qualified on 2026-09-26: NVFP4 routed experts, 8-bit non-expert weights, DFlash2, FP8 KV cache, and one OpenAI-compatible endpoint for text, tools, reasoning and images.

This recipe builds on **tonyd2wild's SM121 vLLM image**, **Jacopo Nardiello's scheduler**, **local-inference-lab's b12x / RoCEnante**, **incoai's DFlash2** and the upstream vLLM kernels. See [full credits and component licences](CREDITS.md).

## Current measurements

The release figures are from **sparkDash DecodeBench**, 256 output tokens, temperature 0, thinking off, on the accepted L2 production stack. Prose c1 uses the median of **5** scored runs; prose c4 uses the median of **3** aggregate-throughput runs. Two warm-ups are excluded. Other matrix cells are single measurements, not repeated-run medians.

Measured on **2026-09-26 after restoring production**, with all 20 jobs / 100 streams validated:

| Prompt | c1 tok/s | c2 aggregate | c4 aggregate | c8 aggregate | c16 aggregate |
|---|---:|---:|---:|---:|---:|
| Prose | **70.19** (n=5) | 107.45 | **152.89** (n=3) | 221.70 | **312.69** |
| Code | 126.44 | — | 185.37 | — | 308.26 |
| Structured | 161.11 | — | — | — | 323.78 |
| JSON | 124.25 | — | — | — | 476.41 |

All unmarked cells have n=1. A dash means this matrix did not measure that cell. [Samples, checks and provenance](docs/results/2026-09-26-l2.md) · [Machine-readable results](docs/results/2026-09-26-l2.json).

The earlier L2 promotion measurement was 70.84 / 151.41 tok/s at prose c1 / aggregate c4. The fresh measurement is a repeat on the same accepted configuration, not another optimization.

Performance is workload-dependent. These short-prompt decode measurements do not predict long-context decode or reasoning throughput. c4 aggregate throughput is the combined output rate, not the rate of each user stream.

The same accepted L2 stack passed **qeval 75/75 at c1 and 75/75 at c4**. Its retained KLD measurement is **0.0288366**, over **17 items / 6,618 teacher-forced positions**, against the recorded BF16-attention reference. These quality results were retained, not rerun with the fresh performance matrix. This is a specific reference and test panel, not a claim of equality to the full BF16 model. The exact KLD panel contains private operational material and is not distributed; its hashes document provenance but do not make that specific value reproducible from this repository alone. The long-context registry sanity gate passed **32/32 lookups at both concurrencies**, with approximately 9.6k-token prompts; it is not a 262k-context quality qualification. See [validation and provenance](docs/validation.md).

## Serving profile

| Component | Accepted setting |
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

`lossless8` is the historical **profile name**, not a mathematically lossless conversion from BF16. The representation was chosen per tensor to stay close to the released weights. See [weight preparation](docs/weights.md) for the conversion and model licences. The API alias is retained for client compatibility; it does not mean the routed experts are FP8.

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

Run the [validation procedure](docs/validation.md) on your deployment. The recorded results qualify the existing production stack; publishing this recipe does **not** constitute a new fresh-clone build-and-boot validation.

## What the optimized stack does

- **Batch-uniform adaptive draft:** picks one draft length for the whole running batch, avoiding mixed-k steps that fall back from full CUDA graph replay.
- **KDA verification-state stash and trims:** reuses verification state and removes redundant copies / flag work. The stash has a measured FP32-state difference at the ulp level; the release does not claim whole-model byte identity.
- **Replicated projection splitting:** distributes supported repeated projection work across ranks. Target paths carry numerical checks; the drafter remains subject to target verification.
- **Vocab-parallel greedy selection:** avoids gathering full target logits where the sampling contract permits exact argmax, including compatible min-token requests.
- **L2 prefetch:** overlaps read-only weight prefetch with attention / communication. The accepted in-boot qualification measured step savings of **0.610 ms at c1** and **0.727 ms at c4**; these are auxiliary step-timer results, not token-throughput measurements.
- **Indexer correctness fixes:** carries padded seed stride, speculative ring and hybrid tail-slot mapping fixes together.
- **Faster loading and persistent JIT caches:** uses the slab loader and keeps compilation caches between boots. A retained warm-cache L2 boot took **129 seconds**; first setup and a cold compile are different workloads.

The enabled switches and their implementation are listed in [runtime notes](docs/runtime.md). Experimental modules may be retained as source dependencies, but GDN metadata optimization, router deduplication, mHC fusion and diagnostic A/B hooks are **off** in the accepted profile.

## Limits and history

- Performance measurements use one four-node switched fleet; other networks and software versions need their own validation.
- Greedy output text can vary on this runtime. Tensor-level checks, quality scores and KLD are the qualification evidence; matching prose alone is insufficient.
- A qeval pass does not establish broad benchmark parity or long-run stability. Native tools, reasoning and vision depend on the pinned parser and model-template versions.
- The drafter has its own licence. The repository licence does not relicense model weights or upstream components.

Previous stacks, benchmark settings, comparisons and rejected experiments are in [history](docs/history.md). The [2026-09-18 README](docs/history-2026-09-18.md) is preserved separately; its numbers are not the current release figures.
