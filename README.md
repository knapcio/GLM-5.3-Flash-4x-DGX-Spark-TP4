# GLM-5.3-Flash on 4× DGX Spark

Run **GLM-5.3-Flash on four NVIDIA DGX Spark / GB10 nodes** with vLLM TP4, NVFP4 routed experts, 8-bit non-expert weights and DFlash2 speculative decoding. The selected GDN/router profile supports **262k context**, up to **32 concurrent sequences**, and an OpenAI-compatible API for text, tools, reasoning and images. The tested fleet uses a RoCE switch.

This recipe builds on **tonyd2wild's SM121 vLLM image**, **Jacopo Nardiello's scheduler**, **local-inference-lab's b12x / RoCEnante**, **incoai's DFlash2** and the upstream vLLM kernels. See [full credits and component licences](CREDITS.md).

The current recipe adds **GDN metadata fusion and router deduplication** to
the L2 profile. The full 20-cell sparkDash results are below; previous results
are in [measurement history](docs/history.md).

**Quality limits (pre-fix panel):** qeval **72/75 at c1** versus accepted **75/75** (one
truncation; `code_two_sum`, `math_m9`, `reason_r4` failed) and **75/75 at c4**.
Teacher-forced mean KL was **0.029189** versus accepted **0.028837** over 17
items/6,618 positions. The results meet the predefined validation gates, but do not establish
unchanged quality. [Full returned panel and limits](docs/results/2026-09-27-gdn-router-admitted.md).

**Correctness update (2026-09-27):** fixed KDA state migration at speculative
block boundaries, which could produce NaNs and repeated output. Existing
installations need a coordinated restart with the updated overlays.
[Fix, validation and upgrade notes](docs/results/2026-09-27-kda-boundary-fix.md).

## Latest measurements

The throughput, prefill and quality measurements below **predate the KDA boundary
fix**. Its performance impact and full qeval/KLD panel have not been remeasured.

**Decode throughput, tok/s — sparkDash, 2026-09-27.** 256 output tokens,
temperature 0, thinking off. At c2–c16, values are aggregate throughput,
with mean per-stream tok/s in parentheses. Prose c1 is median of five runs;
prose c4 is median of three. All other cells are single observations,
including eight later same-boot supplemental cells.

| Prompt | c1 tok/s | c2 aggregate | c4 aggregate | c8 aggregate | c16 aggregate |
|---|---:|---:|---:|---:|---:|
| Prose | **71.64** | 102.43 (53.09) | **149.44** (39.41) | 214.70 (28.06) | 305.59 (20.29) |
| Code | 123.93 | 151.40 (84.87) | 168.71 (50.28) | 243.43 (33.42) | 289.58 (21.67) |
| Structured | 163.07 | 144.88 (83.15) | 194.03 (52.05) | 270.84 (38.69) | 300.21 (21.98) |
| JSON | 126.16 | 126.36 (64.96) | 208.55 (54.33) | 297.32 (38.31) | 439.12 (30.46) |

Against the prior accepted L2 serving panel, prose c1 was **+2.07%**
(70.19→71.64 tok/s), while prose c4 aggregate was **−2.26%**
(152.89→149.44 tok/s). These are descriptive cross-session comparisons
without a confidence interval or causal isolation. [Full 20-cell data and
methodology](docs/results/2026-09-27-gdn-router-admitted.md).

**Fresh prefill probe:** median **2,202 / 2,209 / 2,197 input tok/s** at
nominal 16k / 32k / 64k, derived from actual API prompt tokens to the first
observable generated delta including reasoning (three scored runs each).
The API did not report cached-token counts. [Prefill scope](docs/results/2026-09-27-gdn-router-admitted.md).

Prior accepted L2 measurements and quality history are retained in [measurement history](docs/history.md) and the [original L2 results](docs/results/2026-09-26-l2.md).

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

## Optional switchless transport

For four directly cabled Sparks, see [the opt-in switchless ring configuration](docs/switchless.md). It requires a separately verified patched NCCL library and disables RoCEnante. The switched default and model settings remain unchanged. This source branch is not a new hardware or performance qualification.

## Reproduce

1. Prepare four ARM64 DGX Sparks with Docker GPU access, working SSH management paths, and a verified RoCE fabric. Configure the real interface names, addresses and GID on your fleet.
2. Follow [installation and image build](docs/install.md), including the pinned base image, RoCEnante and NCCL dependencies. Build on idle nodes.
3. Follow [weight preparation](docs/weights.md). Target and drafter directories must exist at matching paths on every node. Model weights are not distributed in this repository.
4. Copy `.env.example` to `.env` and edit the host, fabric and model paths. `profiles/current.env` enables GDN metadata fusion and router deduplication on the L2 base.
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
- **GDN metadata fusion and router deduplication:** fuses integer GDN metadata work and deduplicates supported router work, with bounded source checks enabled in the profile. The complete measured panel and its limitations are reported above.
- **Indexer correctness fixes:** carries padded seed stride, speculative ring and hybrid tail-slot mapping fixes together.
- **Faster loading and persistent JIT caches:** uses the slab loader and keeps compilation caches between boots. First-time compilation takes longer than a warm-cache boot.

See [runtime notes](docs/runtime.md) for enabled switches, source versions and implementation details.

## Further reading

- [Installation](docs/install.md): image build, NCCL, fabric settings and launch.
- [Weight preparation](docs/weights.md): target and drafter conversion, model licences.
- [Validation](docs/validation.md): benchmark commands, quality checks and known limits.
- [History](docs/history.md): earlier results and optimization experiments.
- [Credits](CREDITS.md): upstream authors, projects and component licences.
