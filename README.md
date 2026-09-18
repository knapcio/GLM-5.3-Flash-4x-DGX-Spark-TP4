# GLM-5.3-Flash NVFP4 on 4x NVIDIA DGX Spark (vLLM TP4, DFlash2 adaptive draft)

Serve [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B, 18B active) on four
DGX Spark boxes (GB10, SM121, 121.7 GiB unified memory each) behind a RoCE switch, with NVFP4
weights, Marlin MoE kernels, FP8 KV cache and the DFlash2 drafter with a per-request adaptive draft
length. One OpenAI-compatible endpoint with tool calling, reasoning and images.

Everything here was measured on the same fleet on 2026-09-18, single boot per variant, temperature 0,
streaming, decode tok/s = (completion tokens − 1) / (last − first token). Numbers are single runs;
repeat spread between two boots of the same config was 1–3 %.

## Results

Decode, one stream, `reasoning_effort` high / low:

| Config | code | prose | JSON | agent turn (6k system prompt + tools) |
|---|---|---|---|---|
| official FP8, Triton MoE, adaptive draft 3/7 (previous serving config) | 53 / 59 | 27 / 32 | 54 / 68 | 37 / 83 |
| NVFP4, Marlin, static draft 7 | 76 / 83 | 33 / 33 | 86 / 97 | 58 / 103 |
| NVFP4 (RedHatAI), Marlin, adaptive 3/7 | 78 / 83 | 38 / 37 | 85 / 89 | 47 / 106 |
| NVFP4 on SGLang TP4 (DFLASH k=7, bf16 KV, `docs/sglang/`) | 83 / 75 | 36 / 35 | 90 / 91 | 46 / 109 |
| **NVFP4, Marlin, adaptive draft 3/7 (this repo)** | **76 / 80** | **37 / 36** | **88 / 90** | **54 / 102** |

Per-stream decode at concurrency 1 / 2 / 3 (effort low, distinct prompts started together):

| Config | prose | code | JSON |
|---|---|---|---|
| official FP8, adaptive 3/7 | 29.5 / 23.1 / 18.7 | 60.9 / 37.6 / 31.2 | 65.5 / 48.0 / 37.8 |
| **NVFP4, adaptive 3/7** | **37.0 / 29.5 / 25.6** | **77.6 / 58.6 / 49.6** | **91.1 / 70.2 / 57.1** |
| NVFP4 on SGLang TP4 + RoCEnante all-reduce | 36.5 / 29.0 / 24.7 | 83.1 / 64.4 / 54.5 | 98.0 / 71.8 / 62.6 |
| Mia 1.6.0 EXL3 4bpw on **two** Sparks (TP2, adaptive-k, same prompts) | 22.4 / 17.6 / 15.6 | 38.8 / 26.9 / 24.2 | 51.1 / 33.3 / 26.8 |

Aggregate at c=3: prose 77, code 149, JSON 171 tok/s.

The two-Spark EXL3 row is [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
at `ca85576` (2026-09-18) with `GLM53_ADAPTIVE_K=ema`, `GLM53_DENSE_FP8=dense,kda`, `GLM53_COOP_GEOMETRY=1`, stock
loader, measured with this harness on the same two nodes; her README's prose number uses a different prompt.
Her quality gate is 74/75, the same as the official FP8 (EXL3 4bpw has KLD 0.025 vs NVFP4 0.061).

Quality gate (`bench/qeval.py`, 75 auto-scored checks: code executed against hidden asserts, JSON
schema, numeric answers, format constraints, prose degeneration; greedy, c=1): NVFP4 adaptive 72/75 on
three boots (LibertAI x2, RedHatAI), NVFP4 static 72/75, official FP8 74/75, SGLang NVFP4 68/75. The two
FP8-vs-NVFP4 differences are one math and one reasoning task; at 55 primary tasks that is p=0.06, and a
30-prompt harder set (`bench/hardset.py`: repo bug fixes, multi-step math, logic, facts, Polish and English
prose, tools, JSON) gave identical verifiable answers on every item, with FP8 once running into the 4096-token
reasoning cap. The gate catches degeneration, not subtle reasoning
loss; see *Fidelity* below.

What did not help (all measured, all rejected): `cudagraph_mode FULL_AND_PIECEWISE` (equal to
`FULL_DECODE_ONLY`), trimming the CUDA-graph capture list to `[1,2,4]` (−7 to −12 % at some
concurrencies: the DFlash families are token-count indexed), a host-side shard prewarm during weight
loading (the loader is CPU-bound at 3.9 s per shard, not disk-bound).

## Fidelity: FP8 vs NVFP4

NVFP4 is measurably further from BF16 than FP8 (KL divergence on malaiwah's panel: FP8 0.021, NVFP4
0.061) and NVIDIA's own NVFP4 Flash card shows benchmark parity (GPQA-Diamond 92.2 → 92.1,
Terminal-Bench 2.1 82.6 → 83.2, SciCode 56.2 → 57.7, MMMU-Pro 76.9 → 76.3). The checkpoint used here
(LibertAI, ModelOpt weight-only NVFP4) is a different calibration from NVIDIA's, so treat those
numbers as an analogy. The official FP8 checkpoint stays the reference; the same launcher serves it
with `MODEL_DIR` pointing at it and `MOE_BACKEND=triton`.

## Quick start

```bash
cp .env.example .env            # hosts, fabric, model paths
./start.sh serve                # workers first, then the head; ~14 min to /health 200
./start.sh status
./start.sh logs 0 80
./start.sh stop
```

Prerequisites on every node: the image, `MODEL_DIR` and `DRAFT_DIR` at the same path, Docker with GPU
support, `/dev/infiniband`, and an NCCL 2.30.7 build for the host if you set `NCCL_HOST_DIR` (the
image's NCCL also works over a switch). The endpoint binds to loopback on the head; put your own
tunnel or proxy in front of it.

```bash
curl http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": "What is 19 + 23?"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

`reasoning_effort` is `low`, `high` or `max` (the GLM template has no thinking-off switch; anything
else maps to `max`). Tool calls use the `glm47` parser, reasoning the `glm45` parser.

## What is in the box

```
start.sh                       serve / stop / status / logs for the 4-rank fleet, from .env
overlay/adaptive_draft_scheduler.py   per-request draft length in {K_LO, K_HI} from an acceptance EMA
overlay/adaptive_k_scheduler.py       jnardiello's adaptive verification length (base class)
overlay/sparse_attn_indexer_kpool.py  tonyd2wild's SM121 indexer patch
overlay/glm47_moe.py, abstract_parser.py   GLM parser fixes (literal tool delimiters, stop anchors)
overlay/E=288,N=512,...GB10...json     Triton MoE config for the FP8 lane
scripts/prewarm.py             page-cache prewarm sidecar (measured: no gain here, kept for NFS setups)
bench/bench_matrix.py          single-stream matrix (code / prose / JSON / agent, high and low)
bench/conc_bench.py            concurrency 1..3 per stream and aggregate
bench/qeval.py, qeval_tasks.py 75-check quality gate, paired McNemar comparison
docs/reference-plan-N2.json    the exact docker argv this README was measured with
```

## How the adaptive draft works

DFlash2 proposes `K_HI` (7) tokens per step. On prose the target accepts about 1.2 of them, on code
4.5 and on JSON 5.5, and every proposed token costs a verification row through all experts. The
scheduler keeps an EMA of the fraction of the first three draft positions accepted per request and
switches that request between `K_LO` (3) and `K_HI` (7) with hysteresis; the CUDA-graph families per
running-batch size are the `num_speculative_tokens_per_batch_size` table in `start.sh`. Net effect
against static k=7: prose +12 %, code and JSON unchanged.

## Knobs (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `MODEL_DIR` / `MOE_BACKEND` | NVFP4 / `marlin` | official FP8 checkpoint: `triton` (needs the GB10 MoE JSON, mounted by the launcher) |
| `K_HI` / `K_LO` | 7 / 3 | adaptive draft bounds; static k: set both equal |
| `KV_BYTES` | 12 GiB | FP8 KV per rank; 262k context at 4 sequences |
| `MAX_SEQS` | 4 | CUDA graphs are captured for the batch sizes this implies |
| `CAPTURE_SIZES` | `[1,2,4,6,8,12,16,18,24,32]` | keep: the DFlash families need the token-count sizes |
| `BATCHED_TOKENS` | 4096 | prefill chunk; a 53k prompt freezes other streams ~30 s at this size |
| `PREWARM` | 1 | shard prewarm sidecar; harmless, no measured gain on local NVMe |

## Known limits

- A long prefill (50k+) stalls the other decoding streams for its duration (chunked prefill shares
  the step budget). `--long-prefill-token-threshold` trades newcomer TTFT for decoder responsiveness;
  not enabled here.
- First batch-2 request after boot pays ~6 s of TTFT once (kernel JIT).
- Boot is ~14 min; ~6.5 min of it is 74k per-expert `copy_` calls from mmap-backed tensors at ~0.4 GB/s
  (profiled: iterator 7 s, copies 376 s, Marlin repack 8 s). Page-cache prewarm does not help; the same
  checkpoint loads in 8.5 min under SGLang. A pinned-buffer loader for vLLM is the open item.
- SGLang TP4 (`docs/sglang/`): works on the DSV4.1 image with the GB10 TileLang tile patch (block_I 32, 1
  stage, 128 threads); no adaptive draft for DFLASH there, speed equal to vLLM at the same k, quality gate
  68/75. Not the production path.
- The DFlash2 drafter is CC BY-NC-ND 4.0.

See [CREDITS.md](CREDITS.md): the image, the launch line, the adaptive verification scheduler and the
SM121 patches are other people's work; this repo adds the adaptive draft length, the four-node
profile, the bench harness and the quality gate.
