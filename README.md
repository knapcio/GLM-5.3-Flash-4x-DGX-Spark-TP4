# GLM-5.3-Flash on 4x NVIDIA DGX Spark (vLLM TP4, NVFP4 experts, lossless 8-bit dense, DFlash2)

Serve [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (321B, 18B active) on four
DGX Spark boxes (GB10, SM121, 128 GB unified memory each) behind a RoCE switch. One OpenAI-compatible
endpoint with tool calling, reasoning and images, 262k context, up to 32 concurrent sequences.

- **Weights:** NVIDIA's NVFP4 routed experts, untouched. The dense layers that checkpoint leaves in BF16
  (attention, KDA, shared experts, dense MLP) are stored on 8-bit grids they already fit
  (MXFP8 / block FP8), so they read at half the bytes with 0.25 % output error.
- **Decode:** DFlash2 speculative decoding. The target verifies a draft whose length is chosen per step from the
  drafter's own confidence (on the GPU at one request), a certified argmax skips most of the LM head, and
  sampling at T > 0 couples the draft to the target's noise so more of it is accepted. Marlin W4A16 MoE,
  FP8 KV cache, RoCEnante one-shot RDMA collectives on both ConnectX-7 rails.
- **Prefill:** sharded mHC, FlashKDA, Triton sparse MLA, dense 8-bit GEMMs on cuBLAS and fused routed-MoE
  kernels, about 3,100 tok/s cold at 8k-128k.

**Correctness update (2026-09-29, issue #2).** With prefix caching on, a cache hit could resume the KDA recurrent
state from 1,152 tokens too early, so the model misread the end of a shared prefix. This affected every profile
since 2026-09-19. It is fixed by `GLM_MAMBA_ALIGN_FIX=1` together with `BATCHED_TOKENS=6919`, with no speed change,
and gated by `bench/prefix_scan.py`. Existing installations need a coordinated restart with fresh container names.
[Cause, bisect and measurements](docs/results/2026-09-29-mamba-align-fix.md).

## Current results

Release 3edfbc9 (`profiles/current.env`), fresh clone, measured 2026-09-29 14:04-14:27 on four DGX Spark. Every
table below comes from the same boot and the same benchmark code; only the GPU clock differs.

### sparkDash, default GPU clocks (no cap)

The driver's default boost, as most people run these boxes: SM clock under load median 2489 MHz (p90 2535).

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **90.3** | 116.8 (58.9) | 169.5 (44.1) | 235.9 (30.8) | 331.5 (21.1) |
| code | 124.0 | 149.8 (80.1) | 215.1 (57.7) | 276.3 (36.7) | 348.3 (22.5) |
| structured | 168.8 | 270.2 (135.1) | 349.3 (95.5) | 463.5 (71.4) | 879.1 (55.0) |
| json | 110.7 | 191.9 (97.4) | 280.0 (73.9) | 345.6 (46.4) | 431.2 (29.6) |

**Prefill, cold, tok/s**

| 4k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 2768 | 3426 | 3470 | 3510 | 3462 |

### sparkDash, GPU clock capped at 2200 MHz

The cap this fleet runs with (`spark-clock-cap` service): it keeps the GPUs about 13 °C cooler (hottest GPU 72 °C
against 85 °C uncapped) for a small speed cost, mostly in prefill. SM clock under load median 2177 MHz.

**Decode, aggregate tok/s (per stream in brackets)**

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **85.0** | 126.6 (63.3) | 173.9 (44.6) | 231.1 (30.1) | 339.6 (22.6) |
| code | 119.8 | 170.8 (88.8) | 208.9 (56.9) | 267.5 (37.5) | 324.0 (22.0) |
| structured | 170.3 | 270.5 (135.3) | 367.0 (97.4) | 647.0 (80.9) | 722.3 (47.2) |
| json | 127.0 | 176.9 (90.5) | 244.7 (64.8) | 344.5 (46.7) | 440.3 (29.6) |

**Prefill, cold, tok/s**

| 4k | 16k | 32k | 64k | 128k |
|---:|---:|---:|---:|---:|
| 2563 | 3222 | 3330 | 3369 | 3322 |

### RigMark 1.0.0 decode screen, tok/s

| workload | default clocks | 2200 MHz cap |
|---|---:|---:|
| code | 113.8 | 111.5 |
| prose | 62.1 | 60.6 |
| structured | 153.2 | 154.6 |

**How these were measured.** sparkDash 1.8.8 DecodeBench: 256 new tokens, temperature 0, thinking off, idle
endpoint, two warm-up prose c1 runs discarded; c1 is the median of 3 runs (prose c1 at 2200 MHz: median of 6, three
in the sweep 87.1 / 76.7 / 75.9 and three in a recheck 84.7 / 87.0 / 85.3), c4 the median of 2, c2 / c8 / c16 one
run; every run checked for 256 tokens per stream and no reasoning output. Greedy output is not bit-reproducible run to
run on this stack, so prose c1 moves by several tok/s between runs (one prompt; the tokens per step follow how much
of the draft is accepted), and single-run cells differ by more than the gap between the two clocks; prefill (+4-8 %)
and RigMark code / prose (+2 %) are the clock differences above noise. Prefill: sparkDash prefill bench, cold
(salted prompts), after a 4k warm-up, median of runs 2-3. RigMark: decode only, effort low, median of 4 passes (not a
conformant receipt). Raw output and the per-node clock and temperature samples:
[docs/results/2026-09-29-clocks.md](docs/results/2026-09-29-clocks.md).

- **Quality:** KL divergence 0.0285 over 6618 teacher-forced positions against a BF16-attention reference
  (`bench/kld_probe.py`, `bench/compare_kld_strict.py`), and 0.0098 over 98,500 positions of four 16.5k-40k-token
  prompts against the 2026-09-28 release (`bench/final_bench.py kldlong`), so the prefill path is covered too; these
  two are the primary gates. `bench/qeval.py` (75 auto-scored checks) with the fixed number extractor: 75 and 74 on
  this release; qeval moves by 2-3 points run to run on every stack here (the 2026-09-28-based stack read 73 / 75 / 72
  in one boot), so it is run three times against a same-window reference, not as a single-run floor. T > 0 scan
  (`bench/final_bench.py tscan`, 35 outputs at T 1.0 / 0.6 and mixed batches): 0 garbled outputs.
- **Public benchmarks** (greedy, thinking off, 8 requests at a time, 2026-09-29 on the d80f4fd configuration, whose
  decode path is the same as this release apart from the shared draft-length state): GSM8K test, first 250
  questions, 246 / 250 = 98.4 %; HumanEval pass@1 157 / 164 = 95.7 % (programs run in a container without network).
  Teacher-forced NLL over eight public 3,000-token texts (English and Polish Wikipedia, two public-domain literary works,
  CPython and Go source): 0.531 nats/token.
- **Boot:** 3.6 min to `/health` 200 with warm caches, 9.4 min on the first boot of a fresh tree (every JIT cold).

Earlier measurements and the comparisons with previous releases are in [docs/history.md](docs/history.md) and the
[2026-09-18 first release](docs/history-2026-09-18.md); what changed when is in [CHANGELOG.md](CHANGELOG.md).

## What is in the stack

Each row was measured against the stack without it, in the same boot where possible
(`overlay/glm_ab.py` switches kernels between two CUDA-graph sets in one boot; a result is
promoted only when its confidence interval clears zero and an identical-arm control does not).
"Exact" means the output is bit-identical to the path it replaces; the others are judged by the KL gates above.

**Added on 2026-09-29** (each is one switch in `profiles/current.env`)

| Piece | Where | Effect | Credit |
|---|---|---|---|
| Draft-shape truncation | `GLM_DRAFT_TRUNC`, `overlay/glm_draft_trunc.py` | verifies only the draft prefix that pays for its rows; prose c1 +5.8 %, c4 +4.5 %; exact | ours |
| Device-side verify-shape selection (not in gate 0003) | `DEVSELECT=1`, `overlay/glm_devselect.py` | at one request the GPU picks the verify length itself (one parent graph over the captured 2..8-row graphs), so the host no longer waits for the draft: step −0.45 ms, prose +1.9 %, RigMark T > 0 prose +3.9 %; buffer-identical to host truncation, and device and host share one λ, so both pick the same length | ours |
| Truncation cost refit for batch > 1 (not in gate 0003) | `TRUNC_COST=c4fit` | c4 prose +3.0 % (fleet A/B); c1 unchanged by construction | ours |
| Gumbel-coupled drafting (T > 0) | `GLM_GUMBEL_COUPLED`, `overlay/glm_gumbel_coupled.py` | the drafter reuses the target's Gumbel noise: +3.2 % at T = 1; committed tokens equal plain sampling with the same seed | method: Jim Routh, llama.cpp-lab PR #26 |
| Certified target head | `GLM_CERT_HEAD`, `overlay/glm_cert_head.py`, `cert_math.py` | an 8-bit screen of the LM head with a proven error bound picks the argmax; the full BF16 row is read only when the bound cannot decide; `min_tokens` requests too; exact | ours |
| Decode kernel set | `GLM_GATE_GEMV`, `GLM_EARLY_PLAN`, `GLM_MARLIN_TUNE_ON`, `GLM_DENSE_FAST`, `GLM_L2_PREFETCH_MLA/_DRAFT`, `GLM_MHC_BF16W`, `GLM_DRAFT_CONV_FUSED`, `GLM_ROCE_PROXY_CPUS` | with the certified head: −1.25 ms per c1 step (gate GEMV, early plan, cert head), then −1.90 ms more for the rest together (in-boot A/B, 10 rounds) | Marlin (IST-DASLab, Neural Magic, vLLM); rest ours |
| Prefill package | `GLM_PREFILL_SHARD(_PAD)`, `GLM_FLASHKDA_PREFILL`, `GLM_KDA_CONV_SPLIT`, `GLM_TRITON_MLA_PREFILL`, `GLM_DENSE_FAST_PREFILL` | cold prefill +28 % (4k) to +43 % (128k) | FlashKDA (MoonshotAI, MIT; Matt Mastracci's fp32-state branch); ideas: mmastrac (conv split, sparse MLA), Jacopo Nardiello and FujitsuPolycom (mHC sharding) |
| Routed-MoE prefill kernels (not in gate 0003) | `PF3_ARM=samemath`, `overlay/glm_pf3_*.py` | MoE sum + shared add fused (exact, self-checked before it arms), smaller down tile, gate_up + SiLU in one kernel: prefill +3-4 % | vLLM MoE Marlin template; epilogue idea: mmastrac |
| Large row gathers over NCCL (not in gate 0003) | `GATHER_ROUTE=1`, `overlay/glm_roce_gather_route.py` | the mHC prefill-shard row gathers (> 4 MiB per rank) take NCCL, −20 % per gather; exact | ours, on the RoCEnante shim |
| Corrected L2 prefetch tables (not in gate 0003) | `L2PF_V2=1`, `overlay/glm_l2pf_v2.py` | the all-reduce L2 windows warm the BF16 mHC weights the kernel actually reads instead of the unused FP32 copy; c1 −0.29 ms (n.s.), tok/s c1 +0.45 %, c4 +2.6 % (in-boot A/B); exact | ours |

**From the 2026-09-28 release**

| Piece | Where | Effect | Credit |
|---|---|---|---|
| NVFP4 experts on Marlin (W4A16) | image, `MOE_BACKEND=marlin` | code +40 %, JSON +50 % vs the official FP8 checkpoint, same quality gate | NVIDIA, LibertAI, Red Hat AI checkpoints; alexellis's launch line |
| Dense layers on 8-bit grids | `scripts/build_lossless8.sh`, `glm_quant_mix.py`, `overlay/qmix_patch.py` | step −7.5 ms, output error 0.25 % | tonyd2wild found the 18 GiB left in BF16 |
| DFlash2 drafter, block-FP8 linears | `scripts/drafter_fp8.py` | draft graph 4.24 → 3.10 ms, acceptance unchanged | incoai (drafter) |
| Batch-uniform draft length | `overlay/glm_levers_sched.py`, `profiles/levers_policy.json` | prose c4 136.7 → 148.2 (no eager mixed-k steps) | builds on jnardiello's adaptive-k scheduler and Reederey87's verify-only idea |
| RoCEnante one-shot all-reduce / all-gather | `Dockerfile.roce`, `roce/` | decode collectives over RDMA, both rails | Luke Alonso, Jason Cook (local-inference-lab/b12x#295, vllm#597); tonyd2wild's v11 port; rhys101 |
| Replicated-linear TP split, FP8 draft head, one-gather top-k | `overlay/glm_ds_*.py` | dense −0.48 ms, bit-exact where marked | ported from our DeepSeek-V4.1 stack |
| KDA verify stash, no-copy reads, fused flags | `overlay/glm_kda_stash*.py`, `kda_stash.py` | −0.90 ms (no-copy) | ours |
| L2 prefetch of the next weights | `overlay/glm_l2_prefetch.py` | −0.5 ms | ours (from the DeepSeek-V4.1 stack) |
| Router GEMM dedup | `overlay/glm_router_dedup.py` | −0.5 ms | vllm#55736 (JaredforReal), MiaAI-Lab issue #271 |
| GDN metadata fast path | `overlay/glm_gdn_metadata_fast.py` | exact, host side | ours, on vLLM's builder |
| Vocab-parallel target argmax, greedy path also for `min_tokens` | `overlay/glm_target_argmax.py` | no full-vocab gather at verify; −0.59 ms on `min_tokens` requests | vLLM's draft-side argmax from vllm#34049 (zixi-qi) |
| Padded-vocab clamp in both samplers | `overlay/gumbel.py`, `rejection_sampler_utils.py` | correctness | vllm#50843 (alexbi29) |
| DSA indexer kpool tail fixes | `GLM_KPOOL_FIX=1`, `overlay/glm5next_*.py`, `mla_indexer.py`, `mamba_hybrid.py` | correctness past the first KV block | vllm#57477 (JaredforReal), #58454 (mmastrac, on ivanium's #55219), #53906 (ZJY0516), root cause vcruz305 |
| Prefill cadence + end drain | `overlay/glm_prefill_sched.py`, `glm_prefill_hooks.py` | decoders under a 32k prefill 1.3 → 7.3 tok/s; short newcomer at c4 −22 %; TTFT −8 %; step unchanged | jnardiello (E27, E27b/c, E29), FujitsuPolycom (SparkRing non-DP throttle) |
| Prefix cache for the DFlash2 draft group | `overlay/kv_cache_coordinator.py` | repeated 20k prompt 8.1 s → 0.63 s TTFT | tonyd2wild |
| Fast weight loader, persistent FlashInfer JIT cache | `overlay/glm_fast_load.py`, `start.sh` | boot 271 → 128 s | vllm#58726 (Willian-Zhang) |

Tried and rejected, with numbers: confidence-based verify cut (prose −5 to −9 %), Marlin tile M=32
(+1.2 ms), fused mHC kernels (not bit-exact), a CUDA graph for the DFlash context KV (+0.05 ms),
W4A4 / MXFP4 experts for prefill (1.1-1.3x on MoE at 16-21 % MoE output error), a lower truncation
row cost of 1.6 ms (not measured on the fleet; 2.0 is), a 9216-row prefill chunk budget
(`BATCHED_TOKENS=9223`: prefill −9 to −34 %).

## Build

1. **Image**, on every node (no CUDA compile, a minute or two):
   ```bash
   docker build --platform linux/arm64 -f Dockerfile.roce -t glm53-roce:v11-b58f34ea .
   ```
   It adds the b12x RoCEnante subset (Apache-2.0, `roce/b12x/LICENSE`, pinned in
   `roce/b12x/PROVENANCE.json`) to tonyd2wild's `ghcr.io/tonyd2wild/vllm-glm53-flash` (vLLM `487ecf187`).
2. **Weights**, on every node at the same path (CPU only):
   ```bash
   hf download nvidia/GLM-5.3-Flash-NVFP4 --local-dir ~/models/nvidia/GLM-5.3-Flash-NVFP4
   scripts/build_lossless8.sh ~/models/nvidia/GLM-5.3-Flash-NVFP4 ~/models/glm-quant-mix
   hf download incoai/GLM-5.3-Flash-DFlash2 --local-dir ~/models/incoai/GLM-5.3-Flash-DFlash2
   python3 scripts/drafter_fp8.py ~/models/incoai/GLM-5.3-Flash-DFlash2 ~/models/incoai/GLM-5.3-Flash-DFlash2-fp8blk
   ```
   The drafter is CC BY-NC-ND 4.0: keep the re-encoded copy local.
3. **NCCL** 2.30.7 built for the host (optional, `NCCL_HOST_DIR`; the image's NCCL also works).

## Quick start

```bash
cp .env.example .env            # hosts, fabric, image and weight paths; sources profiles/current.env
./start.sh serve                # workers first, then the head
./start.sh status               # until health 200
./start.sh logs 0 80
./start.sh stop                 # stops, never removes; a new deployment needs a fresh CTN
```

The endpoint binds to loopback on the head; put your own tunnel or proxy in front of it.

```bash
curl http://127.0.0.1:8093/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "GLM-5.3-Flash-FP8", "messages": [{"role": "user", "content": "What is 19 + 23?"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}}'
```

`reasoning_effort` is `low`, `high` or `max`. Tool calls use the `glm47` parser, reasoning the `glm45` parser.

Every switch is a line in `profiles/current.env`; removing a line turns that piece off. The five newest pieces
are knobs you can set on the command line, e.g. `DEVSELECT=0 PF3_ARM=off ./start.sh serve` (`DEVSELECT`,
`TRUNC_COST`, `PF3_ARM`, `GATHER_ROUTE`, `L2PF_V2`; see the comments in the profile). The overlay's CUDA
extensions (dense 8-bit kernels, FlashKDA, routed-MoE prefill kernels) are JIT-built on each node on first use and
cached under the overlay directory; a first boot of a fresh tree takes longer. The `serve` step first checks that
no existing container already uses the target name or overlay path.

## Without a switch (community-contributed)

Four Sparks cabled as a ring, with no RoCE switch, can run this stack with `TRANSPORT=switchless`
(`.env.switchless.example`, [docs/switchless.md](docs/switchless.md)). It needs a patched NCCL 2.30.7 that you build
and pin by SHA256, and it turns RoCEnante off by default, so decode is slower than the numbers above; with the
hardware-forwarded opposite-node paths installed (sparkring) and `SWITCHLESS_ROCE_RING=1` the same decode-size
collectives run on `b12x.comm.roce_ring` and decode returns to the switched step times. Contributed by
[@othexmr](https://github.com/othexmr) ([PR #1](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/pull/1)).
Upstream's own fleet is switched, so its checks cover the switched launch's byte-identity and a render of the
switchless one; the opt-in `SWITCHLESS_ROCE_RING=1` variant has since been booted and measured on four Sparks
([docs/switchless.md](docs/switchless.md#rocenante-on-the-ring-switchless_roce_ring1)). Please
[open an issue](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/issues) if it breaks for you
([what to include](docs/switchless.md#reporting-a-problem)).

## Benchmarks and gates

```
bench/qeval.py              75-check quality gate (run / compare)
bench/kld_probe.py          teacher-forced top-20 logprobs from the live endpoint; compare_kld_strict.py
bench/accept_probe.py       step ms and accepted tokens per step at c1 (the stable speed number)
bench/conc_bench.py         concurrency sweep with 32 distinct prompts per type
bench/prefill_bench.py      cold prefill tok/s by prompt length
bench/final_bench.py        long-prompt KL panel, T > 0 garble scan, sparkDash prefill, RigMark decode screen
overlay/glm_ab.py, scripts/ab_inboot_glm.py   in-boot A/B of kernel switches with an A/A control
```

`tests/run_cpu_tests.sh` runs every CPU test (31 suites; `GLM_IMAGE_SRC` points the source-drift checks at an
extracted copy of the image's vLLM, or run it inside the image). The `*_gpu.py` tests need a free GPU, so run them
with the model stopped.

## Known limits

- Prefill is still behind recipes that use W4A4 experts; this stack keeps the NVFP4 / 8-bit weights.
- Device-side verify-shape selection works at one request only; batches use host truncation.
- GLM greedy output is not bit-reproducible across runs on this stack (batch-dependent kernels), so
  exactness of a change is checked per kernel, not by comparing text.
- The DFlash2 drafter is CC BY-NC-ND 4.0.

## More documentation

- [docs/install.md](docs/install.md): host prerequisites, NCCL, image build, launch and preflight rules
- [docs/weights.md](docs/weights.md): preparing the lossless8 target and the FP8 drafter
- [docs/runtime.md](docs/runtime.md), [docs/validation.md](docs/validation.md): runtime switches and the validation scope
- [docs/switchless.md](docs/switchless.md): four Sparks without a switch (community-contributed, untested here)
- [docs/results/](docs/results/): raw result files per release

See [CREDITS.md](CREDITS.md) for authors and pull requests, and [NOTICE](NOTICE) with [LICENSES/](LICENSES/) for licence boundaries.
