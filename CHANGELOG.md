# Changelog

## 2026-10-01 (RoCEnante on the switchless ring)
- Optional `SWITCHLESS_ROCE_RING=1`: decode-size all-reduce/all-gather run on the vendored
  `b12x.comm.roce_ring` over the hardware-forwarded opposite-node paths (sparkring `f16b5f4` bundle;
  mesh install, `B12X_ROCE_PEER_HCA_MAPS` and the 256 KiB caps come from the DeepSeek-V4.1 recipe's
  `ring_mesh` plan). The launcher validates maps and sizes before any remote action; the default
  switchless launch stays byte-identical, the switched one untouched. `roce/test_ring_smoke.py` is a
  4-rank exactness/transport check without vLLM or NCCL. On four Sparks: decode step at or below the
  switched fleet (prose 33.8 vs 39.0 ms), structured c1 167.8 tok/s, prefill 16k-128k -7 %.
  [docs/switchless.md](docs/switchless.md).

## 2026-09-29 (switchless ring, community PR #1)
- Optional `TRANSPORT=switchless` for four Sparks cabled as a ring without a switch (patched NCCL pinned by
  SHA256, RoCEnante off), contributed by @othexmr. Not tested on our switched fleet. The default switched launch
  is byte-identical; the container preflight now reads only names and mounts from `docker inspect`.
  [docs/switchless.md](docs/switchless.md).

## 2026-09-29 (hotfix, 3edfbc9)
- KDA state checkpoints now align to the 2304-token block (`GLM_MAMBA_ALIGN_FIX=1`, `BATCHED_TOKENS=6919`). With prefix
  caching on, a cache hit could resume the KDA recurrent state 1,152 tokens early (issue #2); affected every profile
  since 2026-09-19. No speed change. [Details](docs/results/2026-09-29-mamba-align-fix.md).
- New release gate `bench/prefix_scan.py`: c8 shared-long-prefix scan at T > 0, cold vs warm.

## 2026-09-29 (release, d80f4fd / 982e258 / 8f49b5c)
- Draft truncation with the verify length chosen on the GPU at one request (shared lambda with the host policy),
  c4 cost table refit, certified LM head with min_tokens, Gumbel-coupled drafting at T > 0.
- Prefill: routed-MoE prefill kernels (fused SiLU epilogue, down tile, exact sum+shared add), large dim-0 gathers
  over NCCL, prefill package (sharded mHC, FlashKDA, Triton sparse MLA, dense 8-bit prefill GEMMs).
- Corrected L2 prefetch tables; qeval number-extractor fix and a three-run quality rule.
- [Results](docs/results/2026-09-29-release.md); comparison with 2026-09-28 in [docs/history.md](docs/history.md).

## 2026-09-28 (release, adb55e3 / 1f5b9eb)
- Lossless 8-bit dense layers, RoCEnante image, decode kernels, LeversScheduler; the repository becomes the serving
  production stack.

## 2026-09-18 (first release)
- NVFP4 experts on Marlin, adaptive draft length, kpool patch; [archived README](docs/history-2026-09-18.md).
