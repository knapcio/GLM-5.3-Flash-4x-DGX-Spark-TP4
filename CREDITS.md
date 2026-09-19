# Credits

This profile stands on other people's work; the list is in dependency order.

- **tonyd2wild** — the ARM64 vLLM image `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (vLLM `487ecf187` with the SM121 DFlash2 and NoPE sparse-attention patches), `overlay/sparse_attn_indexer_kpool.py`, the prefix-cache repair for the DFlash2 draft group
  (`overlay/patch_prefix_cache_draft_group.py`, issue #13/#18 of his 2x recipe), the GB10 Triton MoE config and the boot-time cache flusher. Recipes: [GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark), [GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark).
- **Jacopo Nardiello (jnardiello)** — the four-Spark FP8 launch line and `overlay/adaptive_k_scheduler.py` (per-request adaptive verification length), [GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless).
- **Alex Ellis (alexellis)** — the NVFP4 + Marlin four-Spark launch line and the RigMark methodology, [glm-5.3-flash-4x-dgx-spark-switchless](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless).
- **Reederey87** — the verify-only adaptive draft idea that `overlay/adaptive_draft_scheduler.py` extends to the draft length itself, [glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark).
- **incoai** — the DFlash2 drafter [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (CC BY-NC-ND 4.0: check the licence for your use).
- **LibertAI** and **Red Hat AI** — the NVFP4 checkpoints [LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4), [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4); **malaiwah** for the KLD fidelity panel of the GLM-5.3-Flash quants.
- **local-inference-lab** — the GLM parser fixes carried in `overlay/glm47_moe.py` and `overlay/abstract_parser.py` ([vllm#639](https://github.com/local-inference-lab/vllm/pull/639), [#640](https://github.com/local-inference-lab/vllm/pull/640)).
- **Mia (MiaAI-Lab)** — the two-Spark EXL3 recipe whose launcher and benchmark discipline this profile imitates, and sparkDash.
- **Z.AI** — GLM-5.3-Flash.
