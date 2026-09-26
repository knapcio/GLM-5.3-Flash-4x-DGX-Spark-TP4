# Measurement and recipe history

[Current recipe and results](../README.md) · [Archived 2026-09-18 README](history-2026-09-18.md)

Numbers below retain their original benchmark, sample count and configuration. Different benchmark prompts, reasoning settings, quantizations and boot conditions are not interchangeable baselines.

## 2026-09-26: accepted LVKP-S-L2

The accepted profile adds read-only L2 prefetch to LVKP-S. In the 72-round in-boot qualification, `inboot-target-start-period-cuda-events` measured savings of **0.610 ms at c1** (95% CI [0.548, 0.671]) and **0.727 ms at c4** ([0.617, 0.827]). The duplicate-baseline A/A intervals were [-0.147, 0.029] ms and [-0.127, 0.064] ms, inside the predefined ±0.2 ms band.

The following full sparkDash qualification measured prose c1 **70.84 tok/s** (median of 5: 72.02, 68.86, 70.84, 66.90, 72.21) and c4 **151.41 tok/s aggregate** (median of 3: 146.62, 153.85, 151.41). The preceding accepted LVKP-S sparkDash medians were 68.35 and 149.55 tok/s respectively. Those measurements imply +3.64% and +1.24% for that comparison; they are not the result of the later evening screen.

L2 quality qualification: qeval 75/75 at c1 and c4; KLD 0.0288366 over 17 items / 6,618 teacher-forced positions against the retained BF16-attention reference. A separate approximately 9.6k-token registry sanity test passed 32/32 lookups at each concurrency. No full-context quality claim follows from that test.

The original accepted containers were restored after the evening experiments. A fresh full sparkDash measurement is recorded in [release validation](validation.md), separately from the promotion measurement above.

## 2026-09-26 evening: GDN and router experiments, not promoted

Two independent in-boot experiments found positive target-step contrasts for GDN metadata optimization combined with router deduplication. Neither qualified for deployment under the predefined duplicate-baseline control gate.

The final 32-round run used four active arms in one diagnostic boot: baseline L2, duplicate L2, GDN+router, and GDN. Common-shape coverage passed the 95% gate (minimum 96.078431%); structural and tensor-check-counter reviews passed. The benchmark was `inboot-target-start-period-cuda-events`, not sparkDash:

| Contrast | c1 saving, ms (95% CI) | c4 saving, ms (95% CI) |
|---|---:|---:|
| GDN+router vs L2 | 0.813 [0.663, 0.959] | 0.663 [0.375, 0.909] |
| GDN vs L2 | 0.563 [0.427, 0.695] | 0.121 [-0.437, 0.518] |
| Duplicate baseline A/A | 0.073 [-0.105, 0.233] | -0.151 [-0.381, 0.063] |

Both A/A intervals extend outside ±0.2 ms. The result is **insufficient control precision for promotion**, not evidence that the positive candidate contrast is zero, nor a demonstrated quality regression. The thresholds were retained. No candidate sparkDash, qeval or KLD release run followed, and none of these experimental gains is included in the README throughput.

An earlier 36-round experiment also failed the A/A gate. Its six-arm coverage failure involved CPU-placement arms; a separately documented primary-arm analysis retained the same primary estimator but still did not qualify. The two runs were not pooled, and no failed blocks were removed to manufacture a pass.

## 2026-09-26: bounded dense / MoE investigations

The dense experiment passed 4,496 finite and byte-comparison checks. Eliminating original-workspace zeroing saved approximately 0.4–0.7 microseconds per component call, but the full integration candidate did not beat the installed stock path: one tested cell was slower and the other was within noise. This does not rule out different tiling or a different integration.

The MoE workspace-reuse experiment passed 600 finite and byte-comparison checks. It measured only workspace-zero reuse with fixed alignment and automatic tiles, not a replacement MoE implementation. Some component contrasts were positive, but duplicate graph controls showed bias in other cells. No model throughput claim or production change followed.

A draft-gather qualification tool failed before any tensor test because its staged path was shallower than the generator expected. The repaired tool passed 9 CPU tests and independent source review; GPU qualification and performance remain unmeasured. It is not part of this release.

## Earlier recipes

The [2026-09-18 snapshot](history-2026-09-18.md) preserves the previous FP8 / NVFP4 / EXL3 / SGLang comparisons, high/low reasoning tests, task-time experiments and original boot measurements. They use different settings from the current thinking-off sparkDash table. Historical statements about which ideas were still open or which switches were defaults apply only to that snapshot.
