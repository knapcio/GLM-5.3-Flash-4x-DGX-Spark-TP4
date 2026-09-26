# LVKP-S-L2 supplemental decode cells: 26 September 2026

Eight missing table cells were measured on the unchanged accepted L2 service
with **sparkDash DecodeBench**, after the separate prefill measurement had
finished. Each is one observation. These runs do not replace or recompute
the [earlier prose c1/c4 medians](2026-09-26-l2.md).

| Prompt | Concurrency | Per-stream decode, tokens/s | Aggregate decode, tokens/s |
|---|---:|---:|---:|
| Code | 2 | 81.53 | 160.34 |
| Code | 8 | 31.77 | 231.27 |
| Structured | 2 | 80.32 | 140.48 |
| Structured | 4 | 52.98 | 196.68 |
| Structured | 8 | 32.61 | 242.16 |
| Json | 2 | 61.73 | 115.55 |
| Json | 4 | 51.44 | 193.62 |
| Json | 8 | 41.90 | 303.75 |

All 8 fixed jobs and 38 streams completed without errors. Every stream
reported 256 completion tokens, 255 decode tokens and zero reasoning chunks.
The exact reviewed collector was used, with only its fixed cell list changed;
POST/GET validation, per-stream checks, summaries and no-retry behavior
were unchanged. Each reported value was independently checked against raw.

The [sanitized evidence](2026-09-26-l2-supplement.json) retains the sample
values, individual stream rates and source/raw hashes. No private addresses,
paths, dashboard history or generated text are included. There is no new
quality claim or causal optimization comparison from these observations.

Raw SHA256: `a652a155b10719e5485d6d4bba8499d0a464a2d8e5b6cca365a4311f67664504`.
Collector SHA256: `8665bfea1fed60e162fb042d55b214e8a3e0b1d951f7644d561a540428a4bf7b`.

Credit: glm_prodbench.sh and sparkDash DecodeBench contributors.
