# Validation

The current recipe is LVKP-S-L2. Quantization remains NVFP4 for routed experts
and the existing `lossless8` profile for 8-bit non-expert weights. `lossless8`
is a profile name, not a mathematically lossless conversion. A component speed result does not
qualify a serving change.

## Decode performance

Use sparkDash DecodeBench with its original generic prose/code/structured/JSON
prompts and thinking disabled. `bench/final_sparkdash.py` runs the same matrix as
`glm_prodbench.sh`, with additional prose samples to reach the required counts:

1. Two discarded prose c1 warmups.
2. Three prose c1 runs, code c1, prose c2/c4/c8/c16, code c4/c16,
   structured c1/c16, and JSON c1/c16.
3. Two additional prose c1 and two additional prose c4 runs.

This produces 20 jobs, 18 scored. Report prose c1 median of 5 and prose c4 aggregate
decode throughput median of 3, both in tokens/s. `meanDecodeTps` is per stream;
`aggregateDecodeTps` is the concurrency-wide metric. Do not confuse either with
request throughput including prefill or a component CUDA-event benchmark.

```bash
python3 -B bench/final_sparkdash.py YOUR_NEW_LABEL \
  --base http://localhost:5555/api/sparks/spark-01/llm \
  --model GLM-5.3-Flash-FP8 --out YOUR_NEW_RESULT.json
```

Set `--base` to your dashboard endpoint. The historical API model alias in this
command does not describe the weight format: this recipe retains NVFP4+8-bit.
The collector checks the dashboard's `spark-01` logical identifier and model port
8093. Match these configured identifiers before running. It creates a new output
exclusively, refuses an already-active benchmark, polls each returned job ID,
rejects reused/stale IDs, and never substitutes the dashboard's last result.
There is no automatic retry or cancellation after timeout; inspect the active
job before starting more work. Use an idle service with no competing requests.

A completed result requires `COMPLETE_VALID_MEASUREMENT`, `complete=true`, and
20 unique valid jobs. Every stream must complete 256 tokens, report 255 decode
tokens and zero reasoning chunks, with no stream errors or early-EOS substitute.
Partial output and partial summary fields are diagnostic evidence only.

CPU API-contract tests (no model/GPU):

```bash
cd bench
python3 -B -S -m unittest -v test_final_sparkdash_cpu
```

Credit: the original `glm_prodbench.sh` matrix and sparkDash DecodeBench
contributors. This wrapper adds strict collection/validation and enough repeated
prose samples; it does not replace the dashboard's measurement implementation.

## Quality and changes

Run the bundled qeval panel against your own idle endpoint, preserving both full
75-task runs and their output files. Run from a new output directory because the
producer writes `qeval-LABEL.json` to its current working directory:

```bash
RECIPE_ROOT="$PWD"
mkdir quality-NEW_LABEL
cd quality-NEW_LABEL
python3 -B "$RECIPE_ROOT/bench/qeval.py" run NEW_LABEL-c1 \
  --url http://127.0.0.1:8093/v1/chat/completions --concurrency 1
python3 -B "$RECIPE_ROOT/bench/qeval.py" run NEW_LABEL-c4 \
  --url http://127.0.0.1:8093/v1/chat/completions --concurrency 4
```

The producer uses the historical model alias `GLM-5.3-Flash-FP8`. Do not use
`--only` or `--limit` for the full quality gate, and do not replace failed tasks
with selective retries. Retain all task outputs, score and truncation counts.

Require qeval >=72/75 at c1 and c4, teacher-forced KLD around 0.03, and retained
long-context sanity evidence. KLD must contain all 17 expected calibration items
and 6618 teacher-forced positions with matching per-item lengths and prompt
identity; greedy text equality or a truncated zip is not a substitute. The
reported top-20 folded-tail estimate is not full-vocabulary KL divergence.

The exact retained KLD panel includes private operational material and is not
distributed here. The published hashes identify the original evidence; they do
not make that specific numerical KLD result independently reproducible from this
repository alone. A new public panel needs a separately recorded reference from
the chosen reference configuration and identical prompts/tokenization for each
candidate. Changing or redacting the panel creates a different measurement and
cannot reproduce the published 0.0288366 value. Do not substitute generated
continuation matching for teacher-forced log-probability comparisons.

For your own calibration panel, use `bench/kld_probe.py` unchanged. Supply a JSON
array of objects with `id`, `kind` and `text`; keep the panel, model/tokenizer,
prompt formatting and top-K fixed across the two collections. The reference
endpoint must be a separately qualified reference configuration. These commands
do not build or qualify that reference for you:

```bash
test ! -e reference.json && test ! -e candidate.json
python3 -B "$RECIPE_ROOT/bench/kld_probe.py" collect \
  --url http://REFERENCE_HOST:8093 --model GLM-5.3-Flash-FP8 \
  --texts YOUR_PANEL.json --out reference.json --k 20 --gen 256
python3 -B "$RECIPE_ROOT/bench/kld_probe.py" collect \
  --url http://127.0.0.1:8093 --model GLM-5.3-Flash-FP8 \
  --texts YOUR_PANEL.json --out candidate.json --k 20 --gen 256
python3 -B "$RECIPE_ROOT/bench/compare_kld_strict.py" \
  --texts YOUR_PANEL.json --reference reference.json --candidate candidate.json
```

The original collector can continue with greedy output if prompt log probabilities
are unavailable. The strict comparison refuses that fallback, incomplete item
grids, unequal per-item lengths, invalid probability support and nonfinite values.
It emits artifact hashes and the actual lengths/position count. Confirm the
reference's full lengths against your tokenizer and preserve collection commands
and calibration hashes: equal lengths alone cannot prove the two files came from
the same complete prompts. `STRUCTURAL_PASS` is not a numerical quality gate.
No KLD value is claimed here for an operator-supplied panel.

Bundled source SHA256: qeval `a83ccf00919153cae3acfaee49390be57d2f447b30e12ba4b4cbb9ac364a694e`,
qeval tasks `9072efdda46856028189c8f21a7c83f09ef75fba7e70da8c93d5781493a3bb9e`,
KLD probe `75a2adbb16500fbafe86b01c19680ab64e9282522490e608fdc34559a7a5005e`.

For a new optimization, first use one-boot balanced A/B plus a duplicate baseline
A/A control, at least 20–30 retained rounds. Keep the numerical estimator and
coverage gates declared in advance. The owner gate is a savings confidence
interval wholly above 0.2 ms with A/A within ±0.2 ms at both c1 and c4. Qualify
arithmetic on tensors (bit-exact or ULP), not greedy response text. Run the full
sparkDash/quality gates only for the selected combined stack.

The dated result page distinguishes the fresh throughput rerun from retained
quality qualification. An unchanged recipe can vary between sessions; a difference
from an earlier run alone does not establish a causal speedup or regression.
