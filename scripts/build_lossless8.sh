#!/usr/bin/env bash
# Offline CPU conversion only. Fresh outputs; no deletion or overwrite.
set -euo pipefail
base=${1:?usage: build_lossless8.sh BASE_NVIDIA_NVFP4 OUTPUT_ROOT}
root=${2:?output root required}
tools=$(cd "$(dirname "$0")" && pwd)
[[ -d $base && ! -e $root ]] || { echo 'base must exist and output root must be new' >&2; exit 2; }
mkdir "$root"
python3 "$tools/glm_quant_mix.py" plan --src "$base" --arm lossless8 --out "$root/plan-lossless8.json"
python3 "$tools/glm_quant_mix.py" quant --src "$base" --plan "$root/plan-lossless8.json" --arm lossless8 --out "$root/overlay" --skip-bf16 --threads 4
python3 "$tools/glm_quant_mix.py" assemble --base "$base" --overlay "$root/overlay" --dst "$root/lossless8"
