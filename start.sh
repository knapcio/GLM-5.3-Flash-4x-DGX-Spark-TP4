#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 on 4x DGX Spark, vLLM TP4 with DFlash2 adaptive draft.
# usage: ./start.sh serve|stop|status|logs [rank]   (config in .env, copy .env.example)
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE=${ENV_FILE:-.env}
[[ -f $ENV_FILE ]] || { echo "copy .env.example to .env first (or set ENV_FILE)"; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
CMD=${1:-status}
read -r -a HOSTS <<<"$HOSTS"; read -r -a IPS <<<"$IPS"
OVERLAY_REMOTE=${OVERLAY_REMOTE:-\$HOME/glm53-flash-4x-spark}
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
MOE_JSON='E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json'

# A host named "local" runs on this machine (lets the script run on the head itself); SSH_OPTS adds e.g. -i KEY.
rssh() { if [[ ${DRY:-0} == 1 ]]; then echo "[$1] ${*:2}"; return; fi; if [[ $1 == local ]]; then bash -c "${*:2}"; else ssh -o BatchMode=yes -o ConnectTimeout=10 ${SSH_OPTS:-} "$1" "${@:2}"; fi; }

sync_overlay() {
  if [[ ${DRY:-0} == 1 ]]; then echo "[dry-run] rsync overlay scripts bench to configured hosts"; return 0; fi
  for h in "${HOSTS[@]}"; do
    rssh "$h" "mkdir -p $OVERLAY_REMOTE/cache"
    if [[ $h == local ]]; then
      [[ "$(cd "$OVERLAY_REMOTE" && pwd -P)" == "$(pwd -P)" ]] || rsync -a --exclude .git --exclude .env overlay scripts bench "$OVERLAY_REMOTE/" || return 1
    else
      rsync -a --exclude .git --exclude .env -e "ssh -o BatchMode=yes ${SSH_OPTS:-}" overlay scripts bench "$h:$OVERLAY_REMOTE/" || return 1
    fi
  done
}

# Compact physical memory on every node before the engine allocates (a fragmented head made rank-0 kernels
# 20-80 % slower on the DS4.1 fleet). Needs a NOPASSWD rule for the script; silently skipped otherwise.
compact_mem() {
  for h in "${HOSTS[@]}"; do rssh "$h" "sudo -n /usr/local/sbin/spark-compact-mem.sh >/dev/null 2>&1 || true"; done
}

spec_json() {
  if [[ -n "${SPEC_JSON:-}" ]]; then echo "$SPEC_JSON"; return; fi   # full override, e.g. another drafter method
  # Graph families per running-batch size (draft length seen by the target for 1 / 2 / 3-4 requests);
  # the per-request draft length is chosen inside these families by overlay/adaptive_draft_scheduler.py.
  local table=${SPEC_TABLE:-"[[1, 1, ${K_HI}], [2, 2, 5], [3, ${MAX_SEQS}, 3]]"}
  echo "{\"method\": \"${SPEC_METHOD:-dflash}\", \"model\": \"/draft\", \"num_speculative_tokens\": ${K_HI}, \"kv_cache_dtype\": \"auto\", \"num_speculative_tokens_per_batch_size\": ${table}, \"rejection_sample_method\": \"standard\"}"
}

run_rank() {
  local r=$1 h=${HOSTS[$r]} ip=${IPS[$r]}
  local extra=""; [[ $r != 0 ]] && extra="--headless"
  local nccl_mount=""; [[ -n "${NCCL_HOST_DIR:-}" ]] && nccl_mount="-v $NCCL_HOST_DIR:/opt/nccl:ro -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7"
  # tonyd2wild's two-line glm5next patch: build attention with the checkpoint's quant config (needed for any
  # checkpoint that quantises attention, e.g. scripts/quantize_nonexpert_mxfp8.py output). Off by default.
  local patch_mount=""; [[ ${GLM5NEXT_PATCH:-0} == 1 ]] && patch_mount="-v $OVERLAY_REMOTE/overlay/glm5next_kda.py:$VLLM_PKG/models/glm5next/nvidia/kda.py:ro -v $OVERLAY_REMOTE/overlay/glm5next_model.py:$VLLM_PKG/models/glm5next/nvidia/model.py:ro"
  # GLM_KPOOL_FIX=1 (diagnostics/glm-kpool-audit + glm-kernels-20260926): the DSA indexer kpool tail fixes, only
  # safe together: vllm#57477 padded seed stride (JaredforReal), vllm#58454 spec-decode ring (mmastrac, on ivanium's
  # #55219), and positions for the hybrid tail slot map (#53906 ZJY0516; root cause vcruz305) in a persistent buffer.
  [[ ${GLM_KPOOL_FIX:-0} == 1 ]] && patch_mount="$patch_mount -e GLM_KPOOL_FIX=1 \
    -v $OVERLAY_REMOTE/overlay/glm5next_kpool_compress.py:$VLLM_PKG/models/glm5next/nvidia/ops/kpool_compress.py:ro \
    -v $OVERLAY_REMOTE/overlay/glm5next_attention.py:$VLLM_PKG/models/glm5next/nvidia/attention.py:ro \
    -v $OVERLAY_REMOTE/overlay/mla_indexer.py:$VLLM_PKG/v1/attention/backends/mla/indexer.py:ro \
    -v $OVERLAY_REMOTE/overlay/mamba_hybrid.py:$VLLM_PKG/v1/worker/gpu/model_states/mamba_hybrid.py:ro"
  local load=""; [[ -n "${LOAD_FORMAT:-}" ]] && load="--load-format $LOAD_FORMAT"
  # FI_CACHE_PERSIST (default 1; diagnostics/glm-inboot): FlashInfer keeps its JIT builds under $HOME/.cache/flashinfer,
  # i.e. /root inside the container (XDG_CACHE_HOME is not read), so every boot rebuilt three modules with nvcc: top-k
  # in profile_run (~92 s), batch MLA at capture (~7 s), sampling in the warm-up (~38 s). FLASHINFER_WORKSPACE_BASE
  # moves the cache to the persistent per-node $OVERLAY_REMOTE/cache/fi (keyed by FlashInfer version and arch).
  # Measured on lv-final: boot 271 s (cold cache) -> 128 s (warm). FI_CACHE_PERSIST=0 restores the old behaviour.
  local xfi=""; [[ ${FI_CACHE_PERSIST:-1} == 1 ]] && xfi="-e FLASHINFER_WORKSPACE_BASE=/cache/fi"
  local xenv=""; for kv in ${EXTRA_ENV:-}; do xenv="$xenv -e $kv"; done
  local sched="--scheduler-cls ${SCHEDULER_CLS:-adaptive_draft_scheduler.AdaptiveDraftScheduler}"; [[ ${SCHEDULER_CLS:-} == none ]] && sched=""
  rssh "$h" "docker run -d --restart no --name ${CTN}-r$r --gpus all --network host --ipc host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
    --memory ${CTN_MEM:-112g} --memory-swap ${CTN_MEM:-112g} --entrypoint vllm \
    -v $MODEL_DIR:/model:ro -v $DRAFT_DIR:/draft:ro -v $OVERLAY_REMOTE:/overlay:ro -v $OVERLAY_REMOTE/cache:/cache $nccl_mount $patch_mount $xenv $xfi \
    -v $OVERLAY_REMOTE/overlay/sparse_attn_indexer_kpool.py:$VLLM_PKG/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
    -v $OVERLAY_REMOTE/overlay/glm47_moe.py:$VLLM_PKG/parser/glm47_moe.py:ro \
    -v $OVERLAY_REMOTE/overlay/abstract_parser.py:$VLLM_PKG/parser/abstract_parser.py:ro \
    -v $OVERLAY_REMOTE/overlay/kv_cache_coordinator.py:$VLLM_PKG/v1/core/kv_cache_coordinator.py:ro \
    -v \"$OVERLAY_REMOTE/overlay/$MOE_JSON:$VLLM_PKG/model_executor/layers/fused_moe/configs/$MOE_JSON:ro\" \
    -e VLLM_HOST_IP=$ip -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e HF_HOME=/cache/hf -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm -e PYTHONPATH=/overlay/overlay \
    -e TRITON_CACHE_DIR=/cache/triton -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e CUDA_CACHE_PATH=/cache/nv \
    -e VLLM_ADAPTIVE_K_HI=$K_HI -e VLLM_ADAPTIVE_K_LO=$K_LO -e VLLM_ADAPTIVE_K_MODE=per-request -e VLLM_ADAPTIVE_K_SEED=1.0 \
    -e VLLM_ADAPTIVE_K_DOWN=0.42 -e VLLM_ADAPTIVE_K_UP=0.58 -e VLLM_ADAPTIVE_K_ALPHA=0.15 -e VLLM_ADAPTIVE_K_SIGNAL=pos \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
    -e NCCL_SOCKET_IFNAME==$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE -e NCCL_IB_HCA==$IB_HCA \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
    $IMAGE serve /model --served-model-name $SERVED_NAME --dtype bfloat16 \
    --tensor-parallel-size 4 --nnodes 4 --node-rank $r --master-addr ${IPS[0]} --master-port ${MASTER_PORT:-29669} --distributed-executor-backend mp \
    --max-model-len $MAX_MODEL_LEN --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization ${GPU_UTIL:-0.85} \
    --max-num-seqs $MAX_SEQS --max-num-batched-tokens ${BATCHED_TOKENS:-4096} --block-size 2304 --moe-backend $MOE_BACKEND \
    --enable-prefix-caching --enable-chunked-prefill --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template /model/chat_template.jinja \
    --default-chat-template-kwargs '{\"reasoning_effort\":\"${DEFAULT_EFFORT:-high}\"}' \
    --host ${HOST_BIND:-127.0.0.1} --port ${PORT:-8093} --disable-custom-all-reduce \
    --speculative-config '$(spec_json)' $sched \
    --compilation-config '{\"mode\": 0, \"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": ${CAPTURE_SIZES}, \"max_cudagraph_capture_size\": ${CAPTURE_MAX}}' \
    --limit-mm-per-prompt '{\"image\":${MM_IMAGES:-16},\"video\":0}' --mm-processor-cache-gb ${MM_CACHE_GB:-4} --mm-processor-kwargs '{\"max_pixels\":6422528,\"max_image_tokens\":4096}' $load ${EXTRA_ARGS:-} $extra"
  if [[ ${PREWARM:-1} == 1 ]]; then
    rssh "$h" "nohup python3 $OVERLAY_REMOTE/scripts/prewarm.py $MODEL_DIR ${CTN}-r$r 3 4 > $OVERLAY_REMOTE/cache/prewarm-r$r.log 2>&1 &"
  fi
}

case $CMD in
  serve)
    # Inspect every preserved container before any source synchronization.
    # Ship code as an argument, so preflight works before scripts are installed.
    printf -v preflight_code '%q' "$(cat scripts/preflight_runtime.py)"
    for r in 0 1 2 3; do
      rssh "${HOSTS[$r]}" "python3 -c $preflight_code --container ${CTN}-r$r --overlay $OVERLAY_REMOTE"
    done
    sync_overlay
    [[ ${COMPACT_MEM:-1} == 1 ]] && compact_mem
    for r in 3 2 1 0; do run_rank $r; done
    # Boot request warm-up (scripts/boot_warm.py): once /health is 200, one short chat and BOOT_WARM_COUNT (2) cold
    # prefills of ~BOOT_WARM_TOKENS (16384) tokens, so the first user's long prompt does not pay the first-long-
    # prefill cost (diagnostics/glm-inboot). Log: cache/boot_warm.log, last line "boot-warm done". BOOT_WARM=0 = off.
    if [[ ${BOOT_WARM:-1} == 1 ]]; then
      rssh "${HOSTS[0]}" "cd $OVERLAY_REMOTE && BOOT_WARM_TOKENS=${BOOT_WARM_TOKENS:-16384} BOOT_WARM_COUNT=${BOOT_WARM_COUNT:-2} nohup python3 scripts/boot_warm.py --base http://127.0.0.1:${PORT:-8093} --model $SERVED_NAME > cache/boot_warm.log 2>&1 & echo \$! > cache/boot_warm.pid"
    fi
    echo "launched ${CTN}-r0..3; ./start.sh status until /health is 200 (cold-cache boots can take longer)";;
  stop)
    for r in 0 1 2 3; do
      rssh "${HOSTS[$r]}" "python3 $OVERLAY_REMOTE/scripts/stop_preserving.py --root $OVERLAY_REMOTE --container ${CTN}-r$r"
    done;;
  status)
    for r in 0 1 2 3; do rssh "${HOSTS[$r]}" "echo \$(hostname) \$(docker ps -a --filter name=${CTN}-r$r --format '{{.Status}}') avail=\$(free -g | awk 'NR==2{print \$7}')G"; done
    rssh "${HOSTS[0]}" "curl -s -m 3 -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:${PORT:-8093}/health";;
  logs)
    r=${2:-0}; rssh "${HOSTS[$r]}" "docker logs --tail ${3:-40} ${CTN}-r$r 2>&1 | cut -c1-200";;
  *) echo "usage: $0 serve|stop|status|logs [rank] [lines]"; exit 2;;
esac
