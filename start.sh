#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 on 4x DGX Spark, vLLM TP4 with DFlash2 adaptive draft.
# usage: ./start.sh serve|stop|status|logs [rank]   (config in .env, copy .env.example)
set -euo pipefail
cd "$(dirname "$0")"
[[ -f .env ]] || { echo "copy .env.example to .env first"; exit 1; }
# shellcheck disable=SC1091
source .env
CMD=${1:-status}
read -r -a HOSTS <<<"$HOSTS"; read -r -a IPS <<<"$IPS"
OVERLAY_REMOTE=${OVERLAY_REMOTE:-\$HOME/glm53-flash-4x-spark}
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
MOE_JSON='E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json'

rssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

sync_overlay() {
  for h in "${HOSTS[@]}"; do
    rssh "$h" "mkdir -p $OVERLAY_REMOTE/cache"
    rsync -a --exclude .git --exclude .env -e "ssh -o BatchMode=yes" overlay scripts bench "$h:$OVERLAY_REMOTE/" || return 1
  done
}

spec_json() {
  # Graph families per running-batch size (draft length seen by the target for 1 / 2 / 3-4 requests);
  # the per-request draft length is chosen inside these families by overlay/adaptive_draft_scheduler.py.
  echo "{\"method\": \"dflash\", \"model\": \"/draft\", \"num_speculative_tokens\": ${K_HI}, \"kv_cache_dtype\": \"auto\", \"num_speculative_tokens_per_batch_size\": [[1, 1, ${K_HI}], [2, 2, 5], [3, ${MAX_SEQS}, 3]], \"rejection_sample_method\": \"standard\"}"
}

run_rank() {
  local r=$1 h=${HOSTS[$r]} ip=${IPS[$r]}
  local extra=""; [[ $r != 0 ]] && extra="--headless"
  local nccl_mount=""; [[ -n "${NCCL_HOST_DIR:-}" ]] && nccl_mount="-v $NCCL_HOST_DIR:/opt/nccl:ro -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7"
  rssh "$h" "docker run -d --restart no --name ${CTN}-r$r --gpus all --network host --ipc host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory ${CTN_MEM:-112g} --memory-swap ${CTN_MEM:-112g} --entrypoint vllm \
    -v $MODEL_DIR:/model:ro -v $DRAFT_DIR:/draft:ro -v $OVERLAY_REMOTE:/overlay:ro -v $OVERLAY_REMOTE/cache:/cache $nccl_mount \
    -v $OVERLAY_REMOTE/overlay/sparse_attn_indexer_kpool.py:$VLLM_PKG/model_executor/layers/sparse_attn_indexer_kpool.py:ro \
    -v $OVERLAY_REMOTE/overlay/glm47_moe.py:$VLLM_PKG/parser/glm47_moe.py:ro \
    -v $OVERLAY_REMOTE/overlay/abstract_parser.py:$VLLM_PKG/parser/abstract_parser.py:ro \
    -v \"$OVERLAY_REMOTE/overlay/$MOE_JSON:$VLLM_PKG/model_executor/layers/fused_moe/configs/$MOE_JSON:ro\" \
    -e VLLM_HOST_IP=$ip -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
    -e HF_HOME=/cache/hf -e XDG_CACHE_HOME=/cache -e VLLM_CACHE_ROOT=/cache/vllm -e PYTHONPATH=/overlay/overlay \
    -e VLLM_ADAPTIVE_K_HI=$K_HI -e VLLM_ADAPTIVE_K_LO=$K_LO -e VLLM_ADAPTIVE_K_MODE=per-request -e VLLM_ADAPTIVE_K_SEED=1.0 \
    -e VLLM_ADAPTIVE_K_DOWN=0.42 -e VLLM_ADAPTIVE_K_UP=0.58 -e VLLM_ADAPTIVE_K_ALPHA=0.15 -e VLLM_ADAPTIVE_K_SIGNAL=pos \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
    -e NCCL_SOCKET_IFNAME==$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE -e NCCL_IB_HCA==$IB_HCA \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
    $IMAGE serve /model --served-model-name $SERVED_NAME --dtype bfloat16 \
    --tensor-parallel-size 4 --nnodes 4 --node-rank $r --master-addr ${IPS[0]} --master-port ${MASTER_PORT:-29669} --distributed-executor-backend mp \
    --max-model-len $MAX_MODEL_LEN --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes $KV_BYTES --gpu-memory-utilization 0.85 \
    --max-num-seqs $MAX_SEQS --max-num-batched-tokens ${BATCHED_TOKENS:-4096} --block-size 2304 --moe-backend $MOE_BACKEND \
    --enable-prefix-caching --enable-chunked-prefill --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template /model/chat_template.jinja \
    --default-chat-template-kwargs '{\"reasoning_effort\":\"${DEFAULT_EFFORT:-high}\"}' \
    --host ${HOST_BIND:-127.0.0.1} --port ${PORT:-8093} --disable-custom-all-reduce \
    --speculative-config '$(spec_json)' --scheduler-cls adaptive_draft_scheduler.AdaptiveDraftScheduler \
    --compilation-config '{\"mode\": 0, \"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": ${CAPTURE_SIZES}, \"max_cudagraph_capture_size\": ${CAPTURE_MAX}}' \
    --limit-mm-per-prompt '{\"image\":${MM_IMAGES:-16},\"video\":0}' --mm-processor-cache-gb ${MM_CACHE_GB:-4} --mm-processor-kwargs '{\"max_pixels\":6422528,\"max_image_tokens\":4096}' $extra"
  if [[ ${PREWARM:-1} == 1 ]]; then
    rssh "$h" "nohup python3 $OVERLAY_REMOTE/scripts/prewarm.py $MODEL_DIR ${CTN}-r$r 3 4 > $OVERLAY_REMOTE/cache/prewarm-r$r.log 2>&1 &"
  fi
}

case $CMD in
  serve)
    sync_overlay
    for r in 3 2 1 0; do run_rank $r; done
    echo "launched ${CTN}-r0..3; ./start.sh status until /health is 200 (about 10-16 min)";;
  stop)
    for r in 0 1 2 3; do rssh "${HOSTS[$r]}" "docker rm -f ${CTN}-r$r >/dev/null 2>&1; pkill -f 'prewarm.py .* ${CTN}-r$r' 2>/dev/null; echo ${HOSTS[$r]} stopped"; done;;
  status)
    for r in 0 1 2 3; do rssh "${HOSTS[$r]}" "echo \$(hostname) \$(docker ps -a --filter name=${CTN}-r$r --format '{{.Status}}') avail=\$(free -g | awk 'NR==2{print \$7}')G"; done
    rssh "${HOSTS[0]}" "curl -s -m 3 -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:${PORT:-8093}/health";;
  logs)
    r=${2:-0}; rssh "${HOSTS[$r]}" "docker logs --tail ${3:-40} ${CTN}-r$r 2>&1 | cut -c1-200";;
  *) echo "usage: $0 serve|stop|status|logs [rank] [lines]"; exit 2;;
esac
