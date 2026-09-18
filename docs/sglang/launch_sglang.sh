#!/bin/zsh
# GLM-5.3-Flash on SGLang TP4 across four Sparks, using the DSV4.1 production image.
# usage: ./launch_sglang.sh <variant> [start|stop|status|logs]
#   variants: S1 = LibertAI NVFP4, DFLASH k7 adaptive, NCCL only
#             S2 = S1 + RoCEnante all-reduce (SGLANG_ROCE_ALLREDUCE=1)
#             S3 = official FP8 checkpoint, otherwise S2
#             S4 = S2 with fixed k5 (no adaptive)
set -u
V=${1:-S1}; CMD=${2:-start}
HERE=${0:A:h}
IMAGE=dsv41-4x-spark:dsv41-f80c91a4b-roce-v3-prefetch-20260918
HOSTS=(Spark_01 Spark_02 Spark_03 Spark_04)
IPS=(10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4)
NAME=glm53-sgl-$V
MODEL_NVFP4=/home/knapcio/models/GLM-5.3-Flash-NVFP4
MODEL_FP8=/home/knapcio/models/zai-org/GLM-5.3-Flash
DRAFT=/home/knapcio/models/incoai/GLM-5.3-Flash-DFlash2
STATE=/home/knapcio/glm53-sglang-tp4-20260918
TILELANG=/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
LOADUTILS=/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py

MODEL=$MODEL_NVFP4; ROCE=0; SPEC="--speculative-num-draft-tokens 7 --speculative-adaptive"
case $V in
  S1) ;;
  S2) ROCE=1 ;;
  S3) ROCE=1; MODEL=$MODEL_FP8 ;;
  S4) ROCE=1; SPEC="--speculative-num-draft-tokens 5" ;;
  *) echo "unknown variant $V"; exit 2 ;;
esac

stage() {
  for h in $HOSTS; do
    ssh -o BatchMode=yes $h "mkdir -p $STATE/state-$V" && scp -q -o BatchMode=yes $HERE/tilelang_kernel_gb10.py $h:$STATE/ || return 1
  done
}

run_rank() {
  local h=$1 r=$2
  local ip=${IPS[$((r+1))]}
  local hostflag="--host 127.0.0.1"
  local roce_env=""
  if [[ $ROCE == 1 ]]; then
    roce_env="-e SGLANG_ROCE_ALLREDUCE=1 -e SGLANG_ROCE_MAX_SIZE=2097152 -e B12X_ROCE_HCA=rocep1s0f0,roceP2p1s0f0 -e B12X_ROCE_CACHE_DIR=/state/b12x-roce -e B12X_COMPILE_CACHE_DIR=/state/b12x-compile"
  fi
  ssh -o BatchMode=yes $h "docker run -d --name $NAME-r$r --restart=no --gpus all --network host --ipc host \
    --shm-size 32g --ulimit memlock=-1:-1 --ulimit stack=67108864 --device /dev/infiniband:/dev/infiniband \
    --memory 112g --memory-swap 112g \
    -v $MODEL:/models/glm:ro -v $DRAFT:/draft:ro -v $STATE/state-$V:/state -v /home/knapcio/.cache:/root/.cache \
    -v $STATE/tilelang_kernel_gb10.py:${TILELANG}:ro \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
    -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e NCCL_PROTO=^LL128 -e NCCL_MAX_NCHANNELS=8 \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a -e SGLANG_RUST_BUILD_MODE=never \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False $roce_env \
    --entrypoint bash $IMAGE -c 'sed -i \"s/UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480/UNBALANCED_MODEL_LOADING_TIMEOUT_S = 3600/\" $LOADUTILS && exec python3 -m sglang.launch_server \
      --model-path /models/glm --served-model-name GLM-5.3-Flash-FP8 --trust-remote-code --load-format safetensors \
      --tp 4 --nnodes 4 --node-rank $r --dist-init-addr 10.100.96.2:20000 --dist-timeout 3600 \
      --attention-backend dsa --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --linear-attn-backend triton \
      --moe-runner-backend flashinfer_cutlass --disable-shared-experts-fusion \
      --kv-cache-dtype bfloat16 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 40 \
      --mem-fraction-static 0.80 --context-length 262144 --max-total-tokens 524288 --max-running-requests 4 --cuda-graph-max-bs-decode 4 \
      --chunked-prefill-size 4096 --disable-prefill-cuda-graph \
      --speculative-algorithm DFLASH --speculative-draft-model-path /draft $SPEC --speculative-draft-attention-backend flashinfer \
      --language-only --reasoning-parser glm45 --tool-call-parser glm47 --chat-template /models/glm/chat_template.jinja \
      --enable-metrics --enable-cache-report --sleep-on-idle --watchdog-timeout 1800 --random-seed 0 \
      $hostflag --port 8093'"
}

case $CMD in
  start)
    stage || { echo stage failed; exit 1; }
    for r in 3 2 1 0; do run_rank ${HOSTS[$((r+1))]} $r || exit 1; done
    echo "launched $NAME on 4 ranks; poll: ./launch_sglang.sh $V status" ;;
  stop)
    for h in $HOSTS; do ssh -o BatchMode=yes $h "docker rm -f $NAME-r0 $NAME-r1 $NAME-r2 $NAME-r3 2>/dev/null; echo $h stopped"; done ;;
  status)
    for h in $HOSTS; do ssh -o BatchMode=yes $h "echo \$(hostname) \$(docker ps -a --filter name=$NAME --format '{{.Names}} {{.Status}}') avail=\$(free -g | awk 'NR==2{print \$7}')G"; done
    ssh -o BatchMode=yes Spark_01 "curl -s -m 3 -o /dev/null -w 'health %{http_code}\n' http://127.0.0.1:8093/health" ;;
  logs)
    ssh -o BatchMode=yes ${3:-Spark_01} "docker logs --tail ${4:-40} \$(docker ps -a --filter name=$NAME --format '{{.Names}}' | head -1) 2>&1 | cut -c1-220" ;;
esac
