#!/usr/bin/env bash
# 4-rank RoCEnante test for the GLM image: one container per Spark, real vLLM TP4 groups,
# RoCE vs NCCL correctness, CUDA-graph replay and latency (roce/test_4rank.py).
#
# NOT for a serving fleet. It refuses to start while ~/fleet_busy exists on the head, or while
# any container runs on any of the four hosts (FORCE=1 skips only the container check).
# Footprint per rank: ~160 MiB pinned host memory (RoCE slots) + <100 MiB of GPU tensors.
#
# usage: roce/test_4rank.sh [image]      (default IMAGE=glm53-roce:v11-b58f34ea, built on every node)
# env:   HOSTS, IPS (rank order = head first, as in .env), FABRIC_IFACE, IB_HCA, NCCL_HOST_DIR,
#        B12X_ROCE_HCA (default both rails), PORT (29671), REPLAYS (300), NO_BENCH=1
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${1:-${IMAGE:-glm53-roce:v11-b58f34ea}}
read -r -a H <<<"${HOSTS:-Spark_01 Spark_02 Spark_03 Spark_04}"
read -r -a IP <<<"${IPS:-10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4}"
FABRIC_IFACE=${FABRIC_IFACE:-enp1s0f0np0}
IB_HCA=${IB_HCA:-rocep1s0f0}
NCCL_HOST_DIR=${NCCL_HOST_DIR-/home/knapcio/nccl-2.30.7}
ROCE_HCA=${B12X_ROCE_HCA:-rocep1s0f0,roceP2p1s0f0}
PORT=${PORT:-29671}
REMOTE=${REMOTE:-glm-roce-test}          # under the remote $HOME
CTN=glm-roce-4rank
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-roce/results/4rank-$STAMP}
mkdir -p "$OUT"
rssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

# -- guards ---------------------------------------------------------------------------------
if rssh "${H[0]}" 'test -e ~/fleet_busy'; then echo "REFUSE: ~/fleet_busy exists on ${H[0]}"; exit 3; fi
for h in "${H[@]}"; do
  running=$(rssh "$h" "docker ps --format '{{.Names}}'" | grep -v "^${CTN}-" || true)
  if [[ -n "$running" && ${FORCE:-0} != 1 ]]; then
    echo "REFUSE: containers running on $h: $(echo "$running" | tr '\n' ' ')(stop the fleet first)"; exit 3
  fi
  rssh "$h" "docker image inspect $IMAGE >/dev/null" || { echo "REFUSE: $IMAGE missing on $h"; exit 3; }
done

# -- stage the test file (the image carries the shim; the test itself is copied fresh) ----------
for h in "${H[@]}"; do
  rssh "$h" "mkdir -p ~/$REMOTE && docker rm -f ${CTN}-r0 ${CTN}-r1 ${CTN}-r2 ${CTN}-r3 >/dev/null 2>&1 || true"
  scp -q -o BatchMode=yes roce/test_4rank.py "$h:$REMOTE/test_4rank.py"
done

nccl_args=""
if [[ -n "$NCCL_HOST_DIR" ]]; then
  nccl_args="-v $NCCL_HOST_DIR:/opt/nccl:ro -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7 -e VLLM_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7"
fi
bench_arg=""; [[ ${NO_BENCH:-0} == 1 ]] && bench_arg="--no-bench"

# -- launch rank 3..0 (same NCCL environment as start.sh) --------------------------------------
for r in 3 2 1 0; do
  h=${H[$r]}
  rssh "$h" "docker run -d --name ${CTN}-r$r --gpus all --network host --ipc host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory 24g --memory-swap 24g --entrypoint python3 \
    -v \$HOME/$REMOTE:/work $nccl_args \
    -e GLM_ROCE_ALLREDUCE=1 -e B12X_ROCE_HCA=$ROCE_HCA -e B12X_ROCE_SPIN_LIMIT=20000000 \
    -e B12X_COMPILE_CACHE_DIR=/work/b12x-compile -e XDG_CACHE_HOME=/work/cache \
    -e VLLM_HOST_IP=${IP[$r]} -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_SOCKET_IFNAME==$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE -e NCCL_IB_HCA==$IB_HCA \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
    $IMAGE -u /work/test_4rank.py --rank $r --world 4 --master ${IP[0]} --port $PORT --replays ${REPLAYS:-300} $bench_arg" >/dev/null
  echo "launched rank $r on $h"
done

# -- wait (15 min cap), collect, clean up -----------------------------------------------------------
deadline=$((SECONDS + 900))
for r in 0 1 2 3; do
  h=${H[$r]}
  while [[ $(rssh "$h" "docker inspect -f '{{.State.Running}}' ${CTN}-r$r 2>/dev/null" || echo false) == true ]]; do
    if (( SECONDS > deadline )); then echo "TIMEOUT waiting for rank $r"; break; fi
    sleep 5
  done
done
fail=0
for r in 0 1 2 3; do
  h=${H[$r]}
  rssh "$h" "docker logs ${CTN}-r$r 2>&1" > "$OUT/rank$r.log" || true
  code=$(rssh "$h" "docker inspect -f '{{.State.ExitCode}}' ${CTN}-r$r 2>/dev/null" || echo 99)
  rssh "$h" "docker rm -f ${CTN}-r$r >/dev/null 2>&1" || true
  res=$(grep '^RESULT ' "$OUT/rank$r.log" | tail -1 | cut -c8- || true)
  echo "$res" > "$OUT/rank$r.json"
  ok=$(python3 -c "import json,sys; d=json.loads(sys.argv[1] or '{}'); print(d.get('ok'))" "$res" 2>/dev/null || echo None)
  echo "rank $r ($h): exit=$code ok=$ok"
  [[ $code == 0 && $ok == True ]] || fail=1
done
grep -h "GLM_ROCE_READY\|GLM_ROCE_ROUTE" "$OUT"/rank*.log | cut -c1-220 || true
python3 - "$OUT/rank0.json" <<'EOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
c = d.get("checks", {})
print("rank0 runtime:", d.get("runtime"), "end:", d.get("runtime_end"))
print("max |RoCE-NCCL| over routed sizes:", c.get("max_abs_vs_nccl_routed"))
print("graph:", c.get("graph"))
lat = c.get("latency_us") or {}
for tokens, row in lat.get("all_reduce", {}).items():
    print(f"  all_reduce T={tokens:>4} {row['bytes']:>8} B  roce {row['roce']:>7} us  nccl {row['nccl']:>7} us")
for rows, row in lat.get("all_gather", {}).items():
    print(f"  all_gather rows={rows:>4} shard {row['shard_bytes']:>9} B  roce {row['roce']:>7} us  nccl {row['nccl']:>7} us")
EOF
echo "logs: $OUT"
[[ $fail == 0 ]] && echo "4RANK PASS" || { echo "4RANK FAIL"; exit 1; }
