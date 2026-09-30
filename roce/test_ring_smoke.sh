#!/usr/bin/env bash
# 4-rank smoke test for b12x.comm.roce_ring (roce/test_ring_smoke.py): one container
# per Spark, gloo only, proves the QPs to the opposite ranks traverse the mesh.
#
# Requires the mesh active on all four hosts (docs/switchless.md), the peer maps
# from its plan (B12X_ROCE_PEER_HCA_MAPS) and the roce image. Refuses to start
# while any container runs on the hosts (FORCE=1 skips the check).
#
# usage: roce/test_ring_smoke.sh [image]
# env:   HOSTS, IPS (rank order, head first), IMAGE, IMAGE_TAG, B12X_ROCE_HCA,
#        B12X_ROCE_PEER_HCA_MAPS, GLOO_IFACE (bootstrap interface), PORT (29672)
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${1:-${IMAGE:-glm53-roce:v12-ring-20261001}}
read -r -a H <<<"${HOSTS:-192.168.31.57 192.168.31.19 192.168.31.240 192.168.31.112}"
read -r -a IP <<<"${IPS:-192.168.31.57 192.168.31.19 192.168.31.240 192.168.31.112}"
GLOO_IFACE=${GLOO_IFACE:-wlP9s9}
ROCE_HCA=${B12X_ROCE_HCA:-rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1}
PORT=${PORT:-29672}
REMOTE=${REMOTE:-glm-roce-smoke}
CTN=glm-ring-smoke
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=${OUT:-roce/results/ring-smoke-$STAMP}
mkdir -p "$OUT"
rssh() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "${@:2}"; }

[[ -n ${B12X_ROCE_PEER_HCA_MAPS:-} ]] || { echo 'set B12X_ROCE_PEER_HCA_MAPS (the mesh plan env.txt value)'; exit 2; }

for h in "${H[@]}"; do
  running=$(rssh "$h" "docker ps --format '{{.Names}}'" | grep -v "^${CTN}-" || true)
  if [[ -n "$running" && ${FORCE:-0} != 1 ]]; then
    echo "REFUSE: containers running on $h: $(echo "$running" | tr '\n' ' ')(stop the fleet first)"; exit 3
  fi
  rssh "$h" "docker image inspect $IMAGE >/dev/null" || { echo "REFUSE: $IMAGE missing on $h"; exit 3; }
  rssh "$h" "mkdir -p ~/$REMOTE && docker rm -f ${CTN}-r0 >/dev/null 2>&1 || true"
  scp -q -o BatchMode=yes roce/test_ring_smoke.py "$h:$REMOTE/test_ring_smoke.py"
done

for r in 3 2 1 0; do
  h=${H[$r]}
  rssh "$h" "docker run -d --name ${CTN}-r$r --gpus all --network host --ipc host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory 24g --memory-swap 24g --entrypoint python3 \
    -v \$HOME/$REMOTE:/work \
    -e GLOO_SOCKET_IFNAME=$GLOO_IFACE -e VLLM_HOST_IP=${IP[$r]} \
    -e GLM_ROCE_RING=1 -e GLM_ROCE_ALLREDUCE=1 -e B12X_ROCE_HCA=$ROCE_HCA \
    -e B12X_ROCE_PEER_HCA_MAPS='$B12X_ROCE_PEER_HCA_MAPS' \
    -e GLM_ROCE_MAX_SIZE=${GLM_ROCE_MAX_SIZE:-262144} -e GLM_ROCE_GATHER_MAX_SIZE=${GLM_ROCE_GATHER_MAX_SIZE:-262144} \
    -e B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0 -e B12X_ROCE_SPIN_LIMIT=20000000 \
    -e XDG_CACHE_HOME=/work/cache -e B12X_ROCE_CACHE_DIR=/work/cache \
    $IMAGE -u /work/test_ring_smoke.py --rank $r --world 4 --master ${IP[0]} --port $PORT" >/dev/null
  echo "launched rank $r on $h"
done

deadline=$((SECONDS + 600))
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
python3 - "$OUT/rank0.json" <<'EOF' || true
import json, sys
d = json.load(open(sys.argv[1]))
print("rank0:", {k: d.get(k) for k in ("api", "supported", "hcas", "all_reduce_bytes", "all_reduce_ms", "max_abs_err", "gather_ok", "error")})
EOF
echo "logs: $OUT"
[[ $fail == 0 ]] && echo "RING-SMOKE PASS" || { echo "RING-SMOKE FAIL"; exit 1; }
