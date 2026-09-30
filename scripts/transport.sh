# Transport-only opt-in. Source after the operator's environment/profile.
# Model, scheduler, precision and graph options are left to profiles/current.env.
configure_transport() {
  case ${TRANSPORT:-switched} in
    switched)
      [[ ${SWITCHLESS_ROCE_RING:-0} == 0 ]] || { echo 'SWITCHLESS_ROCE_RING requires TRANSPORT=switchless' >&2; return 2; }
      return 0;;
    switchless) ;;
    *) echo 'TRANSPORT must be switched or switchless' >&2; return 2;;
  esac
  [[ ${SWITCHLESS_ROCE_RING:-0} == 0 || ${SWITCHLESS_ROCE_RING:-0} == 1 ]] || { echo 'SWITCHLESS_ROCE_RING must be 0 or 1' >&2; return 2; }
  # All validation precedes the first remote action. Values enter a remote shell
  # command, so reject whitespace, metacharacters and ambiguous path expansions.
  [[ ${#HOSTS[@]} == 4 && ${#IPS[@]} == 4 ]] || { echo 'switchless requires four hosts and four bootstrap IPs in ring rank order' >&2; return 2; }
  [[ ${NCCL_HOST_DIR:-} =~ ^/[a-zA-Z0-9_./-]+$ && $NCCL_HOST_DIR != / ]] || { echo 'absolute NCCL_HOST_DIR without whitespace required' >&2; return 2; }
  [[ ${SWITCHLESS_NCCL_SHA256:-} =~ ^[a-f0-9]{64}$ ]] || { echo 'SWITCHLESS_NCCL_SHA256 must pin the patched library' >&2; return 2; }
  [[ ${FABRIC_IFACE:-} =~ ^[a-zA-Z0-9_.:-]+$ && ${IB_HCA:-} =~ ^[a-zA-Z0-9_:.,-]+$ ]] || { echo 'explicit bootstrap interface and HCA allowlist required' >&2; return 2; }
  [[ ${NCCL_IB_GID_INDEX:-3} =~ ^[0-9]+$ && ${SWITCHLESS_SUBNET_PREFIX_LEN:-24} =~ ^[0-9]+$ ]] || { echo 'numeric GID index and subnet prefix required' >&2; return 2; }
  # Validate IPv4 addressing without touching the network or reading environment.
  python3 - "${SWITCHLESS_ADDR_RANGE:-}" "${SWITCHLESS_SUBNET_PREFIX_LEN:-24}" "${IPS[@]}" <<'PY'
import ipaddress, sys
try:
    network = ipaddress.IPv4Network(sys.argv[1], strict=True)
    prefix = int(sys.argv[2])
    addresses = [ipaddress.IPv4Address(x) for x in sys.argv[3:]]
    assert network.prefixlen <= prefix <= 32 and len(set(addresses)) == 4
except (ValueError, AssertionError):
    raise SystemExit('valid switchless IPv4 range/subnet prefix and four distinct bootstrap IPs required')
PY
  if [[ ${SWITCHLESS_ROCE_RING:-0} == 1 ]]; then
    # RoCEnante over the hardware-forwarded opposite-node paths: the per-rank peer
    # maps come from the mesh plan (scripts/ring_mesh plan.py env.txt) and are the
    # same value the runtime reads as B12X_ROCE_PEER_HCA_MAPS.  See docs/switchless.md.
    [[ -n ${SWITCHLESS_ROCE_PEER_HCA_MAPS:-} ]] || { echo 'SWITCHLESS_ROCE_RING=1 requires SWITCHLESS_ROCE_PEER_HCA_MAPS (the mesh plan env.txt value)' >&2; return 2; }
    python3 - "${SWITCHLESS_ROCE_PEER_HCA_MAPS}" "$IB_HCA" <<'PY'
import sys
raw, hca_text = sys.argv[1], sys.argv[2]
hcas = [h for h in hca_text.split(',') if h]
try:
    maps = [m.strip() for m in raw.split(';')]
    assert len(maps) == 4, "need four ';'-separated maps, one per rank"
    for rank, entry_text in enumerate(maps):
        seen = {}
        for entry in entry_text.split(','):
            peer_text, paths_text = entry.split('=', 1)
            peer = int(peer_text)
            paths = tuple(int(p) for p in paths_text.split('/'))
            assert peer != rank and 0 <= peer <= 3, f"rank {rank}: bad peer {peer}"
            assert peer not in seen, f"rank {rank}: peer {peer} repeated"
            assert len(paths) == 2 and len(set(paths)) == 2, f"rank {rank}: peer {peer} needs two distinct paths"
            assert all(0 <= p < len(hcas) for p in paths), f"rank {rank}: peer {peer} path outside IB_HCA ({len(hcas)} devices)"
            seen[peer] = paths
        assert sorted(seen) == [r for r in range(4) if r != rank], f"rank {rank}: incomplete peer map"
except (AssertionError, ValueError) as exc:
    raise SystemExit(f'invalid SWITCHLESS_ROCE_PEER_HCA_MAPS: {exc}')
PY
    # Sizes have to fit the hairpin queues the mesh install reports (256 KiB for
    # queue 8192); the two-wave schedule is off once nothing drops.
    local roce_max=${SWITCHLESS_ROCE_MAX_SIZE:-262144} roce_gather=${SWITCHLESS_ROCE_GATHER_MAX_SIZE:-262144}
    [[ $roce_max =~ ^[0-9]+$ && $roce_gather =~ ^[0-9]+$ ]] || { echo 'SWITCHLESS_ROCE_MAX_SIZE and SWITCHLESS_ROCE_GATHER_MAX_SIZE must be numeric' >&2; return 2; }
    (( roce_max > 0 && roce_max % 16 == 0 && roce_gather % 16 == 0 )) || { echo 'ring RoCEnante sizes must be positive multiples of 16 bytes' >&2; return 2; }
  fi
  local kv key clean=""
  for kv in ${EXTRA_ENV:-}; do
    key=${kv%%=*}
    case $key in
      GLM_ROCE_ALLREDUCE|B12X_ROCE_HCA) ;; # set by the transport for this fabric
      GLM_ROCE_RING|B12X_ROCE_PEER_HCA_MAP|B12X_ROCE_PEER_HCA_MAPS|B12X_ROCE_OPPOSITE_PATHS|GLM_ROCE_MAX_SIZE|GLM_ROCE_GATHER_MAX_SIZE|B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES)
        if [[ ${SWITCHLESS_ROCE_RING:-0} == 1 ]]; then
          echo 'EXTRA_ENV overrides the switchless ring RoCEnante configuration' >&2; return 2
        fi
        clean="$clean $kv";;
      NCCL_*|LD_PRELOAD|VLLM_NCCL_SO_PATH|TORCH_USE_RTLD_GLOBAL|GLOO_SOCKET_IFNAME|MN_IF_NAME|TP_SOCKET_IFNAME|VLLM_HOST_IP)
        echo 'transport overrides in EXTRA_ENV conflict with switchless configuration' >&2; return 2;;
      *) clean="$clean $kv";;
    esac
  done
  if [[ ${SWITCHLESS_ROCE_RING:-0} == 1 ]]; then
    EXTRA_ENV="${clean# } GLM_ROCE_ALLREDUCE=1 GLM_ROCE_RING=1 B12X_ROCE_HCA=$IB_HCA B12X_ROCE_PEER_HCA_MAPS=$SWITCHLESS_ROCE_PEER_HCA_MAPS GLM_ROCE_MAX_SIZE=${SWITCHLESS_ROCE_MAX_SIZE:-262144} GLM_ROCE_GATHER_MAX_SIZE=${SWITCHLESS_ROCE_GATHER_MAX_SIZE:-262144} B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0"
  else
    EXTRA_ENV="${clean# } GLM_ROCE_ALLREDUCE=0"
  fi
}

transport_args() {
  if [[ ${TRANSPORT:-switched} != switchless ]]; then
    # Preserve the original switched command exactly, including interface match.
    # The continuation lines keep start.sh's original four-space indent, so the rendered command is byte-identical.
    printf '%s' "-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
    -e NCCL_SOCKET_IFNAME==$FABRIC_IFACE -e GLOO_SOCKET_IFNAME=$FABRIC_IFACE -e NCCL_IB_HCA==$IB_HCA \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN"
    return
  fi
  printf '%s' "-e TORCH_USE_RTLD_GLOBAL=1 -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_ALGO=Ring \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA==$IB_HCA -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_IB_ADDR_RANGE=$SWITCHLESS_ADDR_RANGE -e NCCL_IB_ROCE_VERSION_NUM=2 \
-e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_SUBNET_PREFIX_LEN=${SWITCHLESS_SUBNET_PREFIX_LEN:-24} \
-e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=1 -e NCCL_SOCKET_IFNAME=$FABRIC_IFACE \
-e GLOO_SOCKET_IFNAME=$FABRIC_IFACE -e MN_IF_NAME=$FABRIC_IFACE -e TP_SOCKET_IFNAME=$FABRIC_IFACE \
-e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=INFO -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_MAX_CTAS=4 \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e NCCL_IB_PRESERVE_PCI_DOMAIN=1 -e NCCL_IB_ROUTE_DIAGNOSTICS=1 -e NCCL_DEBUG_SUBSYS=INIT,NET"
}