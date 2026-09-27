# Transport-only opt-in. Source after the operator's environment/profile.
# Model, scheduler, precision and graph options are left to profiles/current.env.
configure_transport() {
  case ${TRANSPORT:-switched} in
    switched) return 0;;
    switchless) ;;
    *) echo 'TRANSPORT must be switched or switchless' >&2; return 2;;
  esac
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
  local kv clean=""
  for kv in ${EXTRA_ENV:-}; do
    case ${kv%%=*} in
      GLM_ROCE_ALLREDUCE|B12X_ROCE_HCA) ;; # disabled for a ring, even if profile enables them
      NCCL_*|LD_PRELOAD|VLLM_NCCL_SO_PATH|TORCH_USE_RTLD_GLOBAL|GLOO_SOCKET_IFNAME|MN_IF_NAME|TP_SOCKET_IFNAME|VLLM_HOST_IP)
        echo 'transport overrides in EXTRA_ENV conflict with switchless configuration' >&2; return 2;;
      *) clean="$clean $kv";;
    esac
  done
  EXTRA_ENV="${clean# } GLM_ROCE_ALLREDUCE=0"
}

transport_args() {
  if [[ ${TRANSPORT:-switched} != switchless ]]; then
    # Preserve the original switched command exactly, including interface match.
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
