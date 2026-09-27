# Optional four-Spark switchless ring

`TRANSPORT=switchless` adapts this recipe to four directly cabled DGX Sparks.
The default remains `switched`, with the original launch arguments. This is a
transport port: checkpoint conversion, model kernels, scheduling, graph shapes,
cache capacity and benchmark settings are unchanged. Existing upstream fixes,
including the KDA boundary repair, are retained.

## Prerequisites

- Four hosts in physical ring order, reachable through a separate management /
  bootstrap network. `HOSTS` and `IPS` have exactly four corresponding entries;
  rank 0 is the API head. This launcher is TP4 only, not TP8.
- Dual-PF configuration selects four PF devices per host: two PCIe-root
  functions on each of the two neighbor-facing ports. The example includes all
  four names; confirm their identity on your hardware. This is not a claim of
  balanced bandwidth across both PCIe roots.
- A configured, independently tested direct-cabled RDMA fabric. This change does
  not configure interfaces, routes, cables, host keys or privileges. Set
  `IB_HCA` to the exact devices in the qualified topology, `SWITCHLESS_ADDR_RANGE`
  to its IPv4 range and `SWITCHLESS_SUBNET_PREFIX_LEN` to the per-link prefix.
  The referenced four-PF build recognizes `10.100.224.0/24` through
  `10.100.227.0/24`; different addressing requires a separately validated
  compatible build. Changing these environment values alone cannot add support.
- A compatible AArch64 NCCL 2.30.7 library **with the switchless patches**, on each
  host. The required contract includes Ring-only routing, subnet-aware GID
  selection, extended IPv4 GIDs and PCI-domain preservation. The four-PF topology
  also needs the four-device support in that build. A two-device-only release is
  not sufficient merely because its filename is the same. See
  [Alex Ellis / OpenFaaS Ltd's switchless NCCL project](https://github.com/alexellis/switchless-nccl)
  and the [othexmr four-PF source profile](https://github.com/othexmr/switchless-nccl/blob/feat/four-pf-source-integration/docs/dual-pf.md)
  for the transport source and patch provenance. Select a tested artifact and record its SHA256;
  do not substitute stock upstream NCCL or infer patch support from its version.
- The same prepared weights and image prerequisites as the switched recipe.
  The RoCEnante-capable image can still be used, but RoCEnante is disabled in
  this mode because its all-peer fabric assumptions do not hold for the ring.

## Configuration

Copy `.env.switchless.example` to a private configuration file and edit its
site-specific fields. Do not overwrite an existing `.env` or an active runtime.
`FABRIC_IFACE` selects the bootstrap interface here; it is independent of the HCA
allowlist and need not carry RDMA traffic. The example IPs are placeholders.

```sh
# Local command rendering only; no SSH, rsync, Docker, fabric or GPU access.
ENV_FILE=.env.switchless DRY=1 ./start.sh serve
```

Before a real launch, the launcher checks the named library's SHA256 and AArch64
ELF header on all four hosts, before overlay synchronization or containers are
started. `NCCL_HOST_DIR` must contain `libnccl.so.2.30.7`; symlinks must resolve
inside that mounted directory. Use absolute paths without whitespace. The
configured digest establishes artifact identity, not transport correctness or
loaded-process identity. Qualify the selected artifact with value-checked
collectives and verify the loaded library in the consuming processes before
claiming serving support. Keep the library immutable throughout the window.

In switchless mode the launcher forces `GLM_ROCE_ALLREDUCE=0`, removes the unused
RoCEnante HCA setting and selects the ring NCCL settings. Both `LD_PRELOAD` and
`VLLM_NCCL_SO_PATH` point to the same read-only library mount. Conflicting
transport settings in `EXTRA_ENV` are rejected. Transport flags match the
reviewed four-Spark ring configuration; `NCCL_MAX_CTAS=4` is the retained transport
configuration, not a universal tuning recommendation. The switched mode keeps
its original GID-index selection; switchless uses the patched subnet-aware
selection instead.

The runtime inventory preflight acquires only container names and the mount
fields it needs. It does not acquire environment variables or whole Docker
inspection output. Existing containers and overlapping runtime mounts still
cause a refusal, including stopped containers needed for provenance.

## Evidence and limits

The transport adaptation reached model readiness and completed three High
92-case tool-eval runs on an older pinned recipe (`dddb0347`) in a private
four-Spark lab window. This branch is based on `364eca1`, which includes newer
upstream model changes; that complete combination has **not** been hardware
qualified. No performance claim, generic topology support or equivalence to the
switched deployment is established by these CPU tests. Do not transplant the
historical scores as results for this branch.

Tests compare all four default switched command hashes against the original
launcher, prove that only transport arguments change in switchless mode, reject
bad/conflicting configuration before any remote command, and exercise artifact
hash/architecture/mount-boundary and selected-field inventory failures. They
run without Docker, SSH, GPU access or model imports:

```sh
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s bench -p 'test_*.py' -v
bash -n start.sh scripts/transport.sh
```
