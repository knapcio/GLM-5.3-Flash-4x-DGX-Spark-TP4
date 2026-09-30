# Optional four-Spark switchless ring

> **Community-contributed.** The base switchless mode was contributed by
> [@othexmr](https://github.com/othexmr) in [PR #1](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/pull/1)
> and rebased onto the 2026-09-29 release; upstream checks only that the switched launch is unchanged and that the
> switchless launch renders. The optional RoCEnante-over-the-ring mode (`SWITCHLESS_ROCE_RING=1`, below) was built,
> booted and measured on four Sparks on 2026-10-01. If the base mode breaks for you,
> please [open an issue](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/issues) with the items listed
> under [Reporting a problem](#reporting-a-problem).

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
  and the [othexmr four-PF source profile](https://github.com/othexmr/switchless-nccl/blob/94c2669e82ea47e9d85dafee9ef46b85b931d21a/docs/dual-pf.md)
  for the transport source and patch provenance. Select a tested artifact and record its SHA256;
  do not substitute stock upstream NCCL or infer patch support from its version.
- The same prepared weights and image prerequisites as the switched recipe.
  The RoCEnante-capable image can still be used, but RoCEnante is off in this
  mode by default because its all-peer fabric assumptions do not hold for the
  ring; `SWITCHLESS_ROCE_RING=1` re-enables it with the ring variant once the
  mesh below is installed (that variant needs an image carrying
  `b12x.comm.roce_ring`).

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

## What the ring gives up from the current profile

`profiles/current.env` is tuned on the switched fleet. On the ring every switch in it still loads, and none changes
the output, but the ones built on RoCEnante have nothing to act on:

| Profile piece | On the ring |
|---|---|
| `GLM_ROCE_ALLREDUCE=1` (RoCEnante one-shot all-reduce and all-gather) | forced to `0` unless `SWITCHLESS_ROCE_RING=1`; without the mesh, every collective runs on the patched NCCL ring |
| `B12X_ROCE_HCA` | removed unless `SWITCHLESS_ROCE_RING=1` (then set from `IB_HCA`) |
| `GATHER_ROUTE=1` (`GLM_ROCE_AG_DIM0_NCCL`, `overlay/glm_roce_gather_route.py`) | inert without the mesh: it only moves gathers off RoCEnante, and they are all on NCCL already |
| `GLM_ROCE_PROXY_CPUS=auto` | inert without the mesh (there is no RoCE proxy thread); in ring mode it pins the ring proxy thread |
| L2 prefetch window A (`GLM_L2_PREFETCH=1`) | works |
| L2 prefetch windows B / C / D (`GLM_L2_PREFETCH_AR`, `GLM_L2_PREFETCH_MLA_AR`, `GLM_L2_PREFETCH_DRAFT`, `L2PF_V2=1`) | arm but never fire without the mesh: they are forked from the RoCEnante all-reduce hook; in ring mode they fork from the ring hook |
| Everything else (kernels, drafting, scheduler, prefill package) | unchanged |

Without the mesh, expect decode to be slower than the switched numbers in the README: every decode-size
all-reduce pays NCCL's latency instead of RoCEnante's, and opposite nodes talk through a transit node. With
`SWITCHLESS_ROCE_RING=1` (next section) the same decode-size collectives run on the path-aware RoCEnante the
DeepSeek-V4.1 recipe uses; prefill stays a few percent under the switched fleet either way (the ring bisection is
one link, not two).

RoCEnante needs a path to every peer. On the ring that path is built in the neighbours' ConnectX-7 hardware
(FujitsuPolycom/sparkring: an RDMA-TX tag plus a tc-hairpin redirect, no CPU in the data path), and the ring
variant of RoCEnante (`b12x.comm.roce_ring`) routes each peer over it. This recipe vendors that package; the
DeepSeek-V4.1 recipe's
[switchless-ring notes](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/main/docs/switchless-ring.md)
document the mesh and its measurements.

## RoCEnante on the ring (`SWITCHLESS_ROCE_RING=1`)

Optional and off by default. It keeps the patched-NCCL pin and the ring NCCL environment, and additionally runs
decode-size all-reduce/all-gather on `b12x.comm.roce_ring` through hardware-forwarded opposite-node paths.

Prerequisites (all four hosts):

- **the mesh**: sparkring's hardware forwarding installed and active (two `/32` routes per node, the tc hairpin
  redirect rules on the neighbour-facing ports of both PCIe domains, and the RDMA-TX source markers), with the
  driver profile sparkring qualifies (`hairpin_num_queues` 4 set at boot, `flow_steering_mode hmfs`, eswitch
  `legacy`, `hw-tc-offload on`). The DeepSeek-V4.1 recipe's `scripts/ring_mesh/plan.py` (sparkring `f16b5f4`)
  plans and installs it; its `env.txt` carries the peer maps;
- **`SWITCHLESS_ROCE_PEER_HCA_MAPS`**: that `env.txt` value — four `;`-separated maps in TP rank order
  (`peer=path0/path1`, absolute HCA indices). The launcher validates the shape and every index against
  `IB_HCA` before any remote action;
- **sizes that fit the hairpin queues**: `SWITCHLESS_ROCE_MAX_SIZE` / `SWITCHLESS_ROCE_GATHER_MAX_SIZE`
  default to 262144, the cap for queue size 8192. Larger messages overflow the hairpin queue and the go-back-N
  retransmits make them slower than NCCL. The forced `B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0` is only safe
  without drops, i.e. with the caps at or under the queue's.

```sh
SWITCHLESS_ROCE_RING=1
SWITCHLESS_ROCE_PEER_HCA_MAPS="1=0/2,2=0/3,3=1/3;0=1/3,2=0/2,3=0/3;0=1/2,1=1/3,3=0/2;0=0/2,1=1/2,2=1/3"
#SWITCHLESS_ROCE_MAX_SIZE=262144
#SWITCHLESS_ROCE_GATHER_MAX_SIZE=262144
```

The launcher sets `GLM_ROCE_ALLREDUCE=1`, `GLM_ROCE_RING=1`, `B12X_ROCE_HCA`, the maps, both caps and
`B12X_ROCE_TWO_WAVE_THRESHOLD_BYTES=0` itself; conflicting `EXTRA_ENV` entries are refused. Fail-stop is kept:
with the default `GLM_ROCE_REQUIRE=1` a runtime that cannot start (missing map, broken mesh) makes every rank
raise instead of quietly serving on NCCL.

Validate on the boot log: one `GLM_ROCE_READY ... hcas=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` line
per rank, plus the ring NCCL lines. On the fabric, an RDMA write to the opposite node with
`--flow_label=16383` should cross the neighbour's tc rule in hardware (its `in_hw` counter rises, the
neighbour's `IpForwDatagrams` stays flat).

Effect (measured on this fleet, 2026-10-01, `glm53-roce:v12-ring-20261001`, default GPU clocks; the switched
column is the README's published numbers):

| | ring + RoCEnante | switched (README) | ring, NCCL only |
|---|---:|---:|---:|
| step ms, prose / code / JSON (`bench/accept_probe.py`) | 33.8 / 42.5 / 43.7 | 39.0 / 48.9 / 50.0 | ~45-47 (prose) |
| sparkDash structured c1, 400 tok (best of two runs) | 167.8 tok/s | 168.8 tok/s | 104.4 tok/s |
| cold prefill, 16k-128k (`bench/prefill_bench.py`) | 3,167-3,279 tok/s | 3,426-3,510 tok/s | — |

The decode step is at or below the switched fleet's published step times; prefill stays ~7 % under (the ring
bisection). Structured decode at c8/c16 is still below the switched single-run cells (287/288 vs 463/879 tok/s):
part of that is the 256 KiB size cap pushing the larger verify all-reduces back to NCCL, part is single-run
acceptance variance — raise `SWITCHLESS_ROCE_MAX_SIZE` only together with the hairpin queue size behind it.

`roce/test_ring_smoke.sh` is a 4-rank check that needs no vLLM and no NCCL (gloo rendezvous, one all-reduce
verified against gloo, one dim-0 all-gather); it prints `RING-SMOKE PASS` when the opposite-node paths carry
data.

Operational note: `hairpin.sh` re-initialises the RDMA functions when it changes `hairpin_queue_size`, and that
flushes the **markers'** RDMA-TX rules. The marker processes stay alive, and `mesh-up.sh` starts them with
`systemctl start` (a no-op for a running unit), so after any hairpin change restart them explicitly
(`sudo systemctl restart dsv41-mesh-marker@<device>` for every marker device of that host). The symptom if
forgotten: writes to the opposite node stall and fail with transport retry counter exceeded (vendor_err 0x81),
while neighbour traffic still works.

## Evidence and limits

The transport adaptation reached model readiness and completed three High
92-case tool-eval runs on an older pinned recipe (`dddb0347`) in a private
four-Spark lab window. It was then rebased onto the 2026-09-29 release (`3ab5ca0`), which includes newer
model changes; that complete combination has **not** been hardware
qualified. No performance claim, generic topology support or equivalence to the
switched deployment is established by these CPU tests. Do not transplant the
historical scores as results for this branch.

Tests compare all four default switched command hashes against the 2026-09-29 release
launcher (`3ab5ca0`), prove that only transport arguments change in switchless mode, reject
bad/conflicting configuration before any remote command, and exercise artifact
hash/architecture/mount-boundary and selected-field inventory failures. They
run without Docker, SSH, GPU access or model imports:

```sh
python3 -m unittest tests/test_switchless.py -v    # also part of tests/run_cpu_tests.sh
bash -n start.sh scripts/transport.sh
```

## Reporting a problem

Please open an issue with:

- the `TRANSPORT=switchless` part of your env file (hosts and IPs can be redacted) and the output of
  `DRY=1 ENV_FILE=<your file> ./start.sh serve`;
- which NCCL build you use (source commit or release, the SHA256 you pinned) and your cabling / addressing;
- from rank 0, the NCCL lines of `./start.sh logs 0 400`: `NCCL_SWITCHLESS_RING_ONLY`, `ListenerRouting`,
  `Subnet-aware routing`, `Connected all rings`, and any `NCCL WARN` or error;
- where it stopped: the library check, the preflight, NCCL init, model load, CUDA graph capture, or a wrong or
  garbled answer after `/health` turned 200.

