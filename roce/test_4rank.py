"""4-rank RoCEnante test through the real vLLM path (one rank per Spark, in the roce image).

Run by ``roce/test_4rank.sh`` (never on a serving fleet).  Each rank initializes vLLM's
distributed state exactly like a TP4 engine (``init_distributed_environment`` +
``ensure_model_parallel_initialized(4, 1)``), so the TP group's ``CudaCommunicator`` gets
the shim through the image's ``glm_roce.pth`` hook, then checks:

 1. wiring: the hook loaded, the TP communicator has an enabled adapter, the image's vLLM
    matches ``verify_targets``;
 2. all-reduce, eager, bf16 [T, 4096] for T = 1..512 (8 KiB..4 MiB) and fp32/fp16: RoCE
    result == fixed-rank-order fp32 sum rounded once (bit-exact), identical bits on every
    rank; T = 513 goes to NCCL; the max |RoCE - NCCL| is reported (expected nonzero for
    bf16: NCCL rounds per hop, RoCE rounds once);
 3. all-gather, eager, logits-shaped [rows, 38720] bf16 along the last dim and [rows, 4096]
    along dim 0: byte-identical to the reference;
 4. CUDA graphs: inside vLLM's ``graph_capture()`` capture a decode-like sequence (4 RoCE
    all-reduces of mixed sizes, one fp32, one logits all-gather, one above-cap all-reduce
    that stays on NCCL), replay it 300 times with fresh inputs, interleave eager
    collectives every 10 replays (sequence numbers must stay aligned), verify every 25th
    replay bit-exactly;
 5. latency: graph replays of back-to-back collectives, RoCE vs NCCL (pynccl captured
    directly): all-reduce at 1..512 tokens (decides GLM_ROCE_MAX_SIZE: 2 or 4 MiB) and the
    logits all-gather at 8..128 rows; median us per collective;
 6. health: ``check_health`` on every rank, runtime stats (error_seq must be 0).

Prints one ``RESULT {json}`` line per rank and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.distributed as dist

HIDDEN = 4096
VOCAB_SHARD = 154880 // 4
AR_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)  # 8 KiB .. 4 MiB (the default cap)
ABOVE_CAP = 513


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def cpu_gather(t: torch.Tensor, group) -> list[torch.Tensor]:
    """Every rank's copy of ``t`` via the gloo group (byte copy, any dtype)."""
    world = dist.get_world_size(group)
    raw = t.detach().contiguous().cpu().view(-1).view(torch.uint8)
    parts = [torch.empty_like(raw) for _ in range(world)]
    dist.all_gather(parts, raw, group=group)
    return [p.view(t.dtype).view(t.shape) for p in parts]


def rank_order_sum(parts: list[torch.Tensor], dtype) -> torch.Tensor:
    """What the one-shot kernel computes: fp32 accumulate in rank order, one rounding."""
    acc = parts[0].float().clone()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(dtype)


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29671)
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    rank = args.rank
    result: dict = {"rank": rank, "ok": False, "checks": {}}
    checks = result["checks"]
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)

    try:
        boot = sys.modules.get("glm_roce.boot")
        checks["hook_loaded"] = bool(boot is not None and boot._finder is not None)
        assert checks["hook_loaded"], "glm_roce.pth hook not active: is GLM_ROCE_ALLREDUCE=1 set in the container?"

        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed import parallel_state as ps
        from vllm.distributed.communication_op import (
            tensor_model_parallel_all_gather,
            tensor_model_parallel_all_reduce,
        )

        with set_current_vllm_config(VllmConfig()):
            ps.init_distributed_environment(
                world_size=args.world, rank=rank, local_rank=0,
                distributed_init_method=f"tcp://{args.master}:{args.port}", backend="nccl",
            )
            ps.ensure_model_parallel_initialized(args.world, 1)
            tp = ps.get_tp_group()
            dc = tp.device_communicator
            comm = getattr(dc, "glm_roce_comm", None)
            checks["adapter_enabled"] = bool(comm is not None and not comm.disabled)
            assert checks["adapter_enabled"], f"TP communicator has no enabled adapter: {comm!r}"
            from glm_roce.install import verify_targets

            problems = verify_targets()
            checks["verify_targets"] = problems
            assert not problems, problems
            cpu = tp.cpu_group
            pynccl = dc.pynccl_comm
            stats0 = comm.stats()
            result["runtime"] = {k: stats0.get(k) for k in ("hcas", "max_size", "max_gather_bytes", "slot_bytes", "spin_limit")}
            log(rank, f"adapter up: {result['runtime']}")

            # -- 2. eager all-reduce ------------------------------------------------------
            g = torch.Generator(device="cpu").manual_seed(1000 + rank)
            ar = []
            max_vs_nccl = 0.0
            for dtype, sizes in ((torch.bfloat16, AR_TOKENS + (ABOVE_CAP,)), (torch.float32, (1, 16, 128, 256)), (torch.float16, (1, 64))):
                for tokens in sizes:
                    x = torch.randn(tokens, HIDDEN, generator=g).to(dtype).to(dev)
                    routed = comm.should_custom_ar(x)
                    y = tensor_model_parallel_all_reduce(x)
                    torch.cuda.synchronize()
                    parts = cpu_gather(x, cpu)
                    ref = rank_order_sum(parts, dtype)
                    exact = torch.equal(y.cpu(), ref)
                    same_everywhere = len(set(dist_all_strings(digest(y), cpu))) == 1
                    y_nccl = pynccl.all_reduce(x.clone())
                    torch.cuda.synchronize()
                    diff = (y.float() - y_nccl.float()).abs().max().item()
                    if routed:
                        max_vs_nccl = max(max_vs_nccl, diff)
                    ar.append({"dtype": str(dtype)[6:], "tokens": tokens, "bytes": x.numel() * x.element_size(),
                               "roce": routed, "exact_rank_order": exact, "same_all_ranks": same_everywhere,
                               "max_abs_vs_nccl": diff})
                    assert same_everywhere, f"ranks disagree at {dtype} T={tokens}"
                    if routed:
                        assert exact, f"RoCE result != rank-order sum at {dtype} T={tokens}"
            expect_routed = [t <= 512 for t in AR_TOKENS + (ABOVE_CAP,)]
            got_routed = [r["roce"] for r in ar if r["dtype"] == "bfloat16"]
            assert got_routed == expect_routed, f"routing {got_routed} != {expect_routed}"
            checks["all_reduce_eager"] = ar
            checks["max_abs_vs_nccl_routed"] = max_vs_nccl
            log(rank, f"eager all-reduce ok, max |RoCE-NCCL| {max_vs_nccl:.3g}")

            # -- 3. eager all-gather ---------------------------------------------------------
            ag = []
            for rows, cols, dim in ((1, VOCAB_SHARD, -1), (8, VOCAB_SHARD, -1), (64, VOCAB_SHARD, -1), (32, HIDDEN, 0)):
                x = torch.randn(rows, cols, generator=g).to(torch.bfloat16).to(dev)
                routed = comm.should_all_gather(x, dim % x.dim())
                y = tensor_model_parallel_all_gather(x, dim=dim)
                torch.cuda.synchronize()
                ref = torch.cat(cpu_gather(x, cpu), dim=dim)
                exact = torch.equal(y.cpu(), ref)
                ag.append({"shape": [rows, cols], "dim": dim, "roce": routed, "exact": exact})
                assert exact and routed, f"all-gather {rows}x{cols} dim {dim}: routed={routed} exact={exact}"
            checks["all_gather_eager"] = ag
            log(rank, "eager all-gather ok")

            # -- 4. CUDA graphs ----------------------------------------------------------------
            specs = [(torch.bfloat16, 1), (torch.bfloat16, 8), (torch.bfloat16, 64), (torch.bfloat16, 256),
                     (torch.float32, 16), (torch.bfloat16, 600)]  # last: 4.7 MiB, NCCL inside the graph
            ins = [torch.zeros(t, HIDDEN, dtype=d, device=dev) for d, t in specs]
            ag_in = torch.zeros(8, VOCAB_SHARD, dtype=torch.bfloat16, device=dev)
            graph = torch.cuda.CUDAGraph()
            with ps.graph_capture(device=dev) as ctx:
                # warm the exact shapes once, eagerly, on the capture stream
                outs = [tensor_model_parallel_all_reduce(t) for t in ins]
                ag_out = tensor_model_parallel_all_gather(ag_in, dim=-1)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=ctx.stream):
                    outs = [tensor_model_parallel_all_reduce(t) for t in ins]
                    ag_out = tensor_model_parallel_all_gather(ag_in, dim=-1)
            torch.cuda.synchronize()
            bad = 0
            verified = 0
            for i in range(args.replays):
                gi = torch.Generator(device="cpu").manual_seed(10_000 * (i + 1) + rank)
                for t in ins:
                    t.copy_(torch.randn(t.shape, generator=gi).to(t.dtype))
                ag_in.copy_(torch.randn(ag_in.shape, generator=gi).to(ag_in.dtype))
                graph.replay()
                if i % 10 == 5:  # eager collectives between replays keep the epoch aligned
                    e = tensor_model_parallel_all_reduce(torch.ones(4, HIDDEN, dtype=torch.bfloat16, device=dev))
                    torch.cuda.synchronize()
                    assert torch.all(e == args.world), "eager all-reduce between replays is wrong"
                if i % 25 == 0 or i == args.replays - 1:
                    torch.cuda.synchronize()
                    verified += 1
                    for (d, _), t, o in zip(specs, ins, outs):
                        ref = rank_order_sum(cpu_gather(t, cpu), d)
                        if d == torch.bfloat16 and t.shape[0] == 600:
                            ok = torch.allclose(o.float().cpu(), ref.float(), atol=0.1, rtol=0.02)  # NCCL order
                        else:
                            ok = torch.equal(o.cpu(), ref)
                        bad += 0 if ok else 1
                    bad += 0 if torch.equal(ag_out.cpu(), torch.cat(cpu_gather(ag_in, cpu), dim=-1)) else 1
            torch.cuda.synchronize()
            checks["graph"] = {"replays": args.replays, "verified": verified, "mismatches": bad}
            assert bad == 0, f"{bad} mismatches in graph replays"
            del graph
            log(rank, f"graph replay ok ({args.replays} replays, {verified} verified)")

            # -- 5. latency --------------------------------------------------------------------
            if not args.no_bench:
                checks["latency_us"] = bench(tp, pynccl, dev, ps, tensor_model_parallel_all_reduce)
                log(rank, f"latency {checks['latency_us']}")

            # -- 6. health ---------------------------------------------------------------------
            comm.check_health()
            st = comm.stats()
            result["runtime_end"] = {k: st.get(k) for k in ("epoch", "error_seq", "error_peer", "error_hca", "ctrl_seq")}
            assert st.get("error_seq") == 0, st
            result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{exc!r}"
        traceback.print_exc()
    print("RESULT " + json.dumps(result, default=str), flush=True)
    return 0 if result["ok"] else 1


def dist_all_strings(s: str, group) -> list[str]:
    out = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, s, group=group)
    return out


def bench(tp, pynccl, dev, ps, all_reduce, reps: int = 32, iters: int = 50):
    """Median us per collective over graph replays of ``reps`` back-to-back collectives."""

    def timed(fn):
        graph = torch.cuda.CUDAGraph()
        with ps.graph_capture(device=dev) as ctx:
            fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=ctx.stream):
                for _ in range(reps):
                    fn()
        times = []
        for _ in range(iters):
            dist.barrier(group=tp.cpu_group)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0 / reps)
        del graph
        return round(statistics.median(times[5:]), 1)

    table = {"all_reduce": {}, "all_gather": {}}
    comm = tp.device_communicator.glm_roce_comm
    for tokens in (1, 4, 16, 64, 128, 256, 384, 512):
        x = torch.randn(tokens, HIDDEN, device=dev).to(torch.bfloat16)
        assert comm.should_custom_ar(x), tokens
        table["all_reduce"][tokens] = {
            "bytes": x.numel() * 2,
            "roce": timed(lambda: all_reduce(x)),
            "nccl": timed(lambda: pynccl.all_reduce(x)),
        }
    world = tp.world_size
    for rows in (8, 64, 128):
        x = torch.randn(rows, VOCAB_SHARD, device=dev).to(torch.bfloat16)
        out = torch.empty(world * rows, VOCAB_SHARD, device=dev, dtype=torch.bfloat16)
        assert comm.should_all_gather(x, 1), rows
        table["all_gather"][rows] = {
            "shard_bytes": x.numel() * 2,
            "roce": timed(lambda: comm.all_gather(x, 1)),
            "nccl": timed(lambda: pynccl.all_gather(out, x)),  # same bytes, dim-0 layout
        }
    return table


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"elapsed {time.time() - t0:.1f}s", flush=True)
    sys.exit(rc)
