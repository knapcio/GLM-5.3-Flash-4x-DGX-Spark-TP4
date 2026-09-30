"""4-rank smoke test for ``b12x.comm.roce_ring`` over the hardware-forwarded mesh.

No vLLM and no NCCL: a gloo rendezvous, one ring runtime per rank, one all-reduce
(fixed-rank-order sum, verified against gloo) and one dim-0 all-gather.  The point
is to prove the QPs to the *opposite* ranks come up and move data through the
neighbours' tc rules — the Step the switchless ring mode depends on.

Run by ``roce/test_ring_smoke.sh`` (one container per Spark, fleet idle).
Prints one ``RESULT {json}`` line per rank and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import torch
import torch.distributed as dist

HIDDEN = 4096
GATHER_ROWS, GATHER_COLS = 8, 64


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--master", required=True)
    ap.add_argument("--port", type=int, default=29672)
    ap.add_argument("--tokens", type=int, default=16)  # 128 KiB at hidden 4096 bf16, inside the 256 KiB cap
    args = ap.parse_args()

    os.environ["MASTER_ADDR"] = args.master
    os.environ["MASTER_PORT"] = str(args.port)
    dist.init_process_group("gloo", rank=args.rank, world_size=args.world)

    result: dict = {"rank": args.rank, "ok": False}
    try:
        from b12x.comm import roce_ring

        result["api"] = roce_ring.API_VERSION
        device = torch.device("cuda", 0)
        supported = bool(roce_ring.is_supported(device))
        result["supported"] = supported
        if not supported:
            raise RuntimeError("roce_ring.is_supported is False")

        max_size = int(os.environ.get("GLM_ROCE_MAX_SIZE", "262144"))
        max_gather = int(os.environ.get("GLM_ROCE_GATHER_MAX_SIZE", "262144"))
        runtime = roce_ring.AllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            max_size=max_size,
            max_gather_bytes=max_gather,
        )
        try:
            runtime.prepare((torch.bfloat16, torch.float16, torch.float32))
            result["hcas"] = list(getattr(runtime, "hca_names", ()) or ())

            x = torch.full((args.tokens, HIDDEN), float(args.rank + 1), dtype=torch.bfloat16, device="cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = runtime.all_reduce(x)
            torch.cuda.synchronize()
            result["all_reduce_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            ref = x.float().clone()
            dist.all_reduce(ref, group=dist.group.WORLD)
            err = float((y.float() - ref).abs().max())
            result["all_reduce_bytes"] = args.tokens * HIDDEN * x.element_size()
            result["max_abs_err"] = err

            shard = torch.full((GATHER_ROWS, GATHER_COLS), float(args.rank), dtype=torch.bfloat16, device="cuda")
            gathered = runtime.all_gather(shard, dim=0)
            parts = [torch.empty_like(shard) for _ in range(args.world)]
            dist.all_gather(parts, shard, group=dist.group.WORLD)
            result["gather_ok"] = bool(torch.equal(gathered, torch.cat(parts, dim=0)))
            result["all_gather_shape"] = list(gathered.shape)

            runtime.check_health()
            result["stats"] = runtime.stats()
            result["ok"] = err == 0.0 and result["gather_ok"]
        finally:
            runtime.close()
    except Exception as exc:  # noqa: BLE001 - reported as the rank's verdict
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()[-1500:]

    print("RESULT " + json.dumps(result), flush=True)
    dist.destroy_process_group()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
