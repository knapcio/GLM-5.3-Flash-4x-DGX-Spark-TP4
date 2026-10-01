#!/usr/bin/env python3
"""CPU test of bench/final_bench.py against a mock OpenAI / sparkDash server (no GPU, no network).

  python3 tests/test_final_bench_cpu.py

Checks: kldlong collect cuts every text to the exact token target and records continuation + teacher-forced logprobs;
compare PASSes an identical arm and a slightly perturbed arm, FAILs a strongly perturbed one and a mismatched panel;
tscan counts a planted CJK salad and a planted loop and nothing else; loop_tail has no false positive on ordinary
prose / code; prefill parses the sparkDash job results.
"""
import http.server
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
FB = os.path.join(HERE, "..", "bench", "final_bench.py")
sys.path.insert(0, os.path.join(HERE, "..", "bench"))
import final_bench as fb  # noqa: E402

FAIL = []
STATE = {"noise": 0.0, "salad_prompt": "lighthouse", "loop_prompt": "INI-like"}


def check(c, m):
    print(("ok   " if c else "FAIL ") + m)
    if not c:
        FAIL.append(m)


def dist(pos, noise):
    rng = random.Random(pos)
    logits = [rng.gauss(0, 2) for _ in range(12)]
    if noise:
        r2 = random.Random(pos * 7 + 1)
        logits = [x + r2.gauss(0, noise) for x in logits]
    m = max(logits)
    z = math.log(sum(math.exp(x - m) for x in logits)) + m
    return {str(i): x - z for i, x in enumerate(logits)}


def topk(d, k):
    return dict(sorted(d.items(), key=lambda kv: -kv[1])[:k])


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if "prefill-bench/" in self.path:
            return self._send({"status": "completed", "results": [
                {"targetTokens": s, "prefillTps": 3000.0 + s / 1000, "ttftMs": s / 3.0} for s in STATE["sizes"]]})
        return self._send({"active": None})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        n = STATE["noise"]
        if self.path == "/tokenize":
            return self._send({"tokens": list(range(len(body["prompt"]) // 3))})
        if self.path == "/v1/completions":
            ids = body["prompt"]
            k = body.get("prompt_logprobs") or body.get("logprobs") or 5
            if body.get("prompt_logprobs"):
                pl = [None] + [{t: {"logprob": v} for t, v in topk(dist(i, n), k).items()} for i in range(1, len(ids))]
                return self._send({"choices": [{"text": "x", "prompt_logprobs": pl}]})
            toks, tops = [], []
            for j in range(body["max_tokens"]):
                d = dist(10 ** 6 + j, n)
                toks.append("token_id:" + max(d, key=d.get))
                tops.append({"token_id:" + t: v for t, v in topk(d, k).items()})
            return self._send({"choices": [{"text": "", "finish_reason": "length",
                                            "logprobs": {"tokens": toks, "top_logprobs": tops}}],
                               "usage": {"prompt_tokens": len(ids)}})
        if self.path == "/v1/chat/completions":
            p = body["messages"][0]["content"]
            if STATE["salad_prompt"] in p and body.get("temperature", 0) > 0.9:
                text = "The keeper opened it 的是在了不和有大 and read."
            elif STATE["loop_prompt"] in p and body.get("temperature", 0) > 0.9 and body.get("seed") is not None:
                text = "def parse(x):\n" + "    return parse(x)  # again\n" * 40
            elif "JSON" in p or body.get("response_format"):
                text = json.dumps({"items": [{"name": "tent", "qty": 1}]})
            else:
                rng = random.Random(p)
                words = "plan rollout schema test backup owner risk staged window metric budget review".split()
                text = " ".join(f"Step {i}: " + " ".join(rng.choice(words) for _ in range(9)) + "." for i in range(20))
            return self._send({"choices": [{"message": {"content": text, "reasoning_content": "thinking about it."},
                                            "finish_reason": "stop"}], "usage": {"completion_tokens": 300}})
        if "prefill-bench" in self.path:
            STATE["sizes"] = body["contextSizes"]
            return self._send({"benchId": "b1"})
        self.send_response(404)
        self.end_headers()


def run(args, env):
    return subprocess.run([sys.executable, FB] + args, capture_output=True, text=True, env=env)


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    env = dict(os.environ, GLM_BASE=f"http://127.0.0.1:{port}", HOME=tempfile.mkdtemp())
    d = tempfile.mkdtemp()
    texts = os.path.join(d, "texts.json")
    json.dump([{"id": "a", "kind": "prose", "tokens": 300, "text": "x" * 3000},
               {"id": "b", "kind": "code", "tokens": 5000, "text": "y" * 1200}], open(texts, "w"))
    outs = {}
    for name, noise in (("ref", 0.0), ("same", 0.0), ("near", 0.05), ("far", 1.5)):
        STATE["noise"] = noise
        r = run(["kldlong", "collect", "--out", os.path.join(d, name + ".json"), "--texts", texts, "--k", "8",
                 "--gen", "16"], env)
        outs[name] = json.load(open(os.path.join(d, name + ".json")))
        check(r.returncode == 0 and "KLDLONG-COLLECT" in r.stdout, f"collect {name}: rc {r.returncode} {r.stderr[-200:]}")
    it = outs["ref"]["items"]
    check(it[0]["n_prompt"] == 300 and it[1]["n_prompt"] == 400 and len(it[0]["prompt_lp"]) == 300
          and len(it[0]["gen_tokens"]) == 16, "collect: exact token cut (or all tokens when shorter), positions, gen")
    for name, want in (("same", 0), ("near", 0), ("far", 1)):
        r = run(["kldlong", "compare", os.path.join(d, "ref.json"), os.path.join(d, name + ".json")], env)
        line = [x for x in r.stdout.splitlines() if x.startswith("GATE long_kld")]
        check(r.returncode == want and line and ("PASS" if want == 0 else "FAIL") in line[0],
              f"compare {name}: rc {r.returncode} ({line[0] if line else r.stdout[-200:]})")
    bad = json.load(open(os.path.join(d, "ref.json")))
    bad["items"][0]["n_prompt"] = 299
    json.dump(bad, open(os.path.join(d, "bad.json"), "w"))
    r = run(["kldlong", "compare", os.path.join(d, "ref.json"), os.path.join(d, "bad.json")], env)
    check(r.returncode == 1 and "MISMATCHED PANEL" in r.stdout, "compare: a mismatched panel fails")
    r = run(["tscan", "t1", "--max-tokens", "64"], env)
    line = [x for x in r.stdout.splitlines() if x.startswith("TSCAN:")]
    # salad prompt: 2 T=1.0 runs (c1 seeded + c4); loop prompt: c1 seeded T=1.0 and c4 (seeded) -> 4 flagged
    check(r.returncode == 1 and line and line[0].startswith("TSCAN: 4 salad of 35"), f"tscan: planted salad/loops found ({line})")
    check(not fb.loop_tail(open(FB).read()[-3000:]) and not fb.loop_tail("The plan is simple. " + "word " * 5),
          "loop_tail: no false positive on code / short text")
    check(fb.loop_tail("intro " + "abc def ghi, " * 30), "loop_tail: catches a repeated unit")
    check(fb.scan_text("x", "的是在了不和有")[0] == ["SALAD"] and fb.scan_text("x", "Zażółć gęślą jaźń")[0] == [],
          "scan_text: CJK flagged, Polish diacritics not")
    check(fb.base_port("http://127.0.0.1:8888") == 8888 and fb.base_port("http://h:8093") == 8093
          and fb.base_port("http://h") == 8093 and fb.base_port("http://[::1]:8888") == 8888,
          "base_port: endpoint port, defaulting to 8093")
    r = run(["prefill", "p1", "--sizes", "32768,131072", "--reps", "2", "--dash",
             f"http://127.0.0.1:{port}/prefill-bench"], env)
    check(r.returncode == 0 and "PREFILL 32768 tps 3033" in r.stdout and "PREFILL 131072" in r.stdout,
          f"prefill: parsed ({r.stdout[-160:]} {r.stderr[-200:]})")
    srv.shutdown()
    print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
