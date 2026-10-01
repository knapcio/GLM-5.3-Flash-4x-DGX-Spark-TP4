#!/usr/bin/env python3
"""Final-stack gate helpers (diagnostics/glm-final-20260928), run ON the head (Spark_01) against :8093. Stdlib only.

  kldlong collect --out F.json [--texts bench/kld_long_texts.json] [--k 10] [--gen 256]
      Long-prompt fidelity panel that really engages the prefill path: every text is tokenized by the server and cut
      to an exact token count (16500 / 18000 / 24003 / 40001: several full prefill chunks, a short final chunk, row
      counts that are not a multiple of TP = 4, so the padded shard path runs). Per text, in this order:
        1. continuation: prompt ids, greedy, --gen tokens with top-k logprobs (a COLD prefill: nothing sent these ids
           before in this boot, then decode on the state that prefill wrote);
        2. teacher-forced: the same ids with prompt_logprobs=k, max_tokens 1 (prompt_logprobs requests never READ the
           prefix cache, so this is a second cold prefill).
  kldlong compare REF.json CAND.json [--max-kl 0.035]
      KL(ref || cand) over ref's top-k support (+ one tail bucket), per text and pooled: every prompt position, the
      last 2048 positions (the final chunk region), and the continuation positions up to and including the first
      divergence (identical prefixes there). Prints "GATE long_kld <mean> <PASS|FAIL> <positions>".
  tscan LABEL [--max-tokens 768]
      T > 0 correctness scan: 11 prompts (prose / think / code / json / Polish / story / json_schema) at
      (T 1.0, top-p 0.95, seeded) and (T 0.6, top-p 0.95, unseeded) at c=1, then one c=4 pass mixing T=1 and T=0
      requests. Flags per output: SALAD (>= 5 CJK characters in a non-CJK task, or >= 3 U+FFFD), LOOP (the tail is
      one unit of <= 200 chars repeated >= 6 times), BADJSON (json tasks; warning only). Prints
      "TSCAN: <n> salad of <m> outputs". Outputs: ~/glm-final/LABEL/tscan.jsonl.
  prefill LABEL [--sizes 32768,131072] [--reps 2]
      sparkDash prefill-bench (cold, salted, c=1): one 4096 warm-up (discarded), then --reps rounds of --sizes.
      Prints "PREFILL <size> tps <median> ttft_ms <median> runs [...]".
  rigmark LABEL [--passes 4] [--effort low] [--protocol 1.0.0]
      RigMark decode screen without the A/B harness: --passes runs of the protocol checkout, decode only
      (--skip-prefill --skip-concurrency, 1 run each: code / prose / structured, temperature 0 as pinned by RigMark),
      median tok/s per workload over passes, plus the salad scan of every output. Not a conformant receipt.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import zlib
from urllib.parse import urlsplit

BASE = os.environ.get("GLM_BASE", "http://127.0.0.1:8093")


def base_port(base=None):
    """Port of the serving endpoint in ``base`` (default 8093, the fleet's standard port).

    rigmark_meta's ``meta`` subcommand looks the serving container up by its ``--port``,
    so a deployment on another port (e.g. 8888 for the sparkDash federation) needs this
    carried over from ``GLM_BASE`` instead of the 8093 default.
    """
    return urlsplit(BASE if base is None else base).port or 8093
MODEL = os.environ.get("GLM_MODEL", "GLM-5.3-Flash-FP8")
OUTROOT = os.path.expanduser("~/glm-final")


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "null")


# ------------------------------------------------------------------------------------------------ long KLD
def tokenize(text):
    for body in ({"model": MODEL, "prompt": text, "add_special_tokens": False}, {"model": MODEL, "prompt": text}):
        try:
            r = post("/tokenize", body, timeout=300)
            if r.get("tokens"):
                return r["tokens"]
        except Exception as exc:  # noqa: BLE001
            err = exc
    raise RuntimeError(f"/tokenize failed: {err!r}")


def lp_dict(d):
    """{token: logprob} from a vLLM logprob entry (dict of dicts or of floats)."""
    if d is None:
        return None
    return {str(k): (v["logprob"] if isinstance(v, dict) else float(v)) for k, v in d.items()}


def kld_collect(a):
    texts = json.load(open(a.texts))
    out = {"model": MODEL, "k": a.k, "gen": a.gen, "items": []}
    for t in texts:
        ids = tokenize(t["text"])
        n = min(len(ids), int(t["tokens"]))
        ids = ids[:n]
        rec = {"id": t["id"], "kind": t.get("kind", ""), "n_prompt": n, "target": int(t["tokens"])}
        t0 = time.time()
        r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": a.gen, "temperature": 0,
                                     "logprobs": a.k, "return_tokens_as_token_ids": True})
        rec["cont_s"] = round(time.time() - t0, 2)
        ch = r["choices"][0]
        lp = ch.get("logprobs") or {}
        rec["gen_tokens"] = lp.get("tokens") or []
        rec["gen_top"] = [lp_dict(x) for x in (lp.get("top_logprobs") or [])]
        rec["finish"] = ch.get("finish_reason")
        rec["usage"] = r.get("usage")
        t0 = time.time()
        r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0,
                                     "prompt_logprobs": a.k})
        rec["tf_s"] = round(time.time() - t0, 2)
        pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs") or []
        rec["prompt_lp"] = [lp_dict(x) for x in pl]
        out["items"].append(rec)
        print(f"  {t['id']:12s} prompt {n} tok (target {t['tokens']}) "
              f"continuation {rec['cont_s']} s ({len(rec['gen_tokens'])} tok, {rec['finish']}) "
              f"teacher-forced {rec['tf_s']} s ({len(rec['prompt_lp'])} positions)", flush=True)
    json.dump(out, open(a.out, "w"))
    short = [i["id"] for i in out["items"] if i["n_prompt"] < 16000]
    print(f"KLDLONG-COLLECT {a.out}: {len(out['items'])} texts, prompts {[i['n_prompt'] for i in out['items']]}"
          + (f"; SHORTER THAN 16k: {short}" if short else ""))


def kl_top(p, q):
    """KL(p || q) over p's top-k support plus one tail bucket; q entries missing from its top-k get q's floor."""
    if not p or not q:
        return None
    floor = min(q.values())
    kl, pm, qm = 0.0, 0.0, 0.0
    for tok, lp in p.items():
        lq = q.get(tok, floor)
        pp = math.exp(lp)
        kl += pp * (lp - lq)
        pm += pp
        qm += math.exp(lq)
    pt, qt = max(1e-12, 1 - pm), max(1e-12, 1 - min(qm, 1 - 1e-12))
    return max(0.0, kl + pt * math.log(pt / qt))


def _stats(xs):
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    return {"n": len(s), "mean": sum(s) / len(s), "p99": s[max(0, int(0.99 * len(s)) - 1)], "max": s[-1]}


def kld_compare(a):
    R, C = json.load(open(a.ref)), json.load(open(a.cand))
    pooled, tail_all, cont_all, top1, n_top = [], [], [], 0, 0
    rows = []
    for r, c in zip(R["items"], C["items"]):
        if r["id"] != c["id"] or r["n_prompt"] != c["n_prompt"]:
            print(f"MISMATCHED PANEL: {r['id']}/{r['n_prompt']} vs {c['id']}/{c['n_prompt']}")
            print("GATE long_kld nan FAIL 0")
            return 1
        kls, t1 = [], 0
        for p, q in zip(r["prompt_lp"], c["prompt_lp"]):
            k = kl_top(p, q)
            if k is None:
                continue
            kls.append(k)
            t1 += max(p, key=p.get) == max(q, key=q.get)
        tail = kls[-2048:]
        gr, gc = r["gen_tokens"], c["gen_tokens"]
        div = next((i for i, (u, v) in enumerate(zip(gr, gc)) if u != v), min(len(gr), len(gc)))
        ck = [k for k in (kl_top(p, q) for p, q in zip(r["gen_top"][:div + 1], c["gen_top"][:div + 1])) if k is not None]
        pooled += kls
        tail_all += tail
        cont_all += ck
        top1 += t1
        n_top += len(kls)
        s, st, sc = _stats(kls), _stats(tail), _stats(ck)
        rows.append(r["id"])
        print(f"{r['id']:12s} prompt {r['n_prompt']:6d}  tf KL mean {s.get('mean', float('nan')):.5f} p99 "
              f"{s.get('p99', float('nan')):.4f} top1 {100 * t1 / max(1, len(kls)):.2f} %  last-2048 KL "
              f"{st.get('mean', float('nan')):.5f}  continuation: identical {div}/{len(gr)} tokens"
              f"{' (full)' if gr == gc else ''}, KL to divergence {sc.get('mean', float('nan')):.5f} ({sc['n']} pos)")
    s = _stats(pooled)
    ok = s["n"] > 0 and s["mean"] <= a.max_kl
    print(f"pooled: {s['n']} teacher-forced positions, mean KL {s.get('mean', float('nan')):.5f}, p99 "
          f"{s.get('p99', float('nan')):.4f}, top-1 {100 * top1 / max(1, n_top):.2f} %; last-2048 mean "
          f"{_stats(tail_all).get('mean', float('nan')):.5f}; continuation mean {_stats(cont_all).get('mean', float('nan')):.5f}")
    print(f"GATE long_kld {s.get('mean', float('nan')):.5f} {'PASS' if ok else 'FAIL'} {s['n']}")
    print(f"GATE long_cont_kld {_stats(cont_all).get('mean', float('nan')):.5f} info {len(cont_all)}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------ T > 0 scan
PROMPTS = [
    ("prose", "low", "Explain how to decide whether a medium-sized software project is ready for a database migration. "
     "Cover dependencies, schema compatibility, tests, rollout and rollback in several paragraphs."),
    ("prose", "low", "Describe how a city could reduce traffic congestion over ten years without building new roads. "
     "Discuss pricing, public transport, zoning and the politics of each option."),
    ("think", "high", "A small town has one bridge that is closing for two years of repairs. The council must choose "
     "between a free ferry, a temporary pontoon bridge, and expanded bus service over a longer road. Reason carefully "
     "through the costs and risks of each option, then recommend one."),
    ("think", "high", "Three friends split a restaurant bill unevenly because one arrived late and ordered less, another "
     "paid the tip, and the third covered a previous taxi. Work out a fair way to settle up step by step."),
    ("code", "low", "Write a Python 3 module with a thread-safe bounded LRU cache with per-item TTL: get, set, delete, "
     "clear, __len__, injectable monotonic clock, plus five unittest tests. Return only code."),
    ("code", "low", "Write a Python function that parses an INI-like config format with sections, comments, multi-line "
     "values and typed getters, with docstrings and doctests. Return only code."),
    ("json", "low", "Return only a JSON object describing three fictional employees: name, role, start_date (ISO 8601), "
     "skills (array of strings), manager (name or null). No prose, no code fences."),
    ("json", "low", "Return only a JSON array of five cities with fields name, country, population (integer), "
     "coordinates {lat, lon} and a one-sentence note. No prose, no code fences."),
    ("polish", "low", "Napisz po polsku kilka akapitów o tym, jak przygotować się do pierwszego maratonu: plan "
     "treningowy, odżywianie, sprzęt i regeneracja."),
    ("story", "low", "Write a short story (about 500 words) about a lighthouse keeper who finds a message in a bottle "
     "written in her own handwriting."),
    ("schema", "low", "Give me a packing list for a three-day hiking trip as JSON."),
]
SCHEMA = {"type": "json_schema", "json_schema": {"name": "packing", "schema": {
    "type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "qty": {"type": "integer"}}, "required": ["name", "qty"]}}},
    "required": ["items"]}}}


def loop_tail(s, min_rep=6, max_unit=200):
    t = s[-1500:]
    if len(t) < 300:
        return False
    for u in range(1, max_unit + 1):
        unit = t[-u:]
        reps, i = 1, len(t) - 2 * u
        while i >= 0 and t[i:i + u] == unit:
            reps += 1
            i -= u
        if reps >= min_rep and reps * u >= 120:
            return True
    return False


def scan_text(kind, text):
    cjk = sum(0x4E00 <= ord(ch) <= 0x9FFF or 0x3040 <= ord(ch) <= 0x30FF or 0xAC00 <= ord(ch) <= 0xD7AF for ch in text)
    bad = text.count("�")
    flags = []
    if cjk >= 5 or bad >= 3:
        flags.append("SALAD")
    if loop_tail(text):
        flags.append("LOOP")
    return flags, cjk, bad


def json_ok(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        json.loads(t)
        return True
    except Exception:  # noqa: BLE001
        return False


def one_chat(kind, effort, prompt, temp, top_p, seed, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temp, "top_p": top_p, "chat_template_kwargs": {"reasoning_effort": effort}}
    if seed is not None:
        body["seed"] = seed
    if kind == "schema":
        body["response_format"] = SCHEMA
    t0 = time.time()
    r = post("/v1/chat/completions", body, timeout=900)
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = r.get("usage") or {}
    return {"kind": kind, "temp": temp, "top_p": top_p, "seed": seed, "finish": r["choices"][0].get("finish_reason"),
            "tokens": usage.get("completion_tokens"), "wall_s": round(time.time() - t0, 2),
            "content": content, "reasoning": reasoning}


def tscan(a):
    out_dir = os.path.join(OUTROOT, a.label)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tscan.jsonl")
    rows = []
    jobs = []
    for (temp, top_p, seeded) in ((1.0, 0.95, True), (0.6, 0.95, False)):
        for i, (kind, effort, prompt) in enumerate(PROMPTS):
            jobs.append(("c1", kind, effort, prompt, temp, top_p, (20260928 + i) if seeded else None))
    for tag, kind, effort, prompt, temp, top_p, seed in jobs:
        try:
            rows.append(dict(one_chat(kind, effort, prompt, temp, top_p, seed, a.max_tokens), phase=tag))
        except Exception as exc:  # noqa: BLE001
            rows.append({"phase": tag, "kind": kind, "temp": temp, "error": repr(exc)[:300]})
    # c = 4: every prompt at T = 1 plus two greedy requests, four at a time (mixed sampled / greedy batches)
    mix = [(k, e, p, 1.0, 0.95, 777 + i) for i, (k, e, p) in enumerate(PROMPTS)]
    mix.insert(3, ("prose", "low", PROMPTS[0][2], 0.0, 1.0, None))
    mix.insert(8, ("code", "low", PROMPTS[4][2], 0.0, 1.0, None))
    with cf.ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(one_chat, k, e, p, t, tp, s, a.max_tokens) for k, e, p, t, tp, s in mix]
        for f in futs:
            try:
                rows.append(dict(f.result(), phase="c4"))
            except Exception as exc:  # noqa: BLE001
                rows.append({"phase": "c4", "error": repr(exc)[:300]})
    salad = errors = badjson = 0
    with open(path, "w") as fo:
        for r in rows:
            if "error" in r:
                errors += 1
                r["flags"] = ["ERROR"]
            else:
                text = (r.get("reasoning") or "") + "\n" + (r.get("content") or "")
                flags, cjk, bad = scan_text(r["kind"], text if r["kind"] != "polish" else text)
                if r["kind"] in ("json", "schema") and r.get("finish") == "stop" and not json_ok(r.get("content") or ""):
                    flags.append("BADJSON")
                    badjson += 1
                r.update(flags=flags, cjk=cjk, fffd=bad)
                salad += any(f in ("SALAD", "LOOP") for f in flags)
            fo.write(json.dumps(r) + "\n")
            print(f"{r.get('phase', '?'):3s} {r.get('kind', '?'):7s} T={r.get('temp')} seed={r.get('seed')} "
                  f"tokens {r.get('tokens')} finish {r.get('finish')} {' '.join(r.get('flags', []))}", flush=True)
    print(f"TSCAN: {salad} salad of {len(rows)} outputs (errors {errors}, bad json {badjson}) -> {path}")
    return 0 if salad == 0 and errors == 0 else 1


# ------------------------------------------------------------------------------------------------ prefill
def prefill(a):
    url = a.dash.rstrip("/")
    sizes = [int(x) for x in a.sizes.split(",")]

    def run(sz):
        st = http(f"{url}?port={a.port}")
        if st and st.get("active"):
            raise RuntimeError(f"sparkDash prefill bench already active: {st.get('active')}")
        bid = http(url, {"port": a.port, "contextSizes": sz}, timeout=120)["benchId"]
        t0 = time.time()
        while time.time() - t0 < 3600:
            time.sleep(3)
            j = http(f"{url}/{bid}")
            if j and j.get("status") != "running":
                if j.get("status") != "completed":
                    raise RuntimeError(f"sparkDash job {bid} {j.get('status')}: {j.get('error')}")
                return j["results"]
        raise RuntimeError("sparkDash prefill job timeout")

    run([4096])  # warm-up, discarded
    res = {s: [] for s in sizes}
    for rep in range(a.reps):
        for r in run(sizes):
            s = int(r.get("targetTokens") or r.get("contextSize") or 0)
            if s in res and r.get("prefillTps"):
                res[s].append((float(r["prefillTps"]), float(r.get("ttftMs") or 0)))
        print(f"rep {rep}: {json.dumps({s: v[-1] if v else None for s, v in res.items()})}", flush=True)
    os.makedirs(os.path.join(OUTROOT, a.label), exist_ok=True)
    json.dump({str(s): v for s, v in res.items()}, open(os.path.join(OUTROOT, a.label, "prefill.json"), "w"))
    for s, v in res.items():
        if v:
            print(f"PREFILL {s} tps {statistics.median(x[0] for x in v):.0f} ttft_ms {statistics.median(x[1] for x in v):.0f} "
                  f"runs {[round(x[0]) for x in v]}")
        else:
            print(f"PREFILL {s} no result")
    return 0


# ------------------------------------------------------------------------------------------------ RigMark decode
def rigmark(a):
    root = os.path.expanduser("~/rigmark-glm")
    here = os.path.dirname(os.path.abspath(__file__))
    meta_py = os.path.join(here, "rigmark_meta.py")
    if not os.path.exists(meta_py):
        meta_py = os.path.join(root, "bin", "rigmark_meta.py")
    cli = os.path.join(root, f"rigmark-{a.protocol}")
    out = os.path.join(root, "final", a.label)
    os.makedirs(out, exist_ok=True)
    pre, meta = os.path.join(out, "preflight.json"), os.path.join(out, "metadata.json")
    with open(pre, "w") as f:
        subprocess.run([sys.executable, meta_py, "preflight", "--base", BASE, "--wait", "180"], stdout=f, check=True)
    subprocess.run([sys.executable, meta_py, "meta", "--base", BASE, "--port", str(base_port()),
                    "--preflight", pre, "--out", meta], check=True)
    model = json.load(open(pre))["model"]
    body = json.dumps({"chat_template_kwargs": {"reasoning_effort": a.effort}})
    per = {}
    salad = 0
    for p in range(a.passes):
        res_path = os.path.join(out, f"p{p}.json")
        cmd = ["./rigmark", "run", "--base-url", BASE, "--model", model, "--label", f"{a.label}-p{p}",
               "--comparison-id", f"{a.label}-p{p}", "--metadata", meta, "--extra-body", body,
               "--runs", "1", "--skip-prefill", "--skip-concurrency", "--output", res_path]
        r = subprocess.run(cmd, cwd=cli, capture_output=True, text=True)
        if not os.path.exists(res_path):
            print(f"pass {p}: rigmark failed rc={r.returncode}: {r.stderr[-400:]}", flush=True)
            continue
        dec = json.load(open(res_path)).get("decode", {})
        line = []
        for w, x in dec.items():
            for run in x.get("runs", []):
                per.setdefault(w, []).append(run.get("decode_tokens_per_second"))
                flags, cjk, bad = scan_text(w, run.get("output") or "")
                if "SALAD" in flags:
                    salad += 1
                line.append(f"{w} {run.get('decode_tokens_per_second')} ({run.get('completion_tokens')} tok, "
                            f"{run.get('finish_reason')}{' ' + ' '.join(flags) if flags else ''})")
        print(f"[{time.strftime('%H:%M:%S')}] pass {p}: " + "; ".join(line), flush=True)
    for w in ("code", "prose", "structured"):
        v = [x for x in per.get(w, []) if x]
        if v:
            print(f"RIGMARK {w} median {statistics.median(v):.1f} runs {[round(x, 1) for x in v]}")
    print(f"RIGMARK-SCAN: {salad} salad run(s)")
    json.dump({"label": a.label, "per": per, "salad": salad}, open(os.path.join(out, "summary.json"), "w"))
    return 0 if salad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    k = sp.add_parser("kldlong")
    ksp = k.add_subparsers(dest="sub", required=True)
    c = ksp.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--texts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "kld_long_texts.json"))
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--gen", type=int, default=256)
    m = ksp.add_parser("compare")
    m.add_argument("ref")
    m.add_argument("cand")
    m.add_argument("--max-kl", type=float, default=0.035)
    t = sp.add_parser("tscan")
    t.add_argument("label")
    t.add_argument("--max-tokens", type=int, default=768)
    p = sp.add_parser("prefill")
    p.add_argument("label")
    p.add_argument("--sizes", default="32768,131072")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--port", type=int, default=8093)
    p.add_argument("--dash", default="http://127.0.0.1:5555/api/sparks/spark-01/llm/prefill-bench")
    r = sp.add_parser("rigmark")
    r.add_argument("label")
    r.add_argument("--passes", type=int, default=4)
    r.add_argument("--effort", default="low")
    r.add_argument("--protocol", default="1.0.0")
    a = ap.parse_args()
    if a.cmd == "kldlong":
        return kld_collect(a) if a.sub == "collect" else kld_compare(a)
    return {"tscan": tscan, "prefill": prefill, "rigmark": rigmark}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
