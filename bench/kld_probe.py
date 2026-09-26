#!/usr/bin/env python3
"""Cheap fidelity probe for a quantization arm, run against the live endpoint (no extra GPU memory).

  collect: for every calibration text, ask the server for teacher-forced prompt logprobs (top-K per position)
           and a greedy continuation with top-K logprobs. One file per arm.
  compare: arm vs reference file -> mean KL(ref || arm) over the reference's top-K support (tail folded into
           one bucket), top-1 agreement, and for the greedy continuation the first divergent token and the
           exact-match rate. Teacher-forced numbers are the clean ones: both arms score the same text.

    python3 kld_probe.py collect --url http://127.0.0.1:8093 --model GLM-5.3-Flash-FP8 --texts calib.json --out A.json
    python3 kld_probe.py compare REF.json ARM.json

If the server rejects prompt_logprobs (some speculative-decoding builds do), collect falls back to greedy only
and says so; the gate then rests on greedy agreement plus bench/qeval.py.
"""
import argparse, json, math, sys, time, urllib.request


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def collect(a):
    texts = json.load(open(a.texts))
    out = {"model": a.model, "k": a.k, "items": []}
    use_prompt_lp = True
    for i, t in enumerate(texts):
        rec = {"id": t.get("id", i), "kind": t.get("kind", "")}
        if use_prompt_lp:
            try:
                r = post(a.url + "/v1/completions", {"model": a.model, "prompt": t["text"], "max_tokens": 1,
                                                    "temperature": 0, "prompt_logprobs": a.k, "logprobs": a.k})
                pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
                rec["prompt_lp"] = [None if d is None else {k: (v["logprob"] if isinstance(v, dict) else v)
                                                            for k, v in d.items()} for d in pl]
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"prompt_logprobs unavailable ({e!r:.160}); greedy only\n")
                use_prompt_lp = False
        r = post(a.url + "/v1/completions", {"model": a.model, "prompt": t.get("prompt", t["text"][:600]),
                                            "max_tokens": a.gen, "temperature": 0, "logprobs": a.k})
        lp = r["choices"][0]["logprobs"]
        rec["gen_tokens"] = lp["tokens"]
        rec["gen_top"] = lp["top_logprobs"]
        rec["gen_text"] = r["choices"][0]["text"]
        out["items"].append(rec)
        print(f"  {i+1}/{len(texts)} {rec['kind']:8s} prompt_lp={'yes' if 'prompt_lp' in rec else 'no'}", flush=True)
    json.dump(out, open(a.out, "w"))


def kl_top(p, q):
    """KL(p||q) over p's top-K support plus one tail bucket; q entries missing from its own top-K get q's floor."""
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
    return kl + pt * math.log(pt / qt)


def compare(a):
    R, A = json.load(open(a.ref)), json.load(open(a.arm))
    kls, top1, n = [], 0, 0
    first_div, exact = [], 0
    for r, x in zip(R["items"], A["items"]):
        if "prompt_lp" in r and "prompt_lp" in x:
            for p, q in zip(r["prompt_lp"], x["prompt_lp"]):
                if p and q:
                    k = kl_top(p, q)
                    if k is not None:
                        kls.append(k)
                        top1 += max(p, key=p.get) == max(q, key=q.get)
                        n += 1
        gr, ga = r["gen_tokens"], x["gen_tokens"]
        d = next((i for i, (u, v) in enumerate(zip(gr, ga)) if u != v), min(len(gr), len(ga)))
        first_div.append(d)
        exact += gr == ga
    if kls:
        kls.sort()
        print(f"teacher-forced: positions {n}  mean KL {sum(kls)/len(kls):.5f}  p99 {kls[int(0.99*len(kls))-1]:.4f}  "
              f"top-1 agreement {top1/n*100:.2f} %")
    else:
        print("teacher-forced: not available (no prompt_logprobs in one of the files)")
    fd = sorted(first_div)
    print(f"greedy: {exact}/{len(first_div)} continuations identical; first divergence median {fd[len(fd)//2]} "
          f"min {fd[0]} tokens")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--url", default="http://127.0.0.1:8093"); c.add_argument("--model", required=True)
    c.add_argument("--texts", required=True); c.add_argument("--out", required=True)
    c.add_argument("--k", type=int, default=20); c.add_argument("--gen", type=int, default=256)
    m = sp.add_parser("compare"); m.add_argument("ref"); m.add_argument("arm")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else compare(a)


if __name__ == "__main__":
    main()
