"""Concurrent streaming decode benchmark for a GLM endpoint on Spark_01.

python3 conc_bench.py <label> [--base http://127.0.0.1:8093] [--model NAME] [--conc 1,2,3] [--types prose,code,json]
Each cell: N identical-type prompts (distinct prefixes so prefix cache does not help) started on a barrier,
temperature 0, thinking off, max_tokens 768. Reports per-stream decode tok/s (mean/min), aggregate, TTFT.
Writes <label>-conc.json next to this file. Not an intelligence benchmark.
"""
import argparse, concurrent.futures, json, pathlib, statistics, threading, time, urllib.request

P = {
 'prose': 'Explain how to decide whether a medium-sized software project is ready for a database migration. Cover dependencies, schema compatibility, tests, rollout and rollback in several paragraphs.',
 'code': 'Write a Python 3 module with a thread-safe bounded LRU cache with per-item TTL: get, set, delete, clear, __len__, injectable monotonic clock, plus five unittest tests. Return only code.',
 'json': 'Return a JSON array of 40 objects, each with keys id (int), sku (string like "SKU-00001"), price (float), tags (3 strings). No prose, no code fence.',
}

def one(base, model, kind, idx, max_tokens, barrier, thinking):
    prefix = 'Request %d of a batch. ' % (idx + 1)
    body = dict(model=model, messages=[{'role': 'user', 'content': prefix + P[kind]}], temperature=0, top_p=1,
                max_tokens=max_tokens, stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'reasoning_effort': thinking})
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    barrier.wait()
    t0 = time.monotonic(); first = last = None; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b'data: '): continue
            p = line[6:].strip()
            if p == b'[DONE]': break
            e = json.loads(p); now = time.monotonic()
            if e.get('usage'): usage = e['usage']
            for c in e.get('choices', []):
                d = c.get('delta', {})
                if d.get('content') or d.get('reasoning_content') or d.get('reasoning'):
                    if first is None: first = now
                    last = now
    ct = usage['completion_tokens'] if usage else 0
    tps = (ct - 1) / (last - first) if (ct > 1 and last and first and last > first) else None
    return dict(kind=kind, idx=idx, completion_tokens=ct, ttft_s=first and round(first - t0, 2), decode_tps=tps and round(tps, 1), wall_s=round(time.monotonic() - t0, 1))

def cell(base, model, kind, n, max_tokens, thinking):
    barrier = threading.Barrier(n)
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(lambda i: one(base, model, kind, i, max_tokens, barrier, thinking), range(n)))
    tps = [r['decode_tps'] for r in rs if r['decode_tps']]
    wall = max(r['wall_s'] for r in rs); toks = sum(r['completion_tokens'] for r in rs)
    out = dict(kind=kind, conc=n, per_stream_mean=round(statistics.mean(tps), 1) if tps else None, per_stream_min=round(min(tps), 1) if tps else None,
               aggregate=round(sum(tps), 1) if tps else None, wall_aggregate=round(toks / wall, 1) if wall else None,
               ttft_mean=round(statistics.mean(r['ttft_s'] for r in rs if r['ttft_s']), 2), ttft_max=max(r['ttft_s'] for r in rs if r['ttft_s']),
               tokens=toks, streams=rs)
    print(json.dumps({k: v for k, v in out.items() if k != 'streams'}), flush=True)
    return out

if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('label'); a.add_argument('--base', default='http://127.0.0.1:8093'); a.add_argument('--model', default='GLM-5.3-Flash-FP8')
    a.add_argument('--conc', default='1,2,3'); a.add_argument('--types', default='prose,code,json'); a.add_argument('--max-tokens', type=int, default=768)
    a.add_argument('--thinking', default='low', help='reasoning_effort low|high|max (template has no off)'); a.add_argument('--repeat', type=int, default=1)
    a = a.parse_args()
    res = []
    for _ in range(a.repeat):
        for n in [int(x) for x in a.conc.split(',')]:
            for k in a.types.split(','):
                res.append(cell(a.base, a.model, k, n, a.max_tokens, a.thinking))
    pathlib.Path(__file__).resolve().parent.joinpath(a.label + '-conc.json').write_text(json.dumps(res, indent=1))
