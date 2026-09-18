"""Single-stream decode matrix for a variant. Run on Spark_01: python3 bench_matrix.py <label>
Four prompts (code, prose, json, agent) x two efforts, streaming, temperature 0. Records TTFT, decode tok/s, acceptance
from /metrics before/after. Agent prompt carries a ~6K-token system prompt + tool schemas to
mimic OpenCode. Writes <label>-matrix.json next to this file. Not an intelligence benchmark."""
import json,pathlib,sys,time,urllib.request,re
L=sys.argv[1];R=pathlib.Path(__file__).resolve().parent;BASE='http://127.0.0.1:8093';MODEL='GLM-5.3-Flash-FP8'
TOOLS=[{'type':'function','function':{'name':n,'description':d,'parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'},'command':{'type':'string'}},'required':[]}}} for n,d in [('read','Read a file'),('write','Write a file'),('edit','Edit a file by replacing a string'),('bash','Run a shell command'),('glob','Find files by pattern'),('grep','Search file contents')]]
SYS='You are an autonomous coding agent working inside a TypeScript monorepo. '+('Follow the repository conventions: strict TypeScript, no any, pnpm workspaces, vitest for tests, eslint flat config, conventional commits, feature flags via LaunchDarkly, tracing via OpenTelemetry. '*60)
P={
 'code':'Write a Python 3 module with a thread-safe bounded LRU cache with per-item TTL: get, set, delete, clear, __len__, injectable monotonic clock, plus five unittest tests. Return only code.',
 'prose':'Explain how to decide whether a medium-sized software project is ready for a database migration. Cover dependencies, schema compatibility, tests, rollout and rollback in several paragraphs.',
 'json':'Return a JSON array of 40 objects, each with keys id (int), sku (string like "SKU-00001"), price (float), tags (3 strings). No prose, no code fence.',
 'agent':'The test suite fails with "TypeError: Cannot read properties of undefined (reading map)" in packages/api/src/routes/orders.ts line 88. Investigate and fix it. Use the tools.',
}
def metrics():
    t=urllib.request.urlopen(BASE+'/metrics',timeout=10).read().decode();out={}
    for k in ['spec_decode_num_accepted_tokens_total','spec_decode_num_draft_tokens_total','spec_decode_num_drafts_total']:
        m=re.search(r'^vllm:'+k+r'(?:\{[^}]*\})?\s+([0-9.e+]+)',t,re.M);out[k]=float(m.group(1)) if m else None
    return out
def run(name,effort):
    msgs=[{'role':'user','content':P[name]}]
    if name=='agent':msgs=[{'role':'system','content':SYS}]+msgs
    body=dict(model=MODEL,messages=msgs,temperature=0,max_tokens=1536,stream=True,stream_options={'include_usage':True},chat_template_kwargs={'reasoning_effort':effort})
    if name=='agent':body['tools']=TOOLS
    m0=metrics();req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    t0=time.monotonic();first=last=None;usage=None;n=0
    with urllib.request.urlopen(req,timeout=1200) as r:
        for line in r:
            if not line.startswith(b'data: '):continue
            p=line[6:].strip()
            if p==b'[DONE]':break
            e=json.loads(p);now=time.monotonic()
            if e.get('usage'):usage=e['usage']
            for c in e.get('choices',[]):
                d=c.get('delta',{})
                if d.get('content') or d.get('reasoning_content') or d.get('reasoning') or d.get('tool_calls'):
                    if first is None:first=now
                    last=now
    m1=metrics();ct=usage['completion_tokens'] if usage else None
    acc=None
    if m0['spec_decode_num_draft_tokens_total'] is not None and m1['spec_decode_num_draft_tokens_total']:
        dd=m1['spec_decode_num_draft_tokens_total']-m0['spec_decode_num_draft_tokens_total'];da=m1['spec_decode_num_accepted_tokens_total']-m0['spec_decode_num_accepted_tokens_total'];dn=m1['spec_decode_num_drafts_total']-m0['spec_decode_num_drafts_total']
        acc=dict(draft_acceptance=round(da/dd,3) if dd else None,mean_accepted_len=round(da/dn,2) if dn else None)
    res=dict(variant=L,test=name,effort=effort,prompt_tokens=usage and usage['prompt_tokens'],completion_tokens=ct,ttft_s=first and round(first-t0,2),decode_tps=(ct and last and first and last>first) and round((ct-1)/(last-first),1),total_s=round(time.monotonic()-t0,1),acceptance=acc)
    print(json.dumps(res,ensure_ascii=False),flush=True);return res
results=[run(n,e) for e in ['high','low'] for n in P]
(R/(L+'-matrix.json')).write_text(json.dumps(results,indent=1,ensure_ascii=False))
