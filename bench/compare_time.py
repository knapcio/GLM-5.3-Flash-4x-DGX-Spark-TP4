"""Paired time-to-task comparison of two hard-set runs.
python3 compare_time.py runs/N8/N8-hardset.json runs/DS/DS-hardset.json
Prints per-category wall seconds, completion tokens, reasoning share and the per-task ratio; then writes a
blind review file (answers shuffled A/B per task) for a human or model judge."""
import json,sys,random,pathlib
A=json.load(open(sys.argv[1])); B=json.load(open(sys.argv[2])); nA=pathlib.Path(sys.argv[1]).stem; nB=pathlib.Path(sys.argv[2]).stem
a={x['id']:x for x in A}; b={x['id']:x for x in B}; ids=[i for i in a if i in b]
def agg(rows): return sum(r['secs'] for r in rows), sum(r['completion_tokens'] or 0 for r in rows), sum(r['reasoning_chars'] for r in rows)//4
cats=sorted(set(a[i]['category'] for i in ids), key=lambda c:[a[i]['category'] for i in ids].index(c))
print(f"{'category':10} | {nA:>22} | {nB:>22} | time ratio")
print(f"{'':10} | {'secs  tok  think':>22} | {'secs  tok  think':>22} |")
for c in cats+['ALL']:
    rows=[i for i in ids if c=='ALL' or a[i]['category']==c]
    sa,ta,ra=agg([a[i] for i in rows]); sb,tb,rb=agg([b[i] for i in rows])
    print(f"{c:10} | {sa:5.0f} {ta:5d} {ra:5d} | {sb:5.0f} {tb:5d} {rb:5d} | {sb/sa if sa else 0:.2f}x")
faster=sum(1 for i in ids if b[i]['secs']<a[i]['secs']); print(f"\n{nB} faster on {faster}/{len(ids)} tasks; capped: {nA} {sum(1 for i in ids if a[i]['finish']=='length')}, {nB} {sum(1 for i in ids if b[i]['finish']=='length')}")
random.seed(7); blind=[]; key={}
for i in ids:
    pair=[(nA,a[i]['answer']),(nB,b[i]['answer'])]; random.shuffle(pair); key[i]={'A':pair[0][0],'B':pair[1][0]}
    blind.append(f"### {i} ({a[i]['category']})\n\n#### A\n{pair[0][1]}\n\n#### B\n{pair[1][1]}\n")
pathlib.Path('blind_review.md').write_text('\n'.join(blind)); pathlib.Path('blind_key.json').write_text(json.dumps(key,indent=1)); print('wrote blind_review.md (judge A/B per task) and blind_key.json')
