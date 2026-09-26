"""Fixed current-stack prefill probe: two discarded warmups, three scored cold requests each at16k/32k/64k."""
import argparse, hashlib, json, statistics, uuid
from pathlib import Path
import prefill_checked as p


def cold_usage(record):
    usage=record['usage']
    values=[]
    if 'cached_tokens' in usage:values.append(usage['cached_tokens'])
    details=usage.get('prompt_tokens_details')
    if details is not None:
        if not isinstance(details,dict):raise ValueError('invalid prompt token details')
        if 'cached_tokens' in details:values.append(details['cached_tokens'])
    if any(type(v) is not int or v!=0 for v in values):raise ValueError('cold request has cached/invalid cached tokens')
    return bool(values)


def run(base,model,result):
    sizes=(16384,32768,65536)
    bodies={size:p.build(base,model,size,size) for size in sizes}
    result.update(bench='prefill-checked-client-first-content-token',sizes=list(sizes),warmups=2,repeat=3,base=base,model=model,body_sha256={str(n):hashlib.sha256(x.encode()).hexdigest() for n,x in bodies.items()})
    schedule=[('warmup',32768)]*2+[("scored",size) for size in sizes for _ in range(3)]
    for i,(phase,size) in enumerate(schedule):
        nonce=uuid.uuid4().hex
        row={'index':i,'phase':phase,'size':size,'nonce':nonce}
        result['records'].append(row)
        prompt=f'[{nonce}] Read the numbered notes below and reply with one word.\n'+bodies[size]
        p.stream(base,model,prompt,row)
        row['cached_tokens_reported']=cold_usage(row)
        print(json.dumps({k:row[k] for k in ('index','phase','size','prompt_tokens','ttft_s','prefill_tps','cached_tokens_reported')}),flush=True)
    summary=[]
    for size in sizes:
        scored=[x for x in result['records'] if x['phase']=='scored' and x['size']==size]
        summary.append(dict(size=size,n=3,median_prefill_tps=statistics.median(x['prefill_tps'] for x in scored),median_ttft_s=statistics.median(x['ttft_s'] for x in scored),prompt_tokens=[x['prompt_tokens'] for x in scored]))
    result.update(status='PASS',summary=summary)


def main():
    a=argparse.ArgumentParser();a.add_argument('--base',default='http://127.0.0.1:8093');a.add_argument('--model',default='GLM-5.3-Flash-FP8');a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    with args.out.open('x') as f:
        result={'status':'FAIL','records':[],'sources':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('run_prefill.py','prefill_checked.py')}}
        try:run(args.base,args.model,result)
        except BaseException as exc:result['error']=repr(exc);raise
        finally:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
