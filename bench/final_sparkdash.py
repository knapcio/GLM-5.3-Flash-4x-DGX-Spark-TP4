#!/usr/bin/env python3
"""Strict sparkDash final-stack matrix, same cells/requests as glm_prodbench.sh.

Adds two prose c1 and two prose c4 samples to reach scored n=5/n=3.
No variant switching, server edits, quality claims or automatic retries.
Credit: glm_prodbench.sh and sparkDash DecodeBench contributors.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request


def cells():
    result = [('warmup', 'prose', 1)] * 2
    result += [('matrix', 'prose', 1)] * 3 + [('matrix', 'code', 1)]
    result += [('matrix', 'prose', c) for c in (2, 4, 8, 16)]
    result += [('matrix', 'code', c) for c in (4, 16)]
    result += [('matrix', kind, c) for kind in ('structured', 'json') for c in (1, 16)]
    result += [('supplement', 'prose', 1)] * 2 + [('supplement', 'prose', 4)] * 2
    return result


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_job(job, bench_id, kind, concurrency, model='GLM-5.3-Flash-FP8'):
    errors = []
    if job.get('benchId') != bench_id:
        errors.append('job identity mismatch')
    if job.get('sparkId') != 'spark-01':
        errors.append('spark identity mismatch')
    if job.get('status') != 'completed' or job.get('error') is not None:
        errors.append('job did not complete without error')
    config = job.get('config') or {}
    for key, expected in dict(port=8093, promptType=kind, concurrencies=[concurrency], maxTokens=256, modelId=model).items():
        if config.get(key) != expected:
            errors.append(f'config mismatch: {key}')
    start, end = job.get('startedAt'), job.get('completedAt')
    if not positive(start) or not positive(end) or end < start:
        errors.append('invalid job timestamps')
    progress = job.get('progress') or {}
    if progress.get('completedLevels') != 1 or progress.get('totalLevels') != 1:
        errors.append('incomplete progress levels')
    results = job.get('results') or []
    if len(results) != 1:
        return errors + ['expected exactly one result']
    wave = results[0]
    for key, expected in dict(concurrency=concurrency, streamsOk=concurrency, streamsFailed=0).items():
        if type(wave.get(key)) is not int or wave[key] != expected:
            errors.append(f'wave mismatch: {key}')
    if wave.get('error') is not None:
        errors.append('wave error')
    for key in ('meanDecodeTps', 'aggregateDecodeTps'):
        if not positive(wave.get(key)):
            errors.append(f'invalid {key}')
    streams = wave.get('streams') or []
    if len(streams) != concurrency:
        errors.append('stream count mismatch')
    if [s.get('index') for s in streams] != list(range(concurrency)):
        errors.append('stream indices mismatch')
    for stream in streams:
        if stream.get('error') is not None:
            errors.append('stream error')
        if stream.get('reasoningChunks') != 0:
            errors.append('thinking-off request produced reasoning or missing counter')
        # Actual public field, DecodeBench.js streamPublicResult(). No early-EOS rescue.
        if type(stream.get('completionTokens')) is not int or stream['completionTokens'] != 256:
            errors.append('stream completionTokens must equal256')
        if type(stream.get('decodeTokens')) is not int or stream['decodeTokens'] != 255:
            errors.append('stream decodeTokens must equal255')
        if not positive(stream.get('decodeTps')):
            errors.append('invalid stream decode measurement')
    if type(wave.get('totalCompletionTokens')) is not int or wave['totalCompletionTokens'] != concurrency * 256:
        errors.append('total completion token count mismatch')
    if type(wave.get('totalDecodeTokens')) is not int or wave['totalDecodeTokens'] != concurrency * 255:
        errors.append('total decode token count mismatch')
    return errors


def http_json(url, body=None, timeout=30):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode()
        return response.status, json.loads(raw)


def collect_job(base, kind, concurrency, timeout, record, save, seen_ids=None,
                model='GLM-5.3-Flash-FP8', http=http_json, clock=time.monotonic, sleep=time.sleep):
    record['request'] = dict(port=8093, concurrencies=[concurrency], maxTokens=256,
                             promptType=kind, modelId=model)
    status, posted = http(base + '/bench', record['request'], timeout=min(30, timeout))
    record['post_status'], record['post_job'] = status, posted
    save()
    if status != 202 or not isinstance(posted.get('benchId'), str) or not posted['benchId']:
        raise RuntimeError('POST did not return202 and a benchId')
    if (posted.get('status') != 'running' or posted.get('error') is not None
            or posted.get('sparkId') != 'spark-01' or not positive(posted.get('startedAt'))
            or posted.get('completedAt') is not None or posted.get('results') != []
            or any((posted.get('config') or {}).get(k) != v for k, v in record['request'].items())
            or (posted.get('progress') or {}).get('completedLevels') != 0
            or (posted.get('progress') or {}).get('totalLevels') != 1):
        raise RuntimeError('POST did not return a fresh running job with requested config')
    bench_id = posted['benchId']
    if seen_ids is not None:
        if bench_id in seen_ids:
            raise RuntimeError('POST reused an already observed benchId')
        seen_ids.add(bench_id)
    deadline = clock() + timeout
    while True:
        left = deadline - clock()
        if left <= 0:
            raise TimeoutError(f'job {bench_id} exceeded deadline; no automatic cancellation')
        status, job = http(base + '/bench/' + urllib.parse.quote(bench_id, safe=''),
                           timeout=min(30, left))
        record['last_status'], record['job'] = status, job
        save()
        if status != 200 or job.get('benchId') != bench_id:
            raise RuntimeError('GET status or job identity mismatch')
        if job.get('startedAt') != posted.get('startedAt'):
            raise RuntimeError('job startedAt changed between POST and GET')
        if job.get('status') != 'running':
            errors = validate_job(job, bench_id, kind, concurrency, model)
            record['validation_errors'] = errors
            save()
            if errors:
                raise RuntimeError('; '.join(errors))
            return job
        sleep(min(2, max(0, deadline - clock())))


def summarize(records):
    groups = collections.defaultdict(list)
    for record in records:
        if record['phase'] != 'warmup' and record.get('valid'):
            groups[(record['kind'], record['concurrency'])].append(record['job']['results'][0])
    return {f'{kind}:c{c}': dict(n=len(rows), meanDecodeTps_median=statistics.median(r['meanDecodeTps'] for r in rows),
                                aggregateDecodeTps_median=statistics.median(r['aggregateDecodeTps'] for r in rows))
            for (kind, c), rows in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label')
    ap.add_argument('--base', default='http://127.0.0.1:5555/api/sparks/spark-01/llm')
    ap.add_argument('--job-timeout', type=float, default=900)
    ap.add_argument('--model', default='GLM-5.3-Flash-FP8')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not math.isfinite(args.job_timeout) or args.job_timeout <= 0:
        ap.error('job-timeout must be finite and positive')
    report = dict(bench='sparkDash DecodeBench', label=args.label,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  planned_cells=cells(), status='RUNNING', complete=False, records=[])
    with args.out.open('x'):
        pass
    def save():
        report['summary'] = summarize(report['records'])
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        status, state = http_json(args.base.rstrip('/') + '/bench')
        report['preflight'] = state
        if status != 200 or state.get('active'):
            raise RuntimeError('sparkDash already active or preflight failed')
        seen_ids = {j['benchId'] for j in [state.get('last'), *(state.get('history') or [])]
                    if isinstance(j, dict) and isinstance(j.get('benchId'), str)}
        for index, (phase, kind, concurrency) in enumerate(cells()):
            record = dict(index=index, phase=phase, kind=kind, concurrency=concurrency, valid=False)
            report['records'].append(record)
            save()
            collect_job(args.base.rstrip('/'), kind, concurrency, args.job_timeout, record, save,
                        seen_ids=seen_ids, model=args.model)
            record['valid'] = True
            save()
            print(json.dumps(dict(index=index, phase=phase, kind=kind, concurrency=concurrency,
                                  benchId=record['job']['benchId'], result=record['job']['results'][0])), flush=True)
            # API start quota: at most one start per3 seconds. No retry loop on429.
            time.sleep(3.1)
        report['status'], report['complete'] = 'COMPLETE_VALID_MEASUREMENT', True
    except Exception as exc:
        report['status'], report['error'] = 'FAIL', f'{type(exc).__name__}: {exc}'
        if isinstance(exc, urllib.error.HTTPError):
            report['http_error_body'] = exc.read().decode(errors='replace')
        save()
        return 2
    save()
    print(json.dumps(report['summary']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
