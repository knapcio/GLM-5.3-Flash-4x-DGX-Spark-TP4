#!/usr/bin/env python3
"""Direct-endpoint nonce-cold prefill-derived rate; parent schedules all requests.

Credits: glm53-flash-4x-spark/bench/prefill_bench.py prompt generator/workload.
Uses stdlib only: run python3 -S to avoid unrelated serving startup hooks.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import urllib.request
import uuid

WORDS = ('system memory network cache latency throughput kernel thread process scheduler compiler garden river mountain '
         'village market harbour teacher student library museum theatre orchestra painter novel poem history economy '
         'policy budget contract engineer doctor patient hospital battery engine turbine bridge tunnel railway airport '
         'weather forecast season harvest winter summer autumn spring ocean island forest desert canyon glacier').split()


def text_block(rng, n_par):
    out = []
    for i in range(n_par):
        sents = []
        for _ in range(rng.randint(3, 6)):
            w = [rng.choice(WORDS) for _ in range(rng.randint(8, 18))]
            sents.append(' '.join(w).capitalize() + '.')
        out.append(f'{i + 1}. ' + ' '.join(sents))
    return '\n'.join(out)


def positive_int(value):
    return type(value) is int and value > 0


def parse_wire_lines(wire):
    """Reconstruct complete SSE data events and receipt times from exact bytes."""
    parts, events = [], []
    previous = 0.0
    for row in wire:
        elapsed = row['elapsed_s']
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < previous:
            raise ValueError('invalid/nonmonotonic wire timestamp')
        previous = elapsed
        raw = base64.b64decode(row['base64'], validate=True)
        if not raw.endswith(b'\n') or b'\n' in raw[:-1]:
            raise ValueError('wire row is not one complete SSE line')
        line = raw.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            parts.append(line[5:].lstrip(' '))
        elif not line and parts:
            events.append(dict(elapsed_s=elapsed, data='\n'.join(parts)))
            parts = []
    if parts:
        raise ValueError('unterminated SSE event')
    return events


def validate_events(events):
    """Validate retained SSE data messages; clocks are elapsed client seconds."""
    first_choice = first_token = usage = finish = None
    done = False
    previous = 0.0
    for event in events:
        elapsed = event['elapsed_s']
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < previous:
            raise ValueError('invalid/nonmonotonic SSE timestamp')
        previous = elapsed
        if done:
            raise ValueError('SSE data after DONE')
        data = event['data']
        if data == '[DONE]':
            done = True
            continue
        obj = json.loads(data)
        if not isinstance(obj, dict) or obj.get('error') is not None:
            raise ValueError('stream error/nonobject')
        if obj.get('usage') is not None:
            if usage is not None:
                raise ValueError('duplicate usage event')
            usage = obj['usage']
        choices = obj.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise ValueError('invalid choice count')
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or type(choice.get('index')) is not int or choice['index'] != 0:
                raise ValueError('invalid choice index')
            if finish is not None:
                raise ValueError('choice after terminal finish')
            if first_choice is None:
                first_choice = elapsed
            delta = choice.get('delta')
            if not isinstance(delta, dict):
                raise ValueError('missing delta')
            values = [delta.get(k) for k in ('content', 'reasoning', 'reasoning_content')]
            if any(v is not None and not isinstance(v, str) for v in values):
                raise ValueError('invalid token delta')
            if first_token is None and any(isinstance(v, str) and len(v) > 0 for v in values):
                first_token = elapsed
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
                if finish not in ('length', 'stop'):
                    raise ValueError('unexpected finish reason')
    if not done or finish is None or first_token is None or first_token <= 0:
        raise ValueError('missing DONE, terminal finish, or observable generated token')
    if not isinstance(usage, dict) or not positive_int(usage.get('prompt_tokens')):
        raise ValueError('missing/invalid prompt usage')
    if type(usage.get('completion_tokens')) is not int or usage['completion_tokens'] != 1:
        raise ValueError('expected exactly one generated token')
    pt = usage['prompt_tokens']
    if type(usage.get('total_tokens')) is not int or usage['total_tokens'] != pt + 1:
        raise ValueError('inconsistent total token usage')
    return dict(prompt_tokens=pt, completion_tokens=1, ttft_s=first_token,
                first_choices_s=first_choice, prefill_tps=pt / first_token,
                finish_reason=finish, usage=usage)


def post(base, path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def ntok(base, model, text):
    with post(base, '/tokenize', {'model': model, 'prompt': text}) as response:
        obj = json.load(response)
    count = obj.get('count')
    if count is None and isinstance(obj.get('tokens'), list):
        count = len(obj['tokens'])
    if not positive_int(count):
        raise ValueError('invalid server tokenize count')
    return count


def build(base, model, target, seed):
    body = text_block(random.Random(seed), 40)
    per = ntok(base, model, body) / 40
    return text_block(random.Random(seed), max(1, int((target - 64) / per)))


def stream(base, model, prompt, record):
    body = dict(model=model, messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1, temperature=0, stream=True,
                stream_options={'include_usage': True},
                chat_template_kwargs={'reasoning_effort': 'low'})
    record['request'] = body
    record['events'] = []
    record['wire_lines'] = []
    started = time.monotonic()
    try:
        with post(base, '/v1/chat/completions', body) as response:
            record['http_status'] = response.status
            if response.status != 200:
                raise ValueError('unexpected HTTP status')
            for raw in response:
                elapsed = time.monotonic() - started
                record['wire_lines'].append(dict(elapsed_s=elapsed,
                                                 base64=base64.b64encode(raw).decode()))
        record['events'] = parse_wire_lines(record['wire_lines'])
        record.update(validate_events(record['events']))
        record['valid'] = True
    except BaseException as exc:
        record['valid'] = False
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise


def validate_report(report, sizes, repeat, source_sha256):
    if (report.get('schema') != 1 or report.get('status') != 'PASS'
            or report.get('sizes') != sizes or report.get('repeat') != repeat
            or report.get('source_sha256') != source_sha256):
        raise ValueError('report status/schema/workload/source mismatch')
    rows = report.get('rows', [])
    if len(rows) != len(sizes) or [r.get('size') for r in rows] != sizes:
        raise ValueError('incomplete/duplicate size rows')
    nonces = set()
    summary = []
    for row in rows:
        cold = row.get('cold', [])
        if len(cold) != repeat or [r.get('index') for r in cold] != list(range(repeat)):
            raise ValueError('incomplete/duplicate cold sample coordinates')
        for record in cold + [row.get('replay', {})]:
            if record.get('valid') is not True or record.get('http_status') != 200:
                raise ValueError('invalid request record')
            if parse_wire_lines(record['wire_lines']) != record['events']:
                raise ValueError('parsed events differ from retained wire bytes/timestamps')
            observed = validate_events(record['events'])
            if any(record.get(k) != v for k, v in observed.items()):
                raise ValueError('stored metrics differ from SSE evidence')
            request = record['request']
            if (request.get('model') != report['model'] or request.get('max_tokens') != 1
                    or request.get('temperature') != 0 or request.get('stream') is not True
                    or request.get('stream_options') != {'include_usage': True}
                    or request.get('chat_template_kwargs') != {'reasoning_effort': 'low'}):
                raise ValueError('request workload changed')
            messages = request.get('messages')
            if not isinstance(messages, list) or len(messages) != 1 or messages[0].get('role') != 'user':
                raise ValueError('request messages changed')
            nonce = record.get('nonce')
            if not isinstance(nonce, str) or len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce):
                raise ValueError('invalid nonce')
            prefix = f'[{nonce}] Read the numbered notes below and reply with one word.\n'
            prompt = messages[0]['content']
            if not prompt.startswith(prefix) or hashlib.sha256(prompt[len(prefix):].encode()).hexdigest() != row['body_sha256']:
                raise ValueError('prompt/nonce/body evidence mismatch')
        for record in cold:
            if record.get('kind') != 'cold' or record['nonce'] in nonces:
                raise ValueError('cold nonce reused or kind changed')
            nonces.add(record['nonce'])
        replay = row['replay']
        if replay.get('kind') != 'replay' or replay['request'] != cold[-1]['request'] or replay['nonce'] != cold[-1]['nonce']:
            raise ValueError('replay is not the final exact cold request')
        tps = statistics.median(r['prefill_tps'] for r in cold)
        ttft = statistics.median(r['ttft_s'] for r in cold)
        if row.get('cold_median_tps') != tps or row.get('cold_median_ttft_s') != ttft:
            raise ValueError('cold median includes wrong samples')
        summary.append(dict(size=row['size'], samples=repeat, cold_median_tps=tps,
                            cold_median_ttft_s=ttft,
                            prompt_tokens=[r['prompt_tokens'] for r in cold],
                            replay_tps=replay['prefill_tps'], body_sha256=row['body_sha256']))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', default='http://127.0.0.1:8093')
    parser.add_argument('--model', default='GLM-5.3-Flash-FP8')
    parser.add_argument('--sizes', default='8192,32768,65536')
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]
    if not sizes or len(set(sizes)) != len(sizes) or any(s <= 64 for s in sizes) or args.repeat < 1:
        parser.error('unique sizes >64 and positive repeat required')
    report = dict(schema=1, status='RUNNING', label=args.label, base=args.base,
                  model=args.model, sizes=sizes, repeat=args.repeat,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  metric='prompt_tokens/client_first_observable_token_latency', rows=[])
    with args.out.open('x') as file:
        json.dump(report, file)
    def save():
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    try:
        for size in sizes:
            body = build(args.base, args.model, size, size)
            row = dict(size=size, body_sha256=hashlib.sha256(body.encode()).hexdigest(), cold=[])
            report['rows'].append(row)
            for index in range(args.repeat):
                nonce = uuid.uuid4().hex
                prompt = f'[{nonce}] Read the numbered notes below and reply with one word.\n' + body
                record = dict(kind='cold', index=index, nonce=nonce)
                row['cold'].append(record)
                stream(args.base, args.model, prompt, record)
                save()
            row['replay'] = dict(kind='replay', nonce=nonce)
            stream(args.base, args.model, prompt, row['replay'])
            row['cold_median_tps'] = statistics.median(r['prefill_tps'] for r in row['cold'])
            row['cold_median_ttft_s'] = statistics.median(r['ttft_s'] for r in row['cold'])
            save()
        report['status'] = 'PASS'
        validate_report(report, sizes, args.repeat, report['source_sha256'])
        save()
    except BaseException as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
