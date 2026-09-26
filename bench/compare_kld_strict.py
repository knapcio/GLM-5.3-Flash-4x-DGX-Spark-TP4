"""Compare complete teacher-forced distributions on an operator-supplied panel.

This structural check cannot establish how a reference was collected. Retain its
configuration, tokenizer, calibration SHA and original full per-item lengths.
Arithmetic credit: kld_probe.kl_top (top-K support with folded tail).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import kld_probe


def compare(panel, reference, candidate):
    if not isinstance(panel, list) or not panel:
        raise ValueError('empty calibration panel')
    expected = [(row.get('id', i), row.get('kind', '')) for i, row in enumerate(panel)]
    if len({x[0] for x in expected}) != len(expected):
        raise ValueError('duplicate calibration ID')
    if reference.get('model') != candidate.get('model') or reference.get('k') != 20 or candidate.get('k') != 20:
        raise ValueError('model/top20 mismatch')
    for data in (reference, candidate):
        rows = data.get('items', [])
        if len(rows) != len(expected) or [(r.get('id'), r.get('kind')) for r in rows] != expected:
            raise ValueError('incomplete or reordered item grid')
        for row in rows:
            lp = row.get('prompt_lp')
            if not isinstance(lp, list) or len(lp) < 2 or lp[0] is not None:
                raise ValueError('missing full teacher-forced logprobs')
            for position in lp[1:]:
                if not isinstance(position, dict) or len(position) not in (20, 21):
                    raise ValueError('incomplete top20 support')
                if not all(isinstance(t, str) and t.isdecimal() and str(int(t)) == t
                           and type(v) in (int, float) and math.isfinite(v) and v <= 0
                           for t, v in position.items()):
                    raise ValueError('invalid token/log probability')
                if sum(math.exp(v) for v in position.values()) > 1.00001:
                    raise ValueError('invalid probability mass')
    values, lengths = [], []
    for ref, arm in zip(reference['items'], candidate['items'], strict=True):
        if len(ref['prompt_lp']) != len(arm['prompt_lp']):
            raise ValueError('per-item length mismatch')
        lengths.append(len(ref['prompt_lp']))
        for p, q in zip(ref['prompt_lp'][1:], arm['prompt_lp'][1:], strict=True):
            values.append(kld_probe.kl_top(p, q))
    return dict(status='STRUCTURAL_PASS', items=len(expected), per_item_lengths=lengths,
                teacher_forced_positions=len(values), mean_kl=sum(values)/len(values),
                estimator='top20 folded-tail estimate', numerical_quality_gate_applied=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--texts', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    a = ap.parse_args()
    paths = (a.texts, a.reference, a.candidate)
    raw = [p.read_bytes() for p in paths]
    result = compare(*(json.loads(x) for x in raw))
    result['artifact_sha256'] = dict(zip(('calibration', 'reference', 'candidate'),
        (hashlib.sha256(x).hexdigest() for x in raw), strict=True))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
