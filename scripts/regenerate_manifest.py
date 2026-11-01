"""Regenerate a frozen-design manifest in deterministic order using independent workers.

No existing output is overwritten. Compare the resulting full SHA-256 with
results/v1/MANIFESTS.sha256; solver differences can cause threshold differences.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import numpy as np
from manifest import entries, build_instance, objectives_for, instance_hash, variant
from certrmp.instance import priority_totals
from certrmp.oracle import Oracle
from certrmp import profiles


def build_entry(task):
    design, k, e = task
    inst = build_instance(e)
    objs = objectives_for(inst, e['p'])
    sigma, _ = priority_totals(inst)
    row = dict(e, id=f'{design}-{k:05d}', instance_sha=instance_hash(inst), models=[])
    if sigma is None:
        row['physically_infeasible'] = True
        return row
    oracle = Oracle(inst, objs, sigma)
    for j, (family, kind) in enumerate(e['models']):
        model, info = profiles.generate(oracle, e['m'], seed=e['seed'] * 10 + j, family=family, kind=kind)
        if 'variants' in e:
            rng = np.random.default_rng(e['seed'] + 77)
            span = np.array(info['fnad']) - np.array(info['fmin'])
            span = np.where(span > 1e-6 * (1 + np.abs(info['fmin'])), span,
                            profiles.SPAN_FLOOR * np.maximum(np.abs(info['fmin']), 1e-6))
            for name in e['variants']:
                md = variant(model, name, rng, span)
                row['models'].append(dict(model=md, variant=name, **dict(info, kind=md['kind'])))
        else:
            row['models'].append(dict(model=model.to_dict(), **info))
    row.pop('variants', None)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('design', choices=('main', 'stress_profiles', 'stress_dimension', 'sensitivity', 'preference'))
    ap.add_argument('output', type=pathlib.Path)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('workers must be positive')
    if args.output.exists():
        raise SystemExit('output already exists')
    tasks = [(args.design, k, e) for k, e in enumerate(entries(args.design))]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(build_entry, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f'{args.design}: {len(rows)}/{len(tasks)}', flush=True)
    encoded = json.dumps({'design': args.design, 'entries': rows}).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        f.write(encoded)
    print(f'{hashlib.sha256(encoded).hexdigest()}  {args.output}', flush=True)


if __name__ == '__main__':
    main()
