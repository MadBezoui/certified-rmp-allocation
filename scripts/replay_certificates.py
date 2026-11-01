"""Replay LCS on frozen manifest pairs, preserving each independently checkable certificate.

Usage: python scripts/replay_certificates.py MANIFEST HISTORICAL_JSONL NEW_OUTPUT_DIR [--workers 2]
Records are append-only. Resume is accepted only for the same inputs, code and environment.
This replay does not replace historical timings or assert that numerical failures are successes.
"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]

from manifest import build_instance, instance_hash, objectives_for
from certrmp.model import model_from_dict
from certrmp.pipeline import solve
from certrmp import verify
import certrmp.oracle as oracle_mod
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def get_code_hashes():
    files = (list((ROOT / 'certrmp').glob('*.py')) + list((ROOT / 'rmp').glob('*.py'))
             + [ROOT / 'experiments/manifest.py', pathlib.Path(__file__).resolve()])
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted(files)}


def run_pair(task):
    entry, mi, old = task
    result = {'id': entry['id'], 'model': mi, 'historical_status': old.get('status'),
              'historical_key': old.get('key'), 'expected_instance_sha': entry['instance_sha']}
    try:
        inst = build_instance(entry)
        observed = instance_hash(inst)
        result['instance_sha'] = observed
        if observed != entry['instance_sha']:
            raise ValueError('regenerated instance hash differs from frozen manifest')
        tol = entry.get('solver_tol', 1e-10)
        oracle_mod.SOLVER_OPTS.update(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
        model = model_from_dict(entry['models'][mi]['model'])
        rec = solve(inst, objectives_for(inst, entry['p']), model, time_limit=60, keep_certificate=True)
        rec.pop('x', None)
        cert = rec.pop('certificate', None)
        result['replay'] = rec
        result['matches_historical_key'] = (rec.get('key') is not None and old.get('key') is not None and
            [Fraction(v) for v in rec['key']] == [Fraction(v) for v in old['key']])
        if cert is not None:
            cert = json.loads(json.dumps(cert))
            result['certificate'] = cert
            result['certificate_sha256'] = digest(canonical(cert))
            result['verification'] = verify.verify(cert)
        else:
            result['verification'] = {'verified': False, 'exact': False, 'errors': ['no certificate produced']}
    except Exception as exc:
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc(limit=8)
        result['verification'] = {'verified': False, 'exact': False, 'errors': [repr(exc)]}
        result['matches_historical_key'] = False
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=pathlib.Path)
    ap.add_argument('historical', type=pathlib.Path)
    ap.add_argument('output', type=pathlib.Path)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('workers must be positive')
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    historical = args.historical.read_bytes()
    expected = {}
    for line in historical.splitlines():
        rec = json.loads(line)
        if rec.get('method') == 'lcs':
            k = (rec['id'], int(rec['model']))
            if k in expected:
                raise SystemExit(f'duplicate historical key {k}')
            expected[k] = rec
    tasks = [(e, mi, expected[(e['id'], mi)]) for e in manifest['entries'] for mi in range(len(e['models']))
             if (e['id'], mi) in expected]
    required = {(e['id'], mi) for e in manifest['entries'] for mi in range(len(e['models']))}
    if {(e['id'], mi) for e, mi, _ in tasks} != required:
        raise SystemExit('historical LCS records do not cover every manifest pair')
    import cvxpy, clarabel, scipy
    config = {'schema': 1, 'manifest_sha256': digest(raw), 'historical_sha256': digest(historical),
              'code_sha256': get_code_hashes(), 'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'cvxpy': cvxpy.__version__,
              'clarabel': clarabel.__version__, 'time_limit_s': 60, 'pairs': len(tasks),
              'workers': args.workers}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise SystemExit('output belongs to different inputs, code, environment or worker count')
    else:
        if any(args.output.iterdir()):
            raise SystemExit('output directory is not empty')
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    path = args.output / 'records.jsonl'
    done = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                k = (row['id'], row['model'])
                if k in done or k not in required:
                    raise SystemExit('duplicate or foreign record in output')
                if 'certificate' in row and row['certificate_sha256'] != digest(canonical(row['certificate'])):
                    raise SystemExit('stored certificate hash mismatch')
                done.add(k)
    remaining = (task for task in tasks if (task[0]['id'], task[1]) not in done)
    with path.open('a') as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_pair, remaining, chunksize=1):
            f.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
            f.flush()
            done.add((row['id'], row['model']))
            print(f"{len(done)}/{len(tasks)} {row['id']} model={row['model']} "
                  f"status={row.get('replay', {}).get('status', 'ERROR')} "
                  f"exact={row['verification'].get('exact', False)} "
                  f"same_key={row['matches_historical_key']}", flush=True)
    print(f'Completed {len(done)}/{len(tasks)} records in {path}')


if __name__ == '__main__':
    main()
