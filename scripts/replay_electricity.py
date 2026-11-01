"""Replay dynamic RMP trajectories with per-epoch certificates in separate gzip blocks.

Usage: replay_electricity.py ELECTRICITY_NPZ HISTORICAL_DIR OUTPUT_DIR [--workers 1]
The processed input is checked against the UCI demand-array hash. Existing complete
blocks are preserved on resume. The historical trajectories and timings are never changed.
"""
import argparse
import gzip
import hashlib
import json
import os
import pathlib
import platform
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'scripts')]
import numpy as np
import electricity
from certrmp import certify, verify
from replay_certificates import canonical, digest, get_code_hashes

DEMAND_SHA = 'f5492f007e61cfd402b7b83fe29aaa59c1349ed7e98d6c0f0df079fa678f151c'
METHODS = ('rmp_cold', 'rmp_reuse', 'rmp_warm')


def replay_block(data_path, block, method, profiles, ramp, horizon=336):
    electricity.DATA = pathlib.Path(data_path)
    captured = []
    original = certify.build
    def capture(*args, **kwargs):
        cert = original(*args, **kwargs)
        captured.append(cert)
        return cert
    certify.build = capture
    try:
        rows = electricity.run_block((block, method, profiles, ramp, horizon))
    finally:
        certify.build = original
    if len(rows) != len(captured):
        raise RuntimeError('the number of certificates differs from the epoch count')
    return [{'trajectory': row, 'certificate': cert,
             'certificate_sha256': digest(canonical(cert)), 'verification': verify.verify(cert)}
            for row, cert in zip(rows, captured)]


def do_task(args):
    data, block, method, profiles, ramp = args
    rows = replay_block(data, block, method, profiles, ramp)
    return block, method, ramp, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data', type=pathlib.Path)
    ap.add_argument('historical_dir', type=pathlib.Path)
    ap.add_argument('output', type=pathlib.Path)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error('workers must be positive')
    with np.load(args.data) as data:
        observed = digest(np.ascontiguousarray(data['demand'], dtype=np.float64).tobytes())
        if observed != DEMAND_SHA or data['demand'].shape != (35040, 32) or list(data['critical']) != list(range(8)):
            raise SystemExit('processed UCI demand data do not match the frozen study')
    profiles = {}
    input_hashes = {'data_npz': digest(args.data.read_bytes())}
    for tag in ('0.5', '2'):
        profile = args.historical_dir / f'profiles_ramp{tag}.json'
        history = args.historical_dir / f'trajectories_ramp{tag}.jsonl'
        if not history.is_file():
            history = history.with_name(history.name + '.gz')
        input_hashes[profile.name] = digest(profile.read_bytes())
        input_hashes[history.name] = digest(history.read_bytes())
        profiles[tag] = json.loads(profile.read_text())['profiles']
    code = get_code_hashes()
    for p in (ROOT / 'experiments/electricity.py', pathlib.Path(__file__).resolve()):
        code[str(p.relative_to(ROOT))] = digest(p.read_bytes())
    config = {'schema': 1, 'input_sha256': input_hashes, 'processed_demand_sha256': observed,
              'code_sha256': code, 'python': sys.version, 'platform': platform.platform(),
              'numpy': np.__version__, 'workers': args.workers, 'blocks': 24,
              'epochs_per_block': 336, 'methods': METHODS, 'ramps': ['0.5', '2']}
    args.output.mkdir(parents=True, exist_ok=True)
    cp = args.output / 'config.json'
    if cp.exists():
        if json.loads(cp.read_text()) != json.loads(json.dumps(config)):
            raise SystemExit('output belongs to different inputs, code, environment or worker count')
    else:
        if any(args.output.iterdir()):
            raise SystemExit('output directory is not empty')
        cp.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    blocks = args.output / 'blocks'
    blocks.mkdir(exist_ok=True)
    tasks = []
    total = 0
    for tag in ('0.5', '2'):
        for b in range(electricity.DEV, electricity.NBLOCKS):
            for method in METHODS:
                total += 1
                dest = blocks / f'ramp{tag}_block{b:02d}_{method}.jsonl.gz'
                if dest.exists():
                    with gzip.open(dest, 'rt') as f:
                        rows = [json.loads(line) for line in f]
                    if len(rows) != 336 or any(digest(canonical(r['certificate'])) != r['certificate_sha256'] for r in rows):
                        raise SystemExit(f'incomplete or corrupt existing block: {dest}')
                else:
                    tasks.append((str(args.data.resolve()), b, method, profiles[tag], float(tag)))
    complete = total - len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for b, method, ramp, rows in pool.map(do_task, tasks, chunksize=1):
            tag = f'{ramp:g}'
            dest = blocks / f'ramp{tag}_block{b:02d}_{method}.jsonl.gz'
            if dest.exists():
                raise SystemExit(f'refusing to overwrite {dest}')
            payload = b''.join(canonical(r) + b'\n' for r in rows)
            tmp = dest.with_suffix(dest.suffix + '.partial')
            with tmp.open('xb') as f:
                f.write(gzip.compress(payload, mtime=0))
            tmp.rename(dest)
            complete += 1
            failures = sum(not r['verification'].get('exact', False) for r in rows)
            print(f'{complete}/{total} ramp={tag} block={b} method={method} '
                  f'exact={len(rows)-failures}/{len(rows)}', flush=True)
    print(f'Completed {complete}/{total} dynamic blocks')


if __name__ == '__main__':
    main()
