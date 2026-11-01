"""Run selected historical tasks sequentially without changing the frozen source.

The historical multiprocessing runner sets a per-worker RLIMIT_AS, which is not
portable to every macOS configuration. This wrapper calls the same frozen task
function in one process, does not enforce that limit, and writes a new file only.
Statuses from the historical verifier are not exact certificates.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'frozen/v1'
sys.path[:0] = [str(FROZEN), str(FROZEN / 'experiments')]
import runner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('output', type=Path, help='new file; existing output is never overwritten')
    ap.add_argument('--id', required=True, help='instance ID in the frozen manifest')
    ap.add_argument('--method', default='lcs')
    ap.add_argument('--time-limit', type=float, default=60)
    args = ap.parse_args()
    raw = args.manifest.read_bytes()
    entries = [e for e in json.loads(raw)['entries'] if e['id'] == args.id]
    if len(entries) != 1:
        ap.error('instance ID must occur exactly once in the manifest')
    entry = entries[0]
    if not runner.applicable(entry, args.method):
        ap.error('method is not applicable to this instance')
    code_sha = runner.code_hash()
    manifest_sha = hashlib.sha256(raw).hexdigest()[:16]
    if code_sha != 'db89d903964c43fe':
        raise SystemExit('frozen solver code hash differs from the historical records')
    with args.output.open('x') as fh:
        for mi in range(max(1, len(entry.get('models', [])))):
            rec = runner.run_task((entry, mi, args.method, args.time_limit))
            rec.update(code_sha=code_sha, manifest_sha=manifest_sha, time_limit=args.time_limit,
                       execution='sequential without per-worker RLIMIT_AS', platform=platform.platform())
            fh.write(json.dumps(rec, default=float, allow_nan=False, separators=(',', ':')) + '\n')
            fh.flush()
            print(entry['id'], mi, args.method, rec.get('status'), flush=True)


if __name__ == '__main__':
    main()
