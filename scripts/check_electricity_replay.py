"""Independently verify dynamic replay certificates and their epoch-by-epoch inputs."""
import argparse
import gzip
import json
import pathlib
import sys
from collections import Counter
from fractions import Fraction

import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'scripts')]
import electricity
from certrmp import certify, verify
from certrmp.dynamic import emergency_instance
from certrmp.model import RMPModel
from certrmp.objectives import values
from replay_certificates import canonical, digest, get_code_hashes
from replay_electricity import METHODS


def historical_hash_matches(path, expected):
    """Accept a gzip copy of a recorded uncompressed JSONL without changing its identity."""
    if path.name in expected:
        return digest(path.read_bytes()) == expected[path.name]
    if path.suffix == '.gz' and path.name[:-3] in expected:
        with gzip.open(path, 'rb') as f:
            return digest(f.read()) == expected[path.name[:-3]]
    return False


def check_block(data_path, block, method, profiles, ramp, rows, historical):
    electricity.DATA = pathlib.Path(data_path)
    demand, tiers, A, starts, _ = electricity.load()
    model = RMPModel.srmp(profiles, list(electricity.ORDER), list(electricity.WEIGHTS))
    previous = demand[starts[block]].copy()
    expected = {r['t']: r for r in historical}
    if len(expected) != len(historical) or set(expected) != set(range(len(rows))):
        raise ValueError('historical epochs are missing or duplicated')
    counts = Counter(epochs=len(rows), verified=0, unresolved=0, invalid=0,
                     historical_key_matches=0, historical_key_differences=0,
                     historical_f_close=0, historical_f_differences=0)
    issues = []
    for t, row in enumerate(rows):
        cert, trajectory = row['certificate'], row['trajectory']
        try:
            original = electricity.epoch_instance(demand[starts[block] + t], previous, t, tiers, A, ramp)
            actual = original if original.physically_feasible() else emergency_instance(original)
            if (trajectory['block'] != block or trajectory['method'] != method or trajectory['t'] != t
                or trajectory['emergency'] != (actual is not original)
                or cert['instance'] != certify.instance_dict(actual)
                or cert['objectives'] != certify.objectives_dict(electricity.objectives_for(actual))
                or cert['model'] != model.to_dict() or digest(canonical(cert)) != row['certificate_sha256']):
                raise ValueError('epoch input, model or hash mismatch')
            check = verify.verify(cert)
            previous = np.array([float(Fraction(v)) for v in cert['witness']['x']])
            f = values(electricity.objectives_for(original), original, previous)
            if not np.allclose(f, trajectory['f'], rtol=1e-12, atol=1e-12):
                raise ValueError('saved trajectory objectives do not match witness')
            key = [str(v) for v in model.key(f)]
            if key != trajectory['key']:
                raise ValueError('saved trajectory key does not match witness')
            if check.get('exact', False):
                counts['verified'] += 1
            else:
                counts['unresolved'] += 1
                issues.append([t, 'certificate not exactly verified', check.get('errors')])
            old = expected[t]
            if [Fraction(v) for v in old['key']] == [Fraction(v) for v in key]:
                counts['historical_key_matches'] += 1
            else:
                counts['historical_key_differences'] += 1
                issues.append([t, 'historical key differs'])
            if np.allclose(old['f'], f, rtol=1e-8, atol=1e-9):
                counts['historical_f_close'] += 1
            else:
                counts['historical_f_differences'] += 1
                issues.append([t, 'historical objective vector differs beyond tolerance'])
        except Exception as exc:
            counts['invalid'] += 1
            issues.append([t, repr(exc)])
    return {**counts, 'issues': issues}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('data', type=pathlib.Path)
    ap.add_argument('historical', type=pathlib.Path)
    ap.add_argument('output', type=pathlib.Path)
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    config = json.loads((args.output / 'config.json').read_text())
    code = get_code_hashes()
    for p in (ROOT / 'experiments/electricity.py', ROOT / 'scripts/replay_electricity.py'):
        code[str(p.relative_to(ROOT))] = digest(p.read_bytes())
    if code != config['code_sha256'] or digest(args.data.read_bytes()) != config['input_sha256']['data_npz']:
        raise SystemExit('code or processed input differs from replay configuration')
    totals, issues, block_hashes = Counter(), [], {}
    for tag in ('0.5', '2'):
        pf = args.historical / f'profiles_ramp{tag}.json'
        history = args.historical / f'trajectories_ramp{tag}.jsonl'
        if not history.is_file():
            history = history.with_name(history.name + '.gz')
        if (digest(pf.read_bytes()) != config['input_sha256'][pf.name]
            or not historical_hash_matches(history, config['input_sha256'])):
            raise SystemExit('historical input hash differs from replay configuration')
        profiles = json.loads(pf.read_text())['profiles']
        opener = gzip.open if history.suffix == '.gz' else open
        grouped = {}
        with opener(history, 'rt') as f:
            for line in f:
                row = json.loads(line)
                if row['method'] in METHODS:
                    grouped.setdefault((row['block'], row['method']), []).append(row)
        for block in range(electricity.DEV, electricity.NBLOCKS):
            for method in METHODS:
                file = args.output / 'blocks' / f'ramp{tag}_block{block:02d}_{method}.jsonl.gz'
                if not file.exists():
                    totals['missing_blocks'] += 1
                    continue
                with gzip.open(file, 'rt') as f:
                    rows = [json.loads(line) for line in f]
                block_hashes[file.name] = digest(file.read_bytes())
                if len(rows) != 336 or len(grouped.get((block, method), [])) != 336:
                    totals['invalid'] += 1
                    issues.append([tag, block, method, 'incomplete block'])
                    continue
                result = check_block(args.data, block, method, profiles, float(tag), rows, grouped[(block, method)])
                for name, count in result.items():
                    if name != 'issues':
                        totals[name] += count
                issues.extend([[tag, block, method, *issue] for issue in result['issues']])
    result = {'counts': dict(totals), 'issues': issues,
              'evidence': {'config_sha256': digest((args.output / 'config.json').read_bytes()),
                           'checker_sha256': digest(pathlib.Path(__file__).read_bytes()),
                           'verifier_sha256': digest((ROOT / 'certrmp/verify.py').read_bytes()),
                           'block_sha256': block_hashes}}
    print(json.dumps(result, indent=2))
    if (totals['invalid'] or totals['unresolved'] or totals['historical_key_differences']
        or (totals['missing_blocks'] and not args.partial)):
        sys.exit(1)


if __name__ == '__main__':
    main()
