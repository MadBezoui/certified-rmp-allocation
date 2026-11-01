"""Independently replay stored certificate checks and bind them to the manifest.

Exit nonzero for missing, invalid or key-disagreeing records. --partial permits missing
records but never treats missing, invalid or discrepant records as verified.
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import sys
from fractions import Fraction

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'scripts')]
from replay_certificates import canonical, digest, get_code_hashes
from manifest import build_instance, instance_hash, objectives_for
from certrmp import certify, verify


def record_lines(directory):
    raw = directory / 'records.jsonl'
    parts = sorted(directory.glob('records-*.jsonl.gz'))
    if raw.exists() and parts:
        raise ValueError('ambiguous raw and sharded records')
    if raw.exists():
        paths = [raw]
    else:
        if not parts or [p.name for p in parts] != [f'records-{i:05d}.jsonl.gz' for i in range(len(parts))]:
            raise ValueError('missing or noncontiguous record shards')
        paths = parts
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as stream:
            yield from stream


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=pathlib.Path)
    ap.add_argument('historical', type=pathlib.Path)
    ap.add_argument('output', type=pathlib.Path)
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    config = json.loads((args.output / 'config.json').read_text())
    raw = args.manifest.read_bytes()
    historical = args.historical.read_bytes()
    if (digest(raw) != config['manifest_sha256'] or digest(historical) != config['historical_sha256']
        or get_code_hashes() != config['code_sha256']):
        raise SystemExit('input or code hash mismatch')
    entries = {e['id']: e for e in json.loads(raw)['entries']}
    required = {(e['id'], i) for e in entries.values() for i in range(len(e['models']))}
    old = {}
    for line in historical.splitlines():
        r = json.loads(line)
        if r.get('method') == 'lcs':
            key = (r['id'], int(r['model']))
            if key in old:
                raise SystemExit('duplicate historical record')
            old[key] = r
    counts = {'expected': len(required), 'read': 0, 'verified': 0, 'matching_keys': 0,
              'unresolved': 0, 'mismatched_keys': 0, 'invalid': 0, 'missing': 0}
    issues, done = [], set()
    stream_hash = hashlib.sha256()
    for line in record_lines(args.output):
            stream_hash.update(line)
            row = json.loads(line)
            key = (row['id'], row['model'])
            counts['read'] += 1
            if key in done or key not in required or key not in old:
                counts['invalid'] += 1
                issues.append([key, 'duplicate or foreign pair'])
                continue
            done.add(key)
            entry = entries[key[0]]
            cert = row.get('certificate')
            try:
                if cert is None:
                    counts['unresolved'] += 1
                    issues.append([key, 'no certificate'])
                    continue
                inst = build_instance(entry)
                if (instance_hash(inst) != entry['instance_sha'] or
                    cert['instance'] != certify.instance_dict(inst) or
                    cert['objectives'] != certify.objectives_dict(objectives_for(inst, entry['p'])) or
                    cert['model'] != entry['models'][key[1]]['model'] or
                    digest(canonical(cert)) != row['certificate_sha256']):
                    raise ValueError('certificate or instance binding mismatch')
                out = verify.verify(cert)
                recorded = row.get('verification', {})
                if any(out.get(k) != recorded.get(k) for k in
                       ('checks', 'verified', 'exact', 'refinement_epsilon', 'refinement_epsilon_relative')):
                    raise ValueError('recorded verification differs from independent verification')
                if not out.get('exact'):
                    counts['unresolved'] += 1
                    issues.append([key, 'certificate not exactly verified', out.get('errors')])
                    continue
                if cert['key'] != row['replay']['key']:
                    raise ValueError('record key differs from certificate key')
                counts['verified'] += 1
                old_key = old[key].get('key')
                if old_key is None:
                    continue
                if [Fraction(v) for v in cert['key']] == [Fraction(v) for v in old_key]:
                    counts['matching_keys'] += 1
                else:
                    counts['mismatched_keys'] += 1
                    issues.append([key, 'key differs from historical record'])
            except Exception as exc:
                counts['invalid'] += 1
                issues.append([key, repr(exc)])
    counts['missing'] = len(required - done)
    evidence = {'records_sha256': stream_hash.hexdigest(),
                'config_sha256': digest((args.output / 'config.json').read_bytes()),
                'checker_sha256': digest(pathlib.Path(__file__).read_bytes()),
                'verifier_sha256': digest((ROOT / 'certrmp/verify.py').read_bytes())}
    print(json.dumps({'counts': counts, 'evidence': evidence, 'issues': issues}, indent=2))
    if counts['invalid'] or counts['unresolved'] or counts['mismatched_keys'] or (counts['missing'] and not args.partial):
        sys.exit(1)


if __name__ == '__main__':
    main()
