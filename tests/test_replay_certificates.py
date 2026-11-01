"""End-to-end checks for auditable replay and its restart contract."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_replay_records_certificate_and_refuses_changed_inputs(tmp_path):
    source = ROOT / 'results/v1'
    manifest = json.loads((source / 'manifest_main.json').read_text())
    first = manifest['entries'][0]
    manifest_path = tmp_path / 'small.json'
    manifest_path.write_text(json.dumps({'design': 'main', 'entries': [first]}))
    expected = tmp_path / 'expected.jsonl'
    with (source / 'main.jsonl').open() as f:
        selected = [json.loads(line) for line in f if '"method": "lcs"' in line]
    selected = [r for r in selected if r['id'] == first['id']]
    expected.write_text(''.join(json.dumps(r) + '\n' for r in selected))
    output = tmp_path / 'replay'
    cmd = [sys.executable, str(ROOT / 'scripts/replay_certificates.py'), str(manifest_path), str(expected), str(output), '--workers', '1']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    rows = [json.loads(x) for x in (output / 'records.jsonl').read_text().splitlines()]
    assert len(rows) == len(first['models'])
    for row in rows:
        assert row['verification']['exact'] and row['matches_historical_key']
        cert = row['certificate']
        payload = json.dumps(cert, sort_keys=True, separators=(',', ':')).encode()
        assert row['certificate_sha256'] == hashlib.sha256(payload).hexdigest()
    subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
    assert len((output / 'records.jsonl').read_text().splitlines()) == len(rows)
    check = [sys.executable, str(ROOT / 'scripts/check_replay.py'), str(manifest_path), str(expected), str(output)]
    checked = subprocess.run(check, env=env, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    import gzip
    raw_records = (output / 'records.jsonl').read_bytes()
    shard = output / 'records-00000.jsonl.gz'
    shard.write_bytes(gzip.compress(raw_records, mtime=0))
    (output / 'records.jsonl').unlink()
    checked = subprocess.run(check, env=env, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    (output / 'records.jsonl').write_bytes(raw_records)
    shard.unlink()
    changed = json.loads(json.dumps(rows))
    changed[0]['verification']['refinement_epsilon_relative'] += 1
    (output / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in changed))
    assert subprocess.run(check, env=env, capture_output=True, text=True).returncode != 0
    rows[0]['certificate']['levels'] = []
    rows[0]['certificate_sha256'] = hashlib.sha256(json.dumps(rows[0]['certificate'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    (output / 'records.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert subprocess.run(check, env=env, capture_output=True, text=True).returncode != 0
    manifest_path.write_text(manifest_path.read_text() + ' ')
    assert subprocess.run(cmd, env=env, capture_output=True, text=True).returncode != 0
