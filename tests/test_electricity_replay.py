"""Dynamic replay keeps one auditable certificate per decision epoch."""
import hashlib
import copy
import importlib.util
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_default_dataset_path_is_within_project():
    import subprocess
    import sys
    import os
    env = dict(os.environ)
    env.pop('ABO_DATA', None)
    run = subprocess.run([sys.executable, '-c', 'import experiments.electricity as e; print(e.DATA)'],
                         cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert pathlib.Path(run.stdout.strip()) == ROOT / 'data/processed/electricity.npz'


def test_historical_hash_matches_compressed_copy(tmp_path):
    import gzip
    spec = importlib.util.spec_from_file_location('electricity_check', ROOT / 'scripts/check_electricity_replay.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    raw = b'{"block":6}\n'
    history = tmp_path / 'trajectories_ramp2.jsonl.gz'
    history.write_bytes(gzip.compress(raw, mtime=0))
    expected = {'trajectories_ramp2.jsonl': hashlib.sha256(raw).hexdigest()}
    assert checker.historical_hash_matches(history, expected)
    assert not checker.historical_hash_matches(history, {'trajectories_ramp2.jsonl': '0' * 64})


def test_dynamic_block_collects_and_hashes_certificates(tmp_path):
    spec = importlib.util.spec_from_file_location('electricity_replay', ROOT / 'scripts/replay_electricity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / 'electricity.npz'
    np.savez(data, demand=np.ones((336, 32)), critical=np.arange(8), timestamps=np.arange(336).astype(str))
    profiles = [[10.0, 10.0, 10.0]] * 3
    rows = module.replay_block(data, 6, 'rmp_cold', profiles, 2.0, horizon=3)
    assert len(rows) == 3
    for t, row in enumerate(rows):
        assert row['trajectory']['t'] == t
        assert row['certificate']['model']['order'] == [1, 0, 2]
        raw = json.dumps(row['certificate'], sort_keys=True, separators=(',', ':')).encode()
        assert row['certificate_sha256'] == hashlib.sha256(raw).hexdigest()
        assert row['verification']['exact']
    spec = importlib.util.spec_from_file_location('electricity_check', ROOT / 'scripts/check_electricity_replay.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    old = [r['trajectory'] for r in rows]
    result = checker.check_block(data, 6, 'rmp_cold', profiles, 2.0, rows, old)
    assert result['verified'] == 3 and result['invalid'] == 0 and result['historical_key_matches'] == 3
    altered = copy.deepcopy(rows)
    altered[1]['trajectory']['f'][0] += 10
    result = checker.check_block(data, 6, 'rmp_cold', profiles, 2.0, altered, old)
    assert result['verified'] == 2 and result['invalid'] == 1
    rows[1]['certificate']['levels'] = []
    rows[1]['certificate_sha256'] = hashlib.sha256(json.dumps(rows[1]['certificate'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    result = checker.check_block(data, 6, 'rmp_cold', profiles, 2.0, rows, old)
    assert result['verified'] == 2 and result['unresolved'] == 1
