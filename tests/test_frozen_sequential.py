"""Smoke test the isolated historical runner without treating its status as proof."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_runner_records_hash_and_refuses_overwrite(tmp_path):
    out = tmp_path / 'historical.jsonl'
    cmd = [sys.executable, str(ROOT / 'scripts/run_frozen_sequential.py'),
           str(ROOT / 'results/v1/manifest_main.json'), str(out), '--id', 'main-00000']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    run = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    data = out.read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    assert len(rows) == 2
    assert {r['model'] for r in rows} == {0, 1}
    assert all(r['code_sha'] == 'db89d903964c43fe' and
               r['execution'] == 'sequential without per-worker RLIMIT_AS' for r in rows)
    again = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert again.returncode != 0
    assert out.read_bytes() == data
