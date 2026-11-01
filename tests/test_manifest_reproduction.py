"""Parallel regeneration must preserve the serialized frozen manifest exactly."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_parallel_preference_manifest_matches_frozen_bytes(tmp_path):
    out = tmp_path / 'manifest_preference.json'
    run = subprocess.run([sys.executable, str(ROOT / 'scripts/regenerate_manifest.py'), 'preference', str(out), '--workers', '2'],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert out.read_bytes() == (ROOT / 'results/v1/manifest_preference.json').read_bytes()
    assert subprocess.run([sys.executable, str(ROOT / 'scripts/regenerate_manifest.py'), 'preference', str(out)],
                          capture_output=True, text=True).returncode != 0
