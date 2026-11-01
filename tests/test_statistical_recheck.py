"""Null-based paired randomization, not a tail of the uncentred bootstrap."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_exact_cluster_sign_flip_and_ties():
    spec = importlib.util.spec_from_file_location('statcheck', ROOT / 'scripts/recheck_statistics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.sign_flip_p([1, 2], n_resamples=100) == 0.25
    assert module.sign_flip_p([0, 0], n_resamples=100) == 1.0
    assert module.sign_flip_p([2, 0], n_resamples=100) == 0.5


def test_holm_adjustment():
    spec = importlib.util.spec_from_file_location('statcheck', ROOT / 'scripts/recheck_statistics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(module.holm([0.01, 0.04])) == [0.02, 0.04]


def test_post_audit_summary_binds_analysis_dependency(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    out = tmp_path / 'stats.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/recheck_statistics.py'),
                    str(ROOT / 'results/v1/main.jsonl'), str(out)], check=True, capture_output=True)
    data = json.loads(out.read_text())
    assert data['analysis_library_sha256'] == hashlib.sha256((ROOT / 'scripts/analyze_benchmark.py').read_bytes()).hexdigest()
