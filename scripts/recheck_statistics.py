"""Post-audit paired sign-flip sensitivity check on the historical benchmark.

This is a newly specified analysis, not the frozen confirmatory analysis. The null
assumption is sign symmetry/exchangeability within an instance cluster. It produces a
separate JSON output and does not modify the historical observations or bootstrap CIs.
"""
import hashlib
import itertools
import json
import pathlib
import sys
import numpy as np
import pandas as pd


def sign_flip_p(cluster_sums, n_resamples=100000, seed=20261101):
    s = np.asarray(cluster_sums, dtype=float)
    if s.ndim != 1 or not np.all(np.isfinite(s)) or len(s) == 0:
        raise ValueError('finite nonempty cluster sums required')
    observed = float(s.sum())
    tolerance = 1e-12 * max(1, abs(observed))
    if len(s) <= 18:
        all_sums = np.array([np.dot(signs, s) for signs in itertools.product((-1, 1), repeat=len(s))])
        return float(np.mean(all_sums >= observed - tolerance))
    rng = np.random.default_rng(seed)
    count = 0
    for start in range(0, n_resamples, 1024):
        batch = min(1024, n_resamples - start)
        signs = 2 * rng.integers(0, 2, size=(batch, len(s)), dtype=np.int8) - 1
        count += int(np.sum(signs @ s >= observed - tolerance))
    return (count + 1) / (n_resamples + 1)


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def clusters(index, values):
    return pd.DataFrame({'id': [idx[0] for idx in index], 'value': values}).groupby('id', sort=True).value.sum().to_numpy()


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: recheck_statistics.py historical_main.jsonl new_output.json')
    import analyze_benchmark as ab
    path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    if out.exists():
        raise SystemExit('output exists; choose a new path')
    df = ab.load(path)
    lcs, scip = ab.paired(df[df.n == df.n.max()], 'lcs', 'scip')
    d = lcs.ok60.astype(float).to_numpy() - scip.ok60.astype(float).to_numpy()
    h1 = sign_flip_p(clusters(lcs.index, d))
    lcs, enum = ab.paired(df, 'lcs', 'enum')
    ca = lcs.calls.where(lcs.calls.notna(), lcs.search_calls).astype(float).to_numpy()
    cb = enum.calls.where(enum.calls.notna(), enum.search_calls).astype(float).to_numpy()
    good = (ca > 0) & (cb > 0)
    diffs = np.log(cb[good]) - np.log(ca[good])
    h2 = sign_flip_p(clusters(lcs.index[good], diffs))
    adj = holm([h1, h2])
    result = {'analysis': 'post-audit sign-flip sensitivity; not preregistered',
              'assumption': 'paired sign symmetry/exchangeability at instance cluster level',
              'resamples': 100000, 'seed': 20261101,
              'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
              'analysis_library_sha256': hashlib.sha256(pathlib.Path(ab.__file__).read_bytes()).hexdigest(),
              'numpy_version': np.__version__, 'pandas_version': pd.__version__, 'python': sys.version,
              'H1': {'p_one_sided': h1, 'p_holm_two_tests': float(adj[0]), 'pairs': len(d),
                     'observed_difference': float(d.mean())},
              'H2': {'p_one_sided': h2, 'p_holm_two_tests': float(adj[1]), 'pairs': int(good.sum()),
                     'geometric_ratio': float(np.exp(diffs.mean())),
                     'note': 'includes observed calls up to stopping for incomplete enumeration'},
              'H3': {'inference': 'descriptive only',
                     'reason': 'an exactly verified optimum cannot be beaten by a feasible comparator; null is asymmetric'}}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
