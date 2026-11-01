"""Liste les candidats meilleurs non separes et les marges rationnelles obtenues."""
import json
import sys
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / '04_Code_et_experiences'
sys.path.insert(0, str(ROOT))
from certrmp import verify


def missed(cert):
    D = verify.Data(cert)
    sig = [F(v) for v in cert['priority']['totals']]
    bounds = []
    for hs in cert['halfspaces']:
        mu = [F(v) for v in hs['mu']]
        if any(v < 0 for v in mu):
            continue
        L = verify.halfspace_bound(D, sig, mu, [F(v) for v in hs['tangent_x']],
            [F(v) for v in hs['tangent_s']], [max(F(v), F(0)) for v in hs['y_cap']],
            [max(F(v), F(0)) for v in hs['y_tier']])
        bounds.append((mu, L))
    frontier = [tuple([None] * D.p)]
    results = []
    for lvl, rec in enumerate(cert['levels']):
        b = D.profiles[D.order[lvl]]
        best = F(cert['key'][lvl]) if verify.D_is_srmp(cert) else int(cert['key'][lvl])
        nxt = [tuple(verify.q(v) for v in u) for u in rec['next_frontier']]
        for u in frontier:
            att = [j for j in range(D.p) if u[j] is not None and u[j] <= b[j]]
            free = [j for j in range(D.p) if j not in att]
            for n in range(len(free)+1):
                for rest in combinations(free, n):
                    S = frozenset(att) | frozenset(rest)
                    w = tuple((b[j] if u[j] is None else min(u[j],b[j])) if j in S else u[j] for j in range(D.p))
                    v = D.importance(S)
                    if v < best:
                        continue
                    covered = v == best and any(all(uu is None or (ww is not None and ww <= uu)
                                                   for ww,uu in zip(w,u2)) for u2 in nxt)
                    if covered:
                        continue
                    diffs = [L - sum((mu[j]*w[j] for j in range(D.p) if mu[j]>0),F(0))
                             for mu,L in bounds if not any(mu[j]>0 and w[j] is None for j in range(D.p))]
                    gap = max(diffs) if diffs else None
                    if gap is None or gap <= 0:
                        results.append({'level': lvl, 'w': ['inf' if x is None else str(x) for x in w],
                                        'importance': str(v), 'separation_margin': float(gap) if gap is not None else None})
        frontier = nxt
    return results


if __name__ == '__main__':
    path=Path(__file__).parent/'tous_sans_relaxation.jsonl'
    for row in (json.loads(line) for line in path.open()):
        if row['new']['status'] != 'OPTIMAL_VERIFIED':
            bad=missed(row['new']['certificate'])
            print(json.dumps({'id':row['id'],'missing':bad}, separators=(',',':')))
