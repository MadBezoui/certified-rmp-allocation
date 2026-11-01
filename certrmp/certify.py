"""Assemble an optimality certificate from a search and a refinement.

The certificate is a JSON-ready dictionary of exact numbers (decimal or 'p/q' strings).
Instance data are read as the shortest decimal representation of each binary64 value,
so that a certificate describes a rational instance. Multipliers, tangent points and
the witness are exact rationals; the witness is first repaired so that it satisfies the
linear constraints exactly.
"""
from fractions import Fraction as F
import math

import numpy as np

from .instance import exact_priority_totals


def dec(v):
    return repr(float(v))


def rat(v):
    return str(F(float(v)))


def exact_data(inst):
    xp = [F(dec(v)) for v in inst.x_prev]
    d = [F(dec(v)) for v in inst.d]
    lo = [F(0)] * inst.n
    hi = list(d)
    if inst.r_down is not None:
        lo = [max(F(0), a - F(dec(r))) for a, r in zip(xp, inst.r_down)]
    if inst.r_up is not None:
        hi = [min(b, a + F(dec(r))) for b, a, r in zip(d, xp, inst.r_up)]
    A = [[F(dec(a)) for a in row] for row in inst.A]
    c = [F(dec(v)) for v in inst.c]
    return d, xp, lo, hi, A, c


def instance_dict(inst):
    return {"n": inst.n, "d": [dec(v) for v in inst.d], "x_prev": [dec(v) for v in inst.x_prev],
            "tiers": [np.asarray(t).tolist() for t in inst.tiers],
            "A": [[dec(a) for a in row] for row in inst.A], "c": [dec(v) for v in inst.c],
            "structure": inst.structure,
            "r_down": None if inst.r_down is None else [dec(v) for v in inst.r_down],
            "r_up": None if inst.r_up is None else [dec(v) for v in inst.r_up]}


def objectives_dict(objs):
    out = []
    for o in objs:
        dct = o.to_dict()
        dct["scale"] = dec(o.scale)
        if dct["weights"] is not None:
            dct["weights"] = [dec(v) for v in o.weights]
        if dct["gammas"] is not None:
            dct["gammas"] = [dec(v) for v in o.gammas]
        out.append(dct)
    return out


def _solve_exact(M, rhs):
    """Least-norm exact solution of M z = rhs via (M M^T) y = rhs; None if inconsistent."""
    k = len(M)
    G = [[sum((a * b for a, b in zip(M[r], M[s]) if a != 0 and b != 0), F(0)) for s in range(k)] + [rhs[r]]
         for r in range(k)]
    rows = list(range(k))
    piv_cols = []
    r = 0
    for col in range(k):
        pr = next((i for i in range(r, k) if G[i][col] != 0), None)
        if pr is None:
            continue
        G[r], G[pr] = G[pr], G[r]
        pv = G[r][col]
        G[r] = [v / pv for v in G[r]]
        for i in range(k):
            if i != r and G[i][col] != 0:
                fct = G[i][col]
                G[i] = [a - fct * b for a, b in zip(G[i], G[r])]
        piv_cols.append(col)
        r += 1
    for i in range(r, k):
        if G[i][k] != 0:
            return None
    y = [F(0)] * k
    for i, col in enumerate(piv_cols):
        y[col] = G[i][k]
    n = len(M[0]) if M else 0
    return [sum((M[r][i] * y[r] for r in range(k) if M[r][i] != 0), F(0)) for i in range(n)]


def repair(inst, sigma, x, tol=1e-7):
    """Exactly feasible rational point near x for the priority set, or None."""
    d, xp, lo, hi, A, c = exact_data(inst)
    n = inst.n
    scale = max(1.0, float(np.max(inst.d)))
    snap = F(tol * scale)
    xs = []
    for i, v in enumerate(x):
        v = min(max(F(float(v)), lo[i]), hi[i])
        if hi[i] - v < snap:
            v = hi[i]
        elif v - lo[i] < snap:
            v = lo[i]
        xs.append(v)
    targets = [sum((d[i] for i in t), F(0)) - F(s) for t, s in zip(inst.tiers, sigma)]
    for attempt in range(3):
        rows, rhs = [], []
        for t, tg in zip(inst.tiers, targets):
            tot = sum((xs[i] for i in t), F(0))
            if tot < tg + F(tol * scale):
                row = [F(0)] * n
                for i in t:
                    row[i] = F(1)
                rows.append(row)
                rhs.append(tg - tot)
        for r in range(len(c)):
            tot = sum((A[r][i] * xs[i] for i in range(n) if A[r][i] != 0), F(0))
            if tot > c[r] - F(tol * scale):
                rows.append(list(A[r]))
                rhs.append(c[r] - tot)
        if not rows:
            return xs
        margin = F(tol * scale) * (10 ** attempt)
        free = [i for i in range(n) if xs[i] - lo[i] > margin and hi[i] - xs[i] > margin]
        Mf = [[row[i] for i in free] for row in rows]
        delta = _solve_exact(Mf, rhs)
        if delta is not None:
            cand = list(xs)
            for i, dv in zip(free, delta):
                cand[i] += dv
            ok = all(lo[i] <= cand[i] <= hi[i] for i in range(n))
            ok &= all(sum((A[r][i] * cand[i] for i in range(n) if A[r][i] != 0), F(0)) <= c[r] for r in range(len(c)))
            ok &= all(sum((cand[i] for i in t), F(0)) >= tg for t, tg in zip(inst.tiers, targets))
            if ok:
                return cand
        tol *= 10
    return None


def tangent_s(inst, x):
    return [rat(x[i] / inst.d[i]) if inst.d[i] > 0 else "0" for i in range(inst.n)]


def halfspace_dict(hs, inst):
    return {"mu": [rat(v) for v in hs.mu], "tangent_x": [rat(v) for v in hs.tangent],
            "tangent_s": tangent_s(inst, hs.tangent), "y_cap": [rat(v) for v in hs.y_cap],
            "y_tier": [rat(v) for v in hs.y_tier], "origin": hs.origin}


def wstr(w):
    return ["inf" if math.isinf(v) else dec(v) for v in w]


def _priority_dict(inst, sigma, exact_priority):
    out = {"totals": [str(F(v)) if isinstance(v, F) else rat(v) for v in sigma]}
    if not exact_priority:
        out["method"] = "declared"
    elif "priority_stages" in inst.meta:
        out["method"] = "lp-exact"
        out["stages"] = [{"x": [str(v) for v in st["x"]], "y": [str(v) for v in st["y"]]}
                         for st in inst.meta["priority_stages"]]
    else:
        out["method"] = "polymatroid"
    return out


def build(inst, objs, model, sigma, result, refinement, exact_priority):
    s = result.searcher
    used = set()
    levels = []
    for lvl, rec in enumerate(result.levels):
        for (w, v, dcs) in rec["decisions"]:
            if dcs.feasible is False:
                used.add(dcs.ref)
        nxt = result.levels[lvl + 1]["frontier"] if lvl + 1 < len(result.levels) else result.frontier
        levels.append({"profile": rec["profile"], "next_frontier": [wstr(u) for u in nxt]})
    # Halfspaces from the individual minima can also separate; include them all (p of them).
    for k, hs in enumerate(s.caches.halfspaces):
        if hs.origin.startswith("min_f"):
            used.add(k)
    used = sorted(used)
    halfspaces = [halfspace_dict(s.caches.halfspaces[k], inst) for k in used]
    x = repair(inst, sigma, refinement["x"])
    cert = {
        "version": 1,
        "status": "OPTIMAL",
        "instance": instance_dict(inst),
        "objectives": objectives_dict(objs),
        "model": model.to_dict(),
        "priority": _priority_dict(inst, sigma, exact_priority),
        "key": [str(v) for v in result.key],
        "levels": levels,
        "halfspaces": halfspaces,
        "witness": {"x": [str(v) for v in x] if x is not None else [rat(v) for v in refinement["x"]],
                    "repaired": x is not None},
    }
    ref_items = []
    for u, r in refinement["per_u"]:
        if r["status"] in ("failed", "infeasible"):
            continue
        ref_items.append({"u": wstr(u), "nu": [rat(v) for v in r["nu"]], "tangent_x": [rat(v) for v in r["x"]],
                          "tangent_s": tangent_s(inst, r["x"]), "y_cap": [rat(v) for v in r["y_cap"]],
                          "y_tier": [rat(v) for v in r["y_tier"]]})
    cert["refinement"] = {"lambda": [rat(v) for v in refinement["lam"]], "bounds": ref_items}
    return cert
