"""Objective audit on the priority-optimal set X (plan section F).

For each audited instance it reports
  - the exact range over X of the linear total shortage (two LPs), which is constant when
    the tier totals are fixed;
  - the range over X of the worst relative shortage max_i (d_i - x_i)/d_i, as a lower
    bound from an LP minimum and the largest value at the payoff-table allocations;
  - for the three retained objectives, the lexicographic payoff table (individual minima
    and the payoff nadir), and whether each retained objective varies on X;
  - the pairwise conflicts: objective j conflicts with objective k when the minimizer of
    j is strictly worse for k than the minimum of k.
Usage: python degeneracy_audit.py <out.json> [structures] [seeds]
"""
import json
import pathlib
import sys
import warnings

import cvxpy as cp
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
warnings.filterwarnings("ignore")

from certrmp.generator import make_instance
from certrmp.instance import priority_totals
from certrmp.objectives import standard_objectives
from certrmp.oracle import Oracle
from certrmp.profiles import payoff_table


def priority_set(inst, sigma, x):
    tau = 1e-9
    cons = [x >= inst.lo, x <= inst.hi, inst.A @ x <= inst.c]
    for t, s in zip(inst.tiers, sigma):
        D = float(inst.d[t].sum())
        cons.append(cp.sum(x[t]) >= D - float(s) - tau * (1 + D))
    return cons


def audit(inst):
    sigma, _ = priority_totals(inst)
    if sigma is None:
        return {"physically_infeasible": True}
    objs = standard_objectives(inst)
    oracle = Oracle(inst, objs, sigma)
    F = payoff_table(oracle)
    fmin, fnad = F.diagonal(), F.max(axis=0)
    x = cp.Variable(inst.n)
    cons = priority_set(inst, sigma, x)
    lin = cp.sum(inst.d - x)
    lo_lin = cp.Problem(cp.Minimize(lin), cons).solve(solver="HIGHS")
    hi_lin = cp.Problem(cp.Maximize(lin), cons).solve(solver="HIGHS")
    rel = cp.max(cp.multiply(inst.d - x, 1.0 / inst.d))
    lo_rel = cp.Problem(cp.Minimize(rel), cons).solve(solver="CLARABEL")
    pts = []
    for j in range(len(objs)):
        lam = np.zeros(len(objs))
        lam[j] = 1.0
        pts.append(oracle.weighted(lam)["x"])
    rel_at_points = [float(np.max((inst.d - p) / inst.d)) for p in pts]
    conflicts = {}
    for j in range(len(objs)):
        for k in range(len(objs)):
            if j != k:
                conflicts[f"{objs[j].name}->{objs[k].name}"] = bool(F[j, k] > fmin[k] + 1e-7 * (1 + abs(fmin[k])))
    varies = {o.name: bool(fnad[j] - fmin[j] > 1e-7 * (1 + abs(fmin[j]))) for j, o in enumerate(objs)}
    return {
        "total_shortage_range": [lo_lin, hi_lin],
        "total_shortage_constant": bool(hi_lin - lo_lin <= 1e-7 * (1 + abs(hi_lin))),
        "worst_relative_shortage_min": lo_rel,
        "worst_relative_shortage_at_payoff_points": rel_at_points,
        "worst_relative_shortage_is_one_everywhere": bool(lo_rel >= 1 - 1e-7),
        "fmin": fmin.tolist(), "fnad": fnad.tolist(), "varies": varies, "conflicts": conflicts,
    }


def main():
    out = sys.argv[1]
    structures = sys.argv[2].split(",") if len(sys.argv) > 2 else ["single", "laminar", "general", "weighted"]
    seeds = range(int(sys.argv[3])) if len(sys.argv) > 3 else range(5)
    rows = []
    for n in (10, 30, 100, 300):
        for st in structures:
            for seed in seeds:
                inst = make_instance(n, st, 1000 + seed)
                row = {"n": n, "structure": st, "seed": 1000 + seed}
                row.update(audit(inst))
                rows.append(row)
                print(n, st, seed, row.get("varies"), row.get("total_shortage_constant"), flush=True)
    pathlib.Path(out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
