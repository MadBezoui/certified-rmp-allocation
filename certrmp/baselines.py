"""Comparators for the exact RMP problem.

  miqcp_scip     generic exact formulation with binary attainment indicators, solved
                 level by level with SCIP (an open-source MINLP solver)
  eps_constraint finite epsilon-constraint sample of the Pareto set, then RMP selection
  nsga2          NSGA-II with an exact repair onto the priority set (single capacity), then
                 RMP selection over the final population
All of them use the same physical constraints, priority totals and objective definitions.
"""
import math
import time

import numpy as np

from .objectives import SHORTFALL, MOVEMENT, DISPARITY, values


def _targets(inst, sigma):
    return [float(inst.d[t].sum()) - float(s) for t, s in zip(inst.tiers, sigma)]


def miqcp_scip(inst, objs, model, sigma, time_limit=60.0, feastol=1e-9):
    """Attainment indicators z[l][j] imply f_j <= b_{sigma_l, j}; levels are optimized in order.

    The formulation has the same boundary convention as the RMP model (equality attains)
    and no strict inequality: z = 0 imposes nothing, and maximization sets z = 1 whenever
    attainment is possible. Big-M values come from valid upper bounds of the objectives
    on the box.
    """
    from pyscipopt import Model, quicksum
    t0 = time.perf_counter()
    M = Model()
    M.hideOutput()
    M.setParam("numerics/feastol", feastol)
    M.setParam("randomization/randomseedshift", 0)
    n, p = inst.n, len(objs)
    lo, hi = inst.lo, inst.hi
    x = [M.addVar(lb=float(lo[i]), ub=float(hi[i]), name=f"x{i}") for i in range(n)]
    for r in range(len(inst.c)):
        nz = np.flatnonzero(inst.A[r])
        M.addCons(quicksum(float(inst.A[r, i]) * x[i] for i in nz) <= float(inst.c[r]))
    tol = 1e-9
    for t, tg in zip(inst.tiers, _targets(inst, sigma)):
        M.addCons(quicksum(x[i] for i in t) >= tg - tol * (1 + abs(tg)))
    F = []
    for j, o in enumerate(objs):
        Fj = M.addVar(lb=0.0, name=f"F{j}")
        F.append(Fj)
        if o.kind == SHORTFALL:
            nz = np.flatnonzero(o.weights)
            M.addCons(quicksum(float(o.weights[i]) * (float(inst.d[i]) - x[i]) * (float(inst.d[i]) - x[i])
                               for i in nz) <= o.scale * Fj)
        elif o.kind == MOVEMENT:
            nz = np.flatnonzero(o.weights)
            terms = []
            for i in nz:
                pp, qq = M.addVar(lb=0.0), M.addVar(lb=0.0)
                M.addCons(x[i] - float(inst.x_prev[i]) == pp - qq)
                terms.append(float(o.weights[i]) * (pp + qq))
            M.addCons(quicksum(terms) <= o.scale * Fj)
        else:
            terms = []
            for J, g in o.active_groups(inst):
                cvar = M.addVar(lb=-1e3, ub=1e3)
                for i in J:
                    s_i = x[i] * (1.0 / float(inst.d[i]))
                    terms.append(g / len(J) * (s_i - cvar) * (s_i - cvar))
            if terms:
                M.addCons(quicksum(terms) <= o.scale * Fj)
            else:
                M.addCons(Fj >= 0)
    ub = np.array([o.upper_bound(inst) for o in objs])
    Z = []
    for lvl in range(model.m):
        b = model.profiles[model.order[lvl]]
        row = []
        for j in range(p):
            z = M.addVar(vtype="B", name=f"z{lvl}_{j}")
            big = max(ub[j] - b[j], 0.0)
            if big > 0:
                M.addCons(F[j] <= float(b[j]) + big * (1 - z))
            row.append(z)
        Z.append(row)
    subsets = model.subsets()
    key, status, xv = [], "OPTIMAL", None
    for lvl in range(model.m):
        remaining = None if time_limit is None else time_limit - (time.perf_counter() - t0)
        if remaining is not None and remaining <= 0:
            status = "TIMEOUT"
            break
        if remaining is not None:
            M.setParam("limits/time", max(remaining, 0.1))
        if model.kind == "srmp":
            from fractions import Fraction
            W = [Fraction(w) for w in model.weights]
            den = math.lcm(*[w.denominator for w in W])
            coef = [int(w * den) for w in W]
            expr = quicksum(coef[j] * Z[lvl][j] for j in range(p))
        else:
            ys = {}
            for S in subsets:
                y = M.addVar(vtype="B")
                ys[S] = y
                for j in S:
                    M.addCons(y <= Z[lvl][j])
            M.addCons(quicksum(ys.values()) == 1)
            expr = quicksum(model.importance(S) * ys[S] for S in subsets)
        M.setObjective(expr, "maximize")
        M.optimize()
        st = M.getStatus()
        if st != "optimal":
            if st in ("timelimit", "gaplimit", "nodelimit"):
                status = "TIMEOUT"
            elif st == "infeasible" and lvl > 0:
                status = "INCONSISTENT"          # a level fixed earlier cannot be met any more
            else:
                status = st.upper()
            if M.getNSols() == 0:
                break
        val = M.getObjVal()
        v = int(round(val))
        if model.kind == "srmp":
            key.append(Fraction(v, den))
        else:
            key.append(v)
        sol = M.getBestSol()
        xv = np.array([M.getSolVal(sol, xi) for xi in x])      # read before the transformed problem is freed
        if st != "optimal":
            break
        M.freeTransform()
        M.addCons(expr >= v)
    out = {"status": status, "claimed_key": tuple(key), "seconds": time.perf_counter() - t0, "x": xv}
    if xv is not None:
        f = values(objs, inst, np.clip(xv, lo, hi))
        out["f"] = f
        out["key"] = model.key(f)
    return out


def eps_constraint(oracle, model, levels=8):
    """min f_1 subject to f_j <= e_j on a regular grid between individual minima and the
    payoff nadir, followed by RMP selection among the computed allocations."""
    t0 = time.perf_counter()
    p = oracle.p
    F = []
    for j in range(p):
        lam = np.zeros(p)
        lam[j] = 1.0
        F.append(oracle.weighted(lam)["f"])
    F = np.array(F)
    fmin, fnad = F.diagonal(), F.max(axis=0)
    grids = [np.linspace(fmin[j], fnad[j], levels) for j in range(1, p)]
    pts = list(F)
    lam = np.zeros(p)
    lam[0] = 1.0
    for combo in np.array(np.meshgrid(*grids)).reshape(p - 1, -1).T:
        w = (math.inf,) + tuple(float(v) for v in combo)
        r = oracle.weighted(lam, w)
        if r["status"] not in ("failed", "infeasible"):
            pts.append(r["f"])
    keys = [model.key(f) for f in pts]
    best = max(range(len(pts)), key=lambda k: keys[k])
    return {"status": "SAMPLE", "key": keys[best], "f": pts[best], "points": len(pts),
            "seconds": time.perf_counter() - t0}


def _capped_simplex(v, upper, budget):
    """Euclidean projection onto {0 <= y <= upper, sum y = budget} by bisection on the multiplier."""
    lo_l, hi_l = np.min(v - upper) - 1.0, np.max(v) + 1.0
    for _ in range(100):
        lam = 0.5 * (lo_l + hi_l)
        s = np.clip(v - lam, 0.0, upper).sum()
        if s > budget:
            lo_l = lam
        else:
            hi_l = lam
    return np.clip(v - 0.5 * (lo_l + hi_l), 0.0, upper)


def _rmp_survival(model):
    """Survival that ranks by RMP key first, then by Pareto rank and crowding inside a key class."""
    from pymoo.core.survival import Survival
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

    crowd = get_crowding_function("cd")

    class RMPSurvival(Survival):
        def __init__(self):
            super().__init__(filter_infeasible=True)

        def _do(self, problem, pop, *args, n_survive=None, **kwargs):
            F = pop.get("F").astype(float, copy=False)
            keys = [model.key(f) for f in F]
            classes = {}
            for i, k in enumerate(keys):
                classes.setdefault(k, []).append(i)
            survivors, rank = [], 0
            for k in sorted(classes, reverse=True):
                idx = np.array(classes[k])
                for front in NonDominatedSorting().do(F[idx]):
                    members = idx[front]
                    cd = crowd.do(F[members], n_remove=0)
                    order = np.argsort(-cd, kind="stable")
                    for j in order:
                        pop[members[j]].set("rank", rank)
                        pop[members[j]].set("crowding", cd[j])
                    room = n_survive - len(survivors)
                    survivors.extend(members[order][:room].tolist())
                    rank += 1
                    if len(survivors) >= n_survive:
                        return pop[survivors]
            return pop[survivors]

    return RMPSurvival()


def nsga2(inst, objs, model, sigma, pop=100, gens=200, seed=0, guided=False):
    """NSGA-II (pymoo) on the single-capacity priority set with an exact projection repair.
    guided=True replaces Pareto ranking by RMP-key ranking in the survival step (Pareto rank
    and crowding then only order allocations with equal keys)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.optimize import minimize
    if inst.A.shape[0] != 1 or not np.all(inst.A == 1):
        raise ValueError("NSGA-II repair implemented for a single capacity only")
    t0 = time.perf_counter()
    lo, hi = inst.lo, inst.hi
    targets = _targets(inst, sigma)

    def project(v):
        out = np.empty_like(v)
        for t, tg in zip(inst.tiers, targets):
            budget = min(max(tg - lo[t].sum(), 0.0), (hi[t] - lo[t]).sum())
            out[t] = lo[t] + _capped_simplex(v[t] - lo[t], hi[t] - lo[t], budget)
        return out

    class Rep(Repair):
        def _do(self, problem, X, **kwargs):
            return np.array([project(x) for x in X])

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=inst.n, n_obj=len(objs), xl=lo, xu=hi)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.array([values(objs, inst, x) for x in X])

    kwargs = {"survival": _rmp_survival(model)} if guided else {}
    algo = NSGA2(pop_size=pop, repair=Rep(), **kwargs)
    res = minimize(Prob(), algo, ("n_gen", gens), seed=seed, verbose=False)
    X = np.atleast_2d(res.pop.get("X") if guided else res.X)
    Fv = np.array([values(objs, inst, x) for x in X])
    keys = [model.key(f) for f in Fv]
    best = max(range(len(X)), key=lambda k: keys[k])
    return {"status": "HEURISTIC", "key": keys[best], "f": Fv[best], "x": X[best],
            "evaluations": pop * gens, "seconds": time.perf_counter() - t0}


def miqcp_gurobi(inst, objs, model, sigma, time_limit=60.0, feastol=1e-9, threads=1):
    """The formulation of miqcp_scip solved with Gurobi (size-limited license: small n only)."""
    import gurobipy as gp
    from gurobipy import GRB
    from fractions import Fraction
    t0 = time.perf_counter()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    M = gp.Model(env=env)
    M.Params.FeasibilityTol = feastol
    M.Params.IntFeasTol = max(feastol, 1e-9)       # the default 1e-5 lets big-M rows claim unattainable thresholds
    M.Params.Threads = threads
    M.Params.Seed = 0
    n, p = inst.n, len(objs)
    lo, hi = inst.lo, inst.hi
    x = M.addVars(n, lb=lo.tolist(), ub=hi.tolist())
    for r in range(len(inst.c)):
        nz = np.flatnonzero(inst.A[r])
        M.addConstr(gp.quicksum(float(inst.A[r, i]) * x[i] for i in nz) <= float(inst.c[r]))
    for t, tg in zip(inst.tiers, _targets(inst, sigma)):
        M.addConstr(gp.quicksum(x[int(i)] for i in t) >= tg - 1e-9 * (1 + abs(tg)))
    F = []
    for j, o in enumerate(objs):
        Fj = M.addVar(lb=0.0)
        F.append(Fj)
        if o.kind == SHORTFALL:
            nz = np.flatnonzero(o.weights)
            M.addQConstr(gp.quicksum(float(o.weights[i]) * (float(inst.d[i]) - x[i]) * (float(inst.d[i]) - x[i])
                                     for i in nz) <= o.scale * Fj)
        elif o.kind == MOVEMENT:
            nz = np.flatnonzero(o.weights)
            pp = M.addVars(len(nz), lb=0.0)
            qq = M.addVars(len(nz), lb=0.0)
            for k, i in enumerate(nz):
                M.addConstr(x[int(i)] - float(inst.x_prev[i]) == pp[k] - qq[k])
            M.addConstr(gp.quicksum(float(o.weights[i]) * (pp[k] + qq[k]) for k, i in enumerate(nz)) <= o.scale * Fj)
        else:
            expr = gp.QuadExpr()
            for J, g in o.active_groups(inst):
                cvar = M.addVar(lb=-1e3, ub=1e3)
                for i in J:
                    s_i = x[int(i)] * (1.0 / float(inst.d[i]))
                    expr += g / len(J) * (s_i - cvar) * (s_i - cvar)
            M.addQConstr(expr <= o.scale * Fj)
    ub = np.array([o.upper_bound(inst) for o in objs])
    Z = []
    for lvl in range(model.m):
        b = model.profiles[model.order[lvl]]
        row = []
        for j in range(p):
            z = M.addVar(vtype=GRB.BINARY)
            big = max(ub[j] - b[j], 0.0)
            if big > 0:
                M.addConstr(F[j] <= float(b[j]) + big * (1 - z))
            row.append(z)
        Z.append(row)
    subsets = model.subsets()
    key, status, xv = [], "OPTIMAL", None
    for lvl in range(model.m):
        remaining = time_limit - (time.perf_counter() - t0)
        if remaining <= 0:
            status = "TIMEOUT"
            break
        M.Params.TimeLimit = max(remaining, 0.1)
        if model.kind == "srmp":
            W = [Fraction(w) for w in model.weights]
            den = math.lcm(*[w.denominator for w in W])
            coef = [int(w * den) for w in W]
            expr = gp.quicksum(coef[j] * Z[lvl][j] for j in range(p))
        else:
            ys = {S: M.addVar(vtype=GRB.BINARY) for S in subsets}
            for S, y in ys.items():
                for j in S:
                    M.addConstr(y <= Z[lvl][j])
            M.addConstr(gp.quicksum(ys.values()) == 1)
            expr = gp.quicksum(model.importance(S) * ys[S] for S in subsets)
        M.setObjective(expr, GRB.MAXIMIZE)
        M.optimize()
        if M.SolCount == 0:
            if M.Status == GRB.TIME_LIMIT:
                status = "TIMEOUT"
            elif M.Status == GRB.INFEASIBLE and lvl > 0:
                status = "INCONSISTENT"          # a level fixed earlier cannot be met any more
            else:
                status = f"STATUS{M.Status}"
            break
        v = int(round(M.ObjVal))
        key.append(Fraction(v, den) if model.kind == "srmp" else v)
        xv = np.array([x[i].X for i in range(n)])
        if M.Status != GRB.OPTIMAL:
            status = "TIMEOUT" if M.Status == GRB.TIME_LIMIT else f"STATUS{M.Status}"
            break
        M.addConstr(expr >= v)
    out = {"status": status, "claimed_key": tuple(key), "seconds": time.perf_counter() - t0, "x": xv}
    if xv is not None:
        f = values(objs, inst, np.clip(xv, lo, hi))
        out["f"] = f
        out["key"] = model.key(f)
    M.dispose()
    env.dispose()
    return out
