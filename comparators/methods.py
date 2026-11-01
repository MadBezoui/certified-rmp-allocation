"""Comparators from the literature, added after the frozen protocol (protocol v1, section 8).

Every method gets the budget of the frozen comparators (60 s of wall time on one thread) and
works on the same instances, priority set, objectives and RMP models. The frozen code is
imported, never changed.

  asf        reference point method (Wierzbicki): one augmented achievement minimization per
             profile, in consultation order, then the best key among these allocations
  lgp        lexicographic goal programming: the profiles are the goals and the consultation
             order gives the priority levels; weighted deviations are minimized level by level,
             then an efficiency step minimizes the scaled sum of the criteria
  eps60      epsilon-constraint sample on nested grids, refined until the budget is spent
  nsga60     NSGA-II (pymoo) until the budget is spent
  rmpga60    the RMP-guided NSGA-II of the frozen protocol until the budget is spent
  rnsga60    R-NSGA-II (pymoo) with the profiles as reference points
  scip_ind   the frozen MIQCP with indicator constraints instead of big-M rows, solved by SCIP

The evolutionary searches repair every individual by the Euclidean projection onto the priority
set: bisection per tier for a single capacity, as in the frozen comparator, and a quadratic
program solved by Clarabel for the other structures. Anytime methods keep the best key found so
far and the times at which it improved; the key reported for the budget counts only
allocations found within it.
"""
import itertools
import math
import time
from fractions import Fraction

import cvxpy as cp
import numpy as np

from certrmp import baselines
from certrmp.objectives import SHORTFALL, MOVEMENT, values
from certrmp.oracle import SOLVER_OPTS, Oracle, solve_with_retries

TAU_REL = 1e-9          # tolerance of the tier totals, as in certrmp.oracle.Oracle


def tier_targets(inst, sigma):
    return [float(inst.d[t].sum()) - float(s) - TAU_REL * (1 + float(inst.d[t].sum()))
            for t, s in zip(inst.tiers, sigma)]


def x_constraints(inst, sigma, x):
    """The priority set X of the oracle: box, capacity rows and minimal tier totals."""
    return ([x >= inst.lo, x <= inst.hi, inst.A @ x <= inst.c]
            + [cp.sum(x[t]) >= tt for t, tt in zip(inst.tiers, tier_targets(inst, sigma))])


def objective_matrix(objs, inst, X):
    """values() for every row of X, with the same formulas vectorized over the rows."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    out = np.empty((len(X), len(objs)))
    for j, o in enumerate(objs):
        if o.kind == SHORTFALL:
            out[:, j] = ((inst.d - X) ** 2) @ o.weights / o.scale
        elif o.kind == MOVEMENT:
            out[:, j] = np.abs(X - inst.x_prev) @ o.weights / o.scale
        else:
            total = np.zeros(len(X))
            for J, g in o.active_groups(inst):
                S = X[:, J] / inst.d[J]
                total += g / len(J) * ((S - S.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
            out[:, j] = total / o.scale
    return out


def residuals(inst, sigma, x):
    """Largest violation of the physical constraints and of the priority totals (float)."""
    x = np.asarray(x, dtype=float)
    phys = max(float(np.max(inst.A @ x - inst.c)), float(np.max(x - inst.hi)), float(np.max(inst.lo - x)), 0.0)
    prio = max([float(inst.d[t].sum() - x[t].sum() - float(s)) for t, s in zip(inst.tiers, sigma)] + [0.0])
    return phys, prio


class Projection:
    """Euclidean projection onto the priority set."""

    def __init__(self, inst, sigma):
        self.inst = inst
        self.lo, self.hi = inst.lo, inst.hi
        self.single = inst.A.shape[0] == 1 and bool(np.all(inst.A == 1))
        if self.single:
            # the repair of the frozen comparator: each tier total equals its priority total
            self.targets = baselines._targets(inst, sigma)
            return
        import clarabel
        import scipy.sparse as sp
        n = inst.n
        T = sp.lil_matrix((len(inst.tiers), n))
        for k, t in enumerate(inst.tiers):
            T[k, t] = 1.0
        eye = sp.identity(n, format="csr")
        A = sp.vstack([sp.csr_matrix(inst.A), -T.tocsr(), eye, -eye]).tocsc()
        b = np.concatenate([inst.c, -np.asarray(tier_targets(inst, sigma)), self.hi, -self.lo])
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.presolve_enable = False          # keeps data updates allowed
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
        settings.max_iter = 400
        self.solver = clarabel.DefaultSolver(sp.identity(n, format="csc"), np.zeros(n), A, b,
                                             [clarabel.NonnegativeConeT(len(b))], settings)
        self.failures = 0

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        if self.single:
            out = np.empty_like(v)
            lo, hi = self.lo, self.hi
            for t, tg in zip(self.inst.tiers, self.targets):
                budget = min(max(tg - lo[t].sum(), 0.0), (hi[t] - lo[t]).sum())
                out[t] = lo[t] + baselines._capped_simplex(v[t] - lo[t], hi[t] - lo[t], budget)
            return out
        self.solver.update(q=-v)
        sol = self.solver.solve()
        if str(sol.status) not in ("Solved", "AlmostSolved"):
            self.failures += 1
        return np.clip(np.asarray(sol.x, dtype=float), self.lo, self.hi)


class Anytime:
    """Best key found so far, the allocation that attains it, and when it improved."""

    def __init__(self, inst, objs, model, t0, limit):
        self.inst, self.objs, self.model, self.t0, self.limit = inst, objs, model, t0, limit
        self.best_key = None          # best over everything evaluated, also after the limit
        self.within = None            # (key, f, x, seconds) of the best found within the limit
        self.trajectory = []          # [seconds, key] at every improvement
        self.evaluations = 0

    def offer(self, X, F=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        F = objective_matrix(self.objs, self.inst, X) if F is None else np.atleast_2d(F)
        self.evaluations += len(X)
        keys = [self.model.key(f) for f in F]
        k = max(range(len(keys)), key=keys.__getitem__)
        if self.best_key is not None and not keys[k] > self.best_key:
            return
        x = X[k].copy()
        f = values(self.objs, self.inst, x)          # the frozen evaluation decides the key
        key = self.model.key(f)
        if self.best_key is not None and not key > self.best_key:
            return
        t = time.perf_counter() - self.t0
        self.best_key = key
        self.trajectory.append([t, key])
        if t <= self.limit:
            self.within = (key, f, x, t)

    def result(self, status, **extra):
        out = {"status": status if self.within is not None else "NO_SOLUTION",
               "seconds": time.perf_counter() - self.t0, "evaluations": self.evaluations,
               "trajectory": [[round(t, 4), k] for t, k in self.trajectory]}
        if self.within is not None:
            key, f, x, t = self.within
            out.update(key=key, f=f, x=x, seconds_to_key=t)
        out.update(extra)
        return out


def emo(inst, objs, model, sigma, variant, time_limit=60.0, pop=100, seed=0):
    """NSGA-II ("nsga"), RMP-guided NSGA-II ("rmpga") or R-NSGA-II ("rnsga") until the budget
    is spent. The population size and operators are those of the frozen comparator."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.rnsga2 import RNSGA2
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.optimize import minimize
    from pymoo.termination.max_time import TimeBasedTermination
    t0 = time.perf_counter()
    proj = Projection(inst, sigma)
    log = Anytime(inst, objs, model, t0, time_limit)

    class Rep(Repair):
        def _do(self, problem, X, **kwargs):
            return np.array([proj(x) for x in X])

    class Prob(Problem):
        def __init__(self):
            super().__init__(n_var=inst.n, n_obj=len(objs), xl=inst.lo, xu=inst.hi)

        def _evaluate(self, X, out, *args, **kwargs):
            F = objective_matrix(objs, inst, X)
            out["F"] = F
            log.offer(X, F)

    if variant == "rnsga":
        algo = RNSGA2(ref_points=np.asarray(model.profiles, dtype=float), pop_size=pop, repair=Rep())
    elif variant == "rmpga":
        algo = NSGA2(pop_size=pop, repair=Rep(), survival=baselines._rmp_survival(model))
    elif variant == "nsga":
        algo = NSGA2(pop_size=pop, repair=Rep())
    else:
        raise ValueError(variant)
    remaining = time_limit - (time.perf_counter() - t0)
    generations = 0
    if remaining > 0:
        # minimize() runs a copy of the algorithm, whose generation count is read from the result
        res = minimize(Prob(), algo, TimeBasedTermination(remaining), seed=seed, verbose=False)
        generations = int(res.algorithm.n_gen or 0)
    return log.result("HEURISTIC", generations=generations, projection_failures=getattr(proj, "failures", 0))


def payoff_table(oracle):
    """Individual minima of the criteria (one weighted call each), the payoff nadir, and the
    objective vectors and allocations of the payoff table."""
    F, X = [], []
    for j in range(oracle.p):
        lam = np.zeros(oracle.p)
        lam[j] = 1.0
        r = oracle.weighted(lam)
        if r["status"] in ("failed", "infeasible"):
            raise RuntimeError(f"individual minimization of criterion {j} failed")
        F.append(r["f"])
        X.append(r["x"])
    F = np.array(F)
    return F.diagonal().copy(), F.max(axis=0), F, np.array(X)


def eps_timed(inst, objs, model, sigma, time_limit=60.0, seed=0, max_level=8):
    """min f_1 subject to f_j <= e_j for j > 1, with e on nested regular grids between the
    individual minima and the payoff nadir (3, 5, 9, ... levels per criterion, each grid
    containing the previous one), new grid points in random order, until the budget is spent;
    then the best key among the computed allocations (the payoff table included)."""
    t0 = time.perf_counter()
    oracle = Oracle(inst, objs, sigma)
    log = Anytime(inst, objs, model, t0, time_limit)
    p = oracle.p
    fmin, fnad, F, X = payoff_table(oracle)
    log.offer(X, F)
    rng = np.random.default_rng(seed)
    lam = np.zeros(p)
    lam[0] = 1.0
    done, points, level = set(), p, 1
    while level <= max_level and time.perf_counter() - t0 < time_limit:
        L = 2 ** level + 1
        new = [c for c in itertools.product(range(L), repeat=p - 1)
               if tuple(Fraction(i, L - 1) for i in c) not in done]
        rng.shuffle(new)
        for c in new:
            if time.perf_counter() - t0 >= time_limit:
                break
            pos = tuple(Fraction(i, L - 1) for i in c)
            done.add(pos)
            w = (math.inf,) + tuple(float(fmin[j + 1] + (fnad[j + 1] - fmin[j + 1]) * float(q))
                                    for j, q in enumerate(pos))
            r = oracle.weighted(lam, w)
            if r["status"] not in ("failed", "infeasible"):
                points += 1
                log.offer(r["x"], r["f"][None, :])
        level += 1
    return log.result("SAMPLE", points=points, levels=level - 1, calls=oracle.calls)


def _scaled_problem_parts(inst, objs, sigma):
    oracle = Oracle(inst, objs, sigma)
    fmin, fnad, _, _ = payoff_table(oracle)
    rng = fnad - fmin
    # payoff range as scale; a flat criterion falls back to the oracle's scale (its upper bound)
    s = np.where(rng > 1e-9 * (1 + np.abs(fnad)), rng, oracle.s)
    x = cp.Variable(inst.n)
    return oracle, s, x, [o.expression(inst, x) for o in objs]


def asf_reference(inst, objs, model, sigma, time_limit=60.0, rho=1e-6):
    """Reference point method: for every profile b, in consultation order,
    min max_j (f_j(x) - b_j)/s_j + rho sum_j f_j(x)/s_j over X, with s the payoff range;
    returns the allocation with the best key among these solutions."""
    t0 = time.perf_counter()
    oracle, s, x, fs = _scaled_problem_parts(inst, objs, sigma)
    p = len(objs)
    t = cp.Variable()
    b = cp.Parameter(p)
    prob = cp.Problem(cp.Minimize(t + rho * cp.sum(cp.hstack([fs[j] / s[j] for j in range(p)]))),
                      x_constraints(inst, sigma, x) + [(fs[j] - b[j]) / s[j] <= t for j in range(p)])
    log = Anytime(inst, objs, model, t0, time_limit)
    stats, solved, failed = {"retries": 0}, 0, 0
    for h in model.order:
        if time.perf_counter() - t0 >= time_limit:
            break
        b.value = np.asarray(model.profiles[h], dtype=float)
        status = solve_with_retries(prob, stats)
        if status in ("optimal", "optimal_inaccurate") and x.value is not None:
            solved += 1
            xv = np.clip(x.value, inst.lo, inst.hi)
            log.offer(xv)
        else:
            failed += 1
    return log.result("HEURISTIC", solves=solved, failures=failed, retries=stats["retries"],
                      complete=solved + failed == model.m, payoff_calls=oracle.calls)


def goal_programming(inst, objs, model, sigma, time_limit=60.0):
    """Lexicographic goal programming. Level l has the goals f_j(x) <= b_j for the l-th
    consulted profile b and minimizes sum_j w_j delta_j / s_j over the deviations delta,
    keeping the deviations reached at the earlier levels; w are the criteria weights of an
    additive model (equal weights for an importance table) and s the payoff range. A final
    step minimizes sum_j f_j/s_j at the reached deviations, so that the result is efficient."""
    t0 = time.perf_counter()
    oracle, s, x, fs = _scaled_problem_parts(inst, objs, sigma)
    p = len(objs)
    w = np.array([float(Fraction(v)) for v in model.weights]) if model.kind == "srmp" else np.ones(p)
    cons = x_constraints(inst, sigma, x)
    stats, levels, status = {"retries": 0}, 0, "optimal"
    for lvl in range(model.m):
        if time.perf_counter() - t0 >= time_limit:
            break
        bvec = model.profiles[model.order[lvl]]
        delta = cp.Variable(p, nonneg=True)
        goals = [fs[j] - delta[j] <= float(bvec[j]) for j in range(p)]
        dev = cp.sum(cp.multiply(w / s, delta))
        prob = cp.Problem(cp.Minimize(dev), cons + goals)
        status = solve_with_retries(prob, stats)
        if status not in ("optimal", "optimal_inaccurate"):
            break
        v = float(prob.value)
        cons = cons + goals + [dev <= v + 1e-8 * (1 + abs(v))]
        levels += 1
    log = Anytime(inst, objs, model, t0, time_limit)
    if levels > 0 and time.perf_counter() - t0 < time_limit:
        prob = cp.Problem(cp.Minimize(cp.sum(cp.hstack([fs[j] / s[j] for j in range(p)]))), cons)
        status = solve_with_retries(prob, stats)
    if x.value is not None and status in ("optimal", "optimal_inaccurate"):
        log.offer(np.clip(x.value, inst.lo, inst.hi))
    return log.result("HEURISTIC", levels=levels, complete=levels == model.m, retries=stats["retries"],
                      payoff_calls=oracle.calls)


def miqcp_scip_indicator(inst, objs, model, sigma, time_limit=60.0, feastol=1e-9):
    """The formulation of certrmp.baselines.miqcp_scip, with the attainment rows
    f_j <= b + M (1 - z) replaced by SCIP indicator constraints z = 1 => f_j <= b."""
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
    for t, tg in zip(inst.tiers, baselines._targets(inst, sigma)):
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
            if ub[j] - b[j] > 0:                      # otherwise the threshold always holds
                M.addConsIndicator(F[j] <= float(b[j]), z)
            row.append(z)
        Z.append(row)
    subsets = model.subsets()
    key, status, xv, den = [], "OPTIMAL", None, 1
    for lvl in range(model.m):
        remaining = time_limit - (time.perf_counter() - t0)
        if remaining <= 0:
            status = "TIMEOUT"
            break
        M.setParam("limits/time", max(remaining, 0.1))
        if model.kind == "srmp":
            W = [Fraction(v) for v in model.weights]
            den = math.lcm(*[v.denominator for v in W])
            coef = [int(v * den) for v in W]
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
                status = "INCONSISTENT"
            else:
                status = st.upper()
            if M.getNSols() == 0:
                break
        v = int(round(M.getObjVal()))
        key.append(Fraction(v, den) if model.kind == "srmp" else v)
        sol = M.getBestSol()
        xv = np.array([M.getSolVal(sol, xi) for xi in x])
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
