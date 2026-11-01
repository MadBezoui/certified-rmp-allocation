"""Convex subproblems over the priority set, compiled once per instance.

Two parametrized programs share the allocation variable x:
  achievement problem   min t  s.t. x in X, f_j(x) - w_j <= t s_j (j = 1..p), t >= -1
  weighted problem      min sum_j lam_j f_j(x)  s.t. x in X, f_j(x) <= w_j
A requirement w_j = +inf is encoded by a value above every attainable f_j, so that
the constraint can never bind. The achievement value t* is <= 0 exactly when the
requirement vector is attainable; when t* > 0 its multipliers mu (sum mu_j s_j = 1)
give the supporting halfspace mu.y >= mu.w + t* of the upper image, which separates w.
"""
import math
import time

import cvxpy as cp
import numpy as np

# warm_start=False builds a fresh Clarabel solver at every call. With cvxpy's default
# (reuse of the cached solver through a data update) the outcome of a call depended on
# the previous calls on the same compiled problem, including spurious solver errors.
SOLVER_OPTS = dict(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10,
                   max_iter=400, warm_start=False)
# Retries after a solver error or a non-optimal status: stronger static regularization,
# then the alternative direct solver. Accuracy only affects how tight the recorded bounds
# are; every claim is re-checked in exact arithmetic by the verifier.
RETRIES = (dict(static_regularization_constant=1e-7), dict(direct_solve_method="faer"))


def solve_with_retries(prob, stats):
    status = "solver_error"
    for k, extra in enumerate(((),) + RETRIES):
        opts = dict(SOLVER_OPTS, **dict(extra))
        try:
            prob.solve(**opts)
            status = prob.status
        except cp.error.SolverError:
            status = "solver_error"
        if status in ("optimal", "optimal_inaccurate", "infeasible"):
            break
        stats["retries"] = stats.get("retries", 0) + 1
    return status


class Oracle:
    def __init__(self, inst, objs, sigma, tau_rel=1e-9):
        self.inst, self.objs = inst, objs
        self.p = len(objs)
        n = inst.n
        self.ub = np.array([o.upper_bound(inst) for o in objs])
        self.s = np.where(self.ub > 0, self.ub, 1.0)
        self.big = self.ub + 2 * self.s
        self.sigma = [float(v) for v in sigma]
        self.tier_targets = [float(inst.d[t].sum()) - sv - tau_rel * (1 + float(inst.d[t].sum()))
                             for t, sv in zip(inst.tiers, self.sigma)]
        x = cp.Variable(n)
        self.x = x
        self.w = cp.Parameter(self.p)
        self.lam = cp.Parameter(self.p, nonneg=True)
        fs = [o.expression(inst, x) for o in objs]
        self.fs = fs
        lo, hi = inst.lo, inst.hi

        def x_constraints():
            box = [x >= lo, x <= hi]
            cap = inst.A @ x <= inst.c
            tiers = [cp.sum(x[t]) >= tt for t, tt in zip(inst.tiers, self.tier_targets)]
            return box, cap, tiers

        t = cp.Variable()
        self.t = t
        box, cap, tiers = x_constraints()
        req = [fs[j] - self.w[j] <= t * self.s[j] for j in range(self.p)]
        self.asf = cp.Problem(cp.Minimize(t), box + [cap] + tiers + req + [t >= -1])
        self.asf_parts = (cap, tiers, req)
        box2, cap2, tiers2 = x_constraints()
        req2 = [fs[j] <= self.w[j] for j in range(self.p)]
        self.ws = cp.Problem(cp.Minimize(cp.sum(cp.hstack([self.lam[j] * fs[j] for j in range(self.p)]))),
                             box2 + [cap2] + tiers2 + req2)
        self.ws_parts = (cap2, tiers2, req2)
        self.calls = 0
        self.solve_time = 0.0
        self.failures = 0
        self.stats = {"retries": 0}

    def encode(self, w):
        return np.array([self.big[j] if math.isinf(w[j]) else w[j] for j in range(self.p)])

    def _duals(self, parts, active):
        cap, tiers, req = parts
        y_cap = np.maximum(np.asarray(cap.dual_value, dtype=float).ravel(), 0.0)
        y_tier = np.array([max(float(c.dual_value), 0.0) for c in tiers])
        mu = np.array([max(float(req[j].dual_value), 0.0) if active[j] else 0.0 for j in range(self.p)])
        return mu, y_cap, y_tier

    def achievement(self, w):
        """Returns dict(status, t, x, f, mu, y_cap, y_tier)."""
        active = [not math.isinf(v) for v in w]
        self.w.value = self.encode(w)
        start = time.perf_counter()
        status = solve_with_retries(self.asf, self.stats)
        self.solve_time += time.perf_counter() - start
        self.calls += 1
        if status not in ("optimal", "optimal_inaccurate") or self.x.value is None:
            self.failures += 1
            return {"status": "failed"}
        x = np.clip(self.x.value, self.inst.lo, self.inst.hi)
        mu, y_cap, y_tier = self._duals(self.asf_parts, active)
        norm = float(mu @ self.s)
        if norm > 0:
            mu, y_cap, y_tier = mu / norm, y_cap / norm, y_tier / norm
        return {"status": status, "t": float(self.t.value), "x": x, "f": self.values(x),
                "mu": mu, "y_cap": y_cap, "y_tier": y_tier}

    def weighted(self, lam, w=None):
        """min lam.f over X with optional requirement w. Returns dict(status, value, x, f, nu, ...)."""
        if w is None:
            w = [math.inf] * self.p
        active = [not math.isinf(v) for v in w]
        self.w.value = self.encode(w)
        self.lam.value = np.asarray(lam, dtype=float)
        start = time.perf_counter()
        status = solve_with_retries(self.ws, self.stats)
        self.solve_time += time.perf_counter() - start
        self.calls += 1
        if status not in ("optimal", "optimal_inaccurate") or self.x.value is None:
            self.failures += 1
            return {"status": "infeasible" if status in ("infeasible", "infeasible_inaccurate") else "failed"}
        x = np.clip(self.x.value, self.inst.lo, self.inst.hi)
        nu, y_cap, y_tier = self._duals(self.ws_parts, active)
        return {"status": status, "value": float(self.ws.value), "x": x, "f": self.values(x),
                "nu": nu, "y_cap": y_cap, "y_tier": y_tier}

    def values(self, x):
        return np.array([o.value(self.inst, x) for o in self.objs])
