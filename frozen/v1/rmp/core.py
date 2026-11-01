"""Reference solver for RMP-optimal allocation under strict service priorities.

Small-instance reference implementation: every threshold requirement pattern is
tested by a convex feasibility problem. It is meant to be slow and transparent,
and to serve as the correctness reference for any accelerated search.
"""
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
import math

import cvxpy as cp
import numpy as np

OPTIMAL_NUMERICAL = "OPTIMAL_NUMERICAL"
PHYSICAL_INFEASIBLE = "PHYSICAL_INFEASIBLE"
UNRESOLVED = "UNRESOLVED"

LP_SOLVER = "HIGHS"
CONIC_SOLVER = "CLARABEL"


@dataclass
class Instance:
    """Physical data of one epoch. Tiers are listed from highest priority."""
    d: np.ndarray                      # requests, shape (n,)
    tiers: list                        # list of index lists, highest priority first
    A: np.ndarray                      # resource matrix, shape (q, n)
    c: np.ndarray                      # capacities, shape (q,)
    x_prev: np.ndarray                 # previous allocation, shape (n,)
    r_down: np.ndarray = None          # ramp limits, None means unlimited
    r_up: np.ndarray = None

    @property
    def n(self):
        return len(self.d)


def physical_constraints(inst, x):
    cons = [x >= 0, x <= inst.d, inst.A @ x <= inst.c]
    if inst.r_down is not None:
        cons.append(x - inst.x_prev >= -inst.r_down)
    if inst.r_up is not None:
        cons.append(x - inst.x_prev <= inst.r_up)
    return cons


def priority_constraints(inst, tol=1e-9):
    """Lexicographic tier-shortage minimization over the reachable physical set.

    Returns (constraints builder, optimal tier shortages) or (None, None) when the
    physical set is empty. Priorities are computed over the set that includes the
    ramp limits, never by intersecting an unconstrained priority face with them.
    """
    x = cp.Variable(inst.n)
    base = physical_constraints(inst, x)
    fixed = []
    shortages = []
    for idx in inst.tiers:
        s = cp.sum(inst.d[idx] - x[idx])
        prob = cp.Problem(cp.Minimize(s), base + fixed)
        prob.solve(solver=LP_SOLVER)
        if prob.status in ("infeasible", "infeasible_inaccurate"):
            return None, None
        if prob.status != "optimal":
            raise RuntimeError(f"priority LP status {prob.status}")
        star = float(prob.value)
        shortages.append(star)
        fixed.append(s <= star + tol * (1 + abs(star)))

    def build(y):
        cons = physical_constraints(inst, y)
        for idx, star in zip(inst.tiers, shortages):
            cons.append(cp.sum(inst.d[idx] - y[idx]) <= star + tol * (1 + abs(star)))
        return cons

    return build, shortages


@dataclass
class Objectives:
    """Three convex objectives to minimize: shortfall, movement, within-tier disparity."""
    w: np.ndarray
    v: np.ndarray
    gamma: np.ndarray                  # one weight per tier
    scale: tuple = (1.0, 1.0, 1.0)
    demand_floor: float = 0.0          # recipients with d <= floor leave the disparity term

    def _groups(self, inst):
        return [[i for i in idx if inst.d[i] > self.demand_floor] for idx in inst.tiers]

    def expressions(self, inst, x):
        a1, a2, a3 = self.scale
        f1 = cp.sum(cp.multiply(self.w, cp.square(inst.d - x))) / a1
        f2 = cp.sum(cp.multiply(self.v, cp.abs(x - inst.x_prev))) / a2
        terms = []
        for k, J in enumerate(self._groups(inst)):
            if len(J) >= 2 and self.gamma[k] > 0:
                s = cp.multiply(x[J], 1.0 / inst.d[J])
                terms.append(self.gamma[k] / len(J) * cp.sum_squares(s - cp.sum(s) / len(J)))
        f3 = (cp.sum(cp.hstack(terms)) if terms else cp.Constant(0.0)) / a3
        return [f1, f2, f3]

    def values(self, inst, x):
        a1, a2, a3 = self.scale
        f1 = float(np.sum(self.w * (inst.d - x) ** 2)) / a1
        f2 = float(np.sum(self.v * np.abs(x - inst.x_prev))) / a2
        f3 = 0.0
        for k, J in enumerate(self._groups(inst)):
            if len(J) >= 2 and self.gamma[k] > 0:
                s = x[J] / inst.d[J]
                f3 += self.gamma[k] / len(J) * float(np.sum((s - s.mean()) ** 2))
        return np.array([f1, f2, f3 / a3])


class RMPModel:
    """Fixed RMP model for minimized criteria.

    profiles: array (m, p), profile h gives upper thresholds b_h.
    order: profile indices in the order they are consulted.
    importance: function frozenset -> comparable value, larger is better. It must be
    monotone for inclusion; check_monotone() verifies this on all 2^p subsets.
    """

    def __init__(self, profiles, order, importance, p):
        self.profiles = np.asarray(profiles, dtype=float)
        self.order = list(order)
        self.importance = importance
        self.p = p

    @classmethod
    def srmp(cls, profiles, order, weights):
        w = [Fraction(str(x)) for x in weights]
        return cls(profiles, order, lambda C: sum((w[j] for j in C), Fraction(0)), len(w))

    @classmethod
    def from_table(cls, profiles, order, table, p):
        """Nonadditive model from an explicit ranking of all criterion subsets."""
        return cls(profiles, order, lambda C: table[frozenset(C)], p)

    def check_monotone(self):
        subsets = [frozenset(j for j in range(self.p) if mask >> j & 1) for mask in range(1 << self.p)]
        for S in subsets:
            for j in range(self.p):
                if j not in S and self.importance(S | {j}) < self.importance(S):
                    return False
        return True

    def coalitions(self, f):
        return [frozenset(j for j in range(self.p) if f[j] <= self.profiles[h, j]) for h in self.order]

    def key(self, f):
        """Ordered profile key; compare lexicographically, larger is better."""
        return tuple(self.importance(C) for C in self.coalitions(f))

    def key_from_requirements(self, u):
        """Key guaranteed by the requirements f_j <= u_j (u_j = inf means none)."""
        return tuple(self.importance(frozenset(j for j in range(self.p) if u[j] <= self.profiles[h, j]))
                     for h in self.order)

    def threshold_levels(self):
        return [sorted(set(self.profiles[:, j].tolist())) + [math.inf] for j in range(self.p)]


@dataclass
class Result:
    status: str
    key: tuple = None
    x: np.ndarray = None
    f: np.ndarray = None
    shortages: list = None
    patterns_tested: int = 0
    convex_calls: int = 0
    excluded: list = field(default_factory=list)       # better patterns proved infeasible
    optimal_patterns: list = field(default_factory=list)
    boundary_flags: list = field(default_factory=list)


def _requirement_problem(inst, obj, build, u, objective=None):
    x = cp.Variable(inst.n)
    fs = obj.expressions(inst, x)
    cons = build(x) + [fs[j] <= u[j] for j in range(len(u)) if math.isfinite(u[j])]
    prob = cp.Problem(cp.Minimize(objective(fs) if objective else 0), cons)
    prob.solve(solver=CONIC_SOLVER)
    return prob, x


def solve(inst, obj, model, tie_weights=(1.0, 1.0, 1.0), feas_tol=1e-7, boundary_tol=1e-6):
    build, shortages = priority_constraints(inst)
    if build is None:
        return Result(status=PHYSICAL_INFEASIBLE)
    levels = model.threshold_levels()
    patterns = list(product(*levels))
    res = Result(status=OPTIMAL_NUMERICAL, shortages=shortages)
    feasible = {}
    unresolved = False
    for u in patterns:
        prob, x = _requirement_problem(inst, obj, build, u)
        res.convex_calls += 1
        if prob.status == "optimal":
            fv = obj.values(inst, x.value)
            ok = all(fv[j] <= u[j] + feas_tol * (1 + abs(u[j])) for j in range(len(u)) if math.isfinite(u[j]))
            feasible[u] = ok
            unresolved |= not ok
        elif prob.status == "infeasible":
            feasible[u] = False
        else:
            feasible[u] = None
            unresolved = True
    res.patterns_tested = len(patterns)
    best = max(model.key_from_requirements(u) for u in patterns if feasible[u])
    res.key = best
    res.excluded = [u for u in patterns if model.key_from_requirements(u) > best]
    if any(feasible[u] is None for u in res.excluded):
        res.status = UNRESOLVED
    opt = [u for u in patterns if feasible[u] and model.key_from_requirements(u) == best]
    weakest = [u for u in opt if not any(o != u and all(o[j] >= u[j] for j in range(len(u))) for o in opt)]
    res.optimal_patterns = weakest
    # Pareto refinement over the whole optimal class: minimize a positive weighted sum
    # over each weakest optimal requirement set; their union is the optimal class.
    lam = np.asarray(tie_weights, dtype=float)
    assert np.all(lam > 0)
    best_val, best_x = math.inf, None
    for u in weakest:
        prob, x = _requirement_problem(inst, obj, build, u, objective=lambda fs: sum(l * f for l, f in zip(lam, fs)))
        res.convex_calls += 1
        if prob.status == "optimal" and prob.value < best_val:
            best_val, best_x = prob.value, x.value.copy()
    if best_x is None:
        res.status = UNRESOLVED
        return res
    res.x = best_x
    res.f = obj.values(inst, best_x)
    if model.key(res.f) != best:
        res.boundary_flags.append("witness key differs from requirement key")
    for h in model.order:
        for j in range(model.p):
            if abs(res.f[j] - model.profiles[h, j]) <= boundary_tol * (1 + abs(model.profiles[h, j])):
                res.boundary_flags.append(f"f{j + 1} at profile {h} threshold")
    if unresolved:
        res.status = UNRESOLVED
    return res
