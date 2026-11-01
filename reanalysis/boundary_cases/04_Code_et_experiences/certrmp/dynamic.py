"""Repeated decisions with safe reuse of separating halfspaces.

A supporting-halfspace certificate is a tuple (mu, tangent point, row multipliers). For
any data, the bound it yields is valid: convexity gives the tangent inequality and weak
duality holds for every nonnegative multiplier. Reusing a certificate at a new epoch
therefore only requires recomputing its bound with the new requests, capacities and
previous allocation. Whether the recomputed bound still separates a candidate is
decided afresh; reuse can save convex calls but never changes a conclusion.
"""
import math
import time

import cvxpy as cp
import numpy as np

from .instance import Instance, priority_totals
from .oracle import Oracle, SOLVER_OPTS, solve_with_retries
from .search import Searcher, lcs, refine
from . import certify, verify as verifier


def float_bound(inst, objs, sigma, mu, xbar, y_cap, y_tier):
    """Floating-point evaluation of the halfspace bound (mirrors verify.halfspace_bound)."""
    n = inst.n
    alpha, V, lin = np.zeros(n), np.zeros(n), np.zeros(n)
    const = 0.0
    for j, o in enumerate(objs):
        a = mu[j]
        if a == 0:
            continue
        if o.kind == "shortfall":
            alpha += a * o.weights / o.scale
        elif o.kind == "movement":
            V += a * o.weights / o.scale
        else:
            for J, g in o.active_groups(inst):
                kappa = a * g / (o.scale * len(J))
                s = xbar[J] / inst.d[J]
                m = s.mean()
                const += kappa * float(np.sum(m * m - s * s))
                lin[J] += 2 * kappa * (s - m) / inst.d[J]
    slope = -2 * alpha * (inst.d - xbar)
    const += float(np.sum(alpha * (inst.d - xbar) ** 2 - slope * xbar))
    lin += slope
    lin += inst.A.T @ y_cap
    const -= float(y_cap @ inst.c)
    D = np.array([inst.d[t].sum() for t in inst.tiers])
    for k, t in enumerate(inst.tiers):
        lin[t] -= y_tier[k]
    const -= float(y_tier @ (np.asarray(sigma, dtype=float) - D))
    lo, hi, xp = inst.lo, inst.hi, inst.x_prev
    cands = [lin * lo + V * np.abs(lo - xp), lin * hi + V * np.abs(hi - xp)]
    inside = (lo < xp) & (xp < hi)
    mid = np.where(inside, lin * xp, np.inf)
    return const + float(np.sum(np.minimum(np.minimum(cands[0], cands[1]), mid)))


def emergency_instance(inst):
    """Relax requests and capacities just enough for the lowest reachable allocation to fit.

    For nonnegative A the relaxed set contains lo, and the violations it permits are the
    smallest unavoidable ones coordinatewise and row by row."""
    from fractions import Fraction
    from .instance import ex
    lo, _ = inst.exact_bounds()

    def up(v):                       # a 9-decimal value >= v, exact under the decimal convention
        return math.ceil(v * 10 ** 9) / 10 ** 9

    d = np.array([max(float(inst.d[i]), up(lo[i])) if lo[i] > ex(inst.d[i]) else inst.d[i] for i in range(inst.n)])
    c = []
    for r in range(len(inst.c)):
        need = sum((ex(a) * lo[i] for i, a in enumerate(inst.A[r]) if a != 0), Fraction(0))
        c.append(up(need) if need > ex(inst.c[r]) else float(inst.c[r]))
    c = np.array(c)
    return Instance(d=d, x_prev=inst.x_prev, tiers=inst.tiers, A=inst.A, c=c, structure=inst.structure,
                    r_down=inst.r_down, r_up=inst.r_up, meta=dict(inst.meta, emergency=True))


class RMPController:
    """Per-epoch RMP-optimal allocation with optional halfspace reuse and certification."""

    def __init__(self, model, objectives_for, reuse=True, certify_each=True, carry_limit=200, warm=False):
        """warm=True also skips the p individual minimizations when carried halfspaces exist:
        the search then starts from the re-validated evidence of the previous epoch only."""
        self.warm = warm
        self.model = model
        self.objectives_for = objectives_for
        self.reuse = reuse
        self.certify_each = certify_each
        self.carry = []
        self.carry_limit = carry_limit
        self.prev_key = None

    def step(self, inst):
        rec = {}
        t0 = time.perf_counter()
        sigma, exact = priority_totals(inst)
        if sigma is None:
            inst = emergency_instance(inst)
            sigma, exact = priority_totals(inst)
            rec["emergency"] = True
        else:
            rec["emergency"] = False
        objs = self.objectives_for(inst)
        oracle = Oracle(inst, objs, sigma)
        s = Searcher(oracle, self.model)
        if not (self.warm and self.reuse and self.carry):
            s.initialize()
        reused = 0
        if self.reuse:
            for (mu, xbar, yc, yt) in self.carry:
                L = float_bound(inst, objs, sigma, mu, xbar, yc, yt)
                s.add_halfspace(mu, L, xbar, yc, yt, "reused")
                reused += 1
        res = lcs(oracle, self.model, searcher=s)
        ref = refine(oracle, res.frontier)
        rec.update(key=res.key, calls=oracle.calls, counts=dict(res.counts), carried=reused,
                   separated_by_reused=sum(1 for lvl in res.levels for (_, _, d) in lvl["decisions"]
                                           if d.feasible is False and d.how == "halfspace"
                                           and s.caches.halfspaces[d.ref].origin == "reused"))
        rec["key_switch"] = self.prev_key is not None and res.key != self.prev_key
        self.prev_key = res.key
        x = ref["x"]
        if self.certify_each:
            cert = certify.build(inst, objs, self.model, sigma, res, ref, exact)
            out = verifier.verify(cert)
            rec["verified"] = bool(out["verified"] and out["exact"])
            if cert["witness"]["repaired"]:
                x = np.array([float(v) for v in certify.repair(inst, sigma, ref["x"])])
        rec["seconds"] = time.perf_counter() - t0
        # Carry the halfspaces that did separate something, newest first.
        keep = []
        for hs in s.caches.halfspaces:
            if hs.origin in ("achievement", "reused"):
                keep.append((hs.mu, hs.tangent, hs.y_cap, hs.y_tier))
        self.carry = keep[-self.carry_limit:]
        return x, rec


class BaselineController:
    """Operational rules on the same priority set (projected feedback and its relatives)."""

    def __init__(self, rule, alpha=0.25, kappa=0.1, lam1=3.0, lam2=1.0):
        self.rule, self.alpha, self.kappa, self.lam1, self.lam2 = rule, alpha, kappa, lam1, lam2
        self.smoothed = None

    def step(self, inst):
        t0 = time.perf_counter()
        sigma, exact = priority_totals(inst)
        emergency = sigma is None
        if emergency:
            inst = emergency_instance(inst)
            sigma, exact = priority_totals(inst)
        n = inst.n
        x = cp.Variable(n)
        targets = [float(inst.d[t].sum()) - float(s) - 1e-9 * (1 + float(inst.d[t].sum()))
                   for t, s in zip(inst.tiers, sigma)]
        cons = [x >= inst.lo, x <= inst.hi, inst.A @ x <= inst.c] + \
               [cp.sum(x[t]) >= tg for t, tg in zip(inst.tiers, targets)]
        Q = 0.5 * cp.sum_squares(x - inst.d)
        prev = inst.x_prev
        if self.rule == "myopic":
            obj = Q
        elif self.rule == "feedback":
            obj = 0.5 * cp.sum_squares(x - ((1 - self.alpha) * prev + self.alpha * inst.d))
        elif self.rule == "smoothing":
            self.smoothed = inst.d.copy() if self.smoothed is None else \
                (1 - self.alpha) * self.smoothed + self.alpha * inst.d
            obj = 0.5 * cp.sum_squares(x - self.smoothed)
        elif self.rule == "absolute":
            obj = Q + self.kappa * cp.norm1(x - prev)
        elif self.rule == "robd":
            star = cp.Variable(n)
            p0 = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(star - inst.d)),
                            [star >= inst.lo, star <= inst.hi, inst.A @ star <= inst.c] +
                            [cp.sum(star[t]) >= tg for t, tg in zip(inst.tiers, targets)])
            solve_with_retries(p0, {})
            xs = star.value
            obj = Q + self.lam1 / 2 * cp.sum_squares(x - prev) + self.lam2 / 2 * cp.sum_squares(x - xs)
        else:
            raise ValueError(self.rule)
        prob = cp.Problem(cp.Minimize(obj), cons)
        status = solve_with_retries(prob, {})
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"{self.rule} step failed: {status}")
        xv = np.clip(x.value, inst.lo, inst.hi)
        return xv, {"emergency": emergency, "seconds": time.perf_counter() - t0}
