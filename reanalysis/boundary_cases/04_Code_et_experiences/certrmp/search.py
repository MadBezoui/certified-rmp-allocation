"""Exact search for the best attainable RMP key.

The attainable requirement vectors form the upper image Y = f(X) + R^p_+, a convex set.
A requirement vector w (entries may be +inf) is attainable iff its finite part lies in
the projection of Y. Each convex call either returns an attainable point (a witness) or
a supporting halfspace mu.y >= L that separates w. Both are cached: a later vector is
attainable if a cached point is componentwise below it, and unattainable if a cached
halfspace (supported on its finite coordinates) separates it.

Lexicographic coalition search (LCS) builds the optimal key profile by profile. At
level h it keeps the frontier U of weakest requirement vectors that attain the optimal
prefix, forms the candidates u ^ b_S for every coalition S, and decides them in
decreasing importance until the best attainable importance is found. Ties are kept in
the next frontier. Enumeration decides every vector of the threshold grid instead.
"""
from dataclasses import dataclass, field
from itertools import combinations, product
import math
import time

import numpy as np

INF = math.inf


def finite_mask(w):
    return np.array([not math.isinf(v) for v in w])


def weaker_or_equal(a, b):
    """True if a >= b componentwise (a imposes no more than b)."""
    return all(x >= y for x, y in zip(a, b))


@dataclass
class Halfspace:
    mu: np.ndarray
    L: float
    tangent: np.ndarray
    y_cap: np.ndarray
    y_tier: np.ndarray
    origin: str

    def separates(self, w, margin=1e-9):
        fm = finite_mask(w)
        if np.any((self.mu > 0) & ~fm):
            return False
        val = float(sum(self.mu[j] * w[j] for j in range(len(w)) if self.mu[j] > 0))
        return val < self.L - margin * (1 + abs(self.L))


@dataclass
class Caches:
    points: list = field(default_factory=list)       # (f, x)
    halfspaces: list = field(default_factory=list)
    use_points: bool = True
    use_halfspaces: bool = True


@dataclass
class Decision:
    feasible: bool
    how: str               # 'point', 'halfspace', 'solve', 'dominance'
    ref: int = -1          # index of point or halfspace
    t: float = None


class Searcher:
    def __init__(self, oracle, model, use_points=True, use_halfspaces=True, time_limit=None):
        self.oracle, self.model = oracle, model
        self.p = oracle.p
        self.caches = Caches(use_points=use_points, use_halfspaces=use_halfspaces)
        self.memo = {}
        self.counts = {"solve": 0, "point": 0, "halfspace": 0, "dominance": 0, "memo": 0}
        self.time_limit = time_limit
        self.start = time.perf_counter()

    def timed_out(self):
        return self.time_limit is not None and time.perf_counter() - self.start > self.time_limit

    # -- caches ---------------------------------------------------------------
    def add_point(self, f, x):
        self.caches.points.append((np.asarray(f, dtype=float), np.asarray(x, dtype=float)))
        return len(self.caches.points) - 1

    def add_halfspace(self, mu, L, tangent, y_cap, y_tier, origin):
        self.caches.halfspaces.append(Halfspace(np.asarray(mu), float(L), tangent, y_cap, y_tier, origin))
        return len(self.caches.halfspaces) - 1

    def initialize(self):
        """Individual minima: one point and one halfspace e_j.y >= min f_j per criterion."""
        for j in range(self.p):
            lam = np.zeros(self.p)
            lam[j] = 1.0
            r = self.oracle.weighted(lam)
            if r["status"] == "failed":
                raise RuntimeError("individual minimum failed")
            self.add_point(r["f"], r["x"])
            self.add_halfspace(lam, r["f"][j], r["x"], r["y_cap"], r["y_tier"], f"min_f{j + 1}")
        self.fmin = np.array([min(pt[0][j] for pt in self.caches.points) for j in range(self.p)])

    def decide(self, w):
        w = tuple(w)
        if w in self.memo:
            self.counts["memo"] += 1
            return self.memo[w]
        fm = finite_mask(w)
        if self.caches.use_points:
            for k, (f, _) in enumerate(self.caches.points):
                if np.all(f[fm] <= np.asarray(w)[fm]):
                    self.counts["point"] += 1
                    d = Decision(True, "point", k)
                    self.memo[w] = d
                    return d
        if self.caches.use_halfspaces:
            for k, hs in enumerate(self.caches.halfspaces):
                if hs.separates(w):
                    self.counts["halfspace"] += 1
                    d = Decision(False, "halfspace", k)
                    self.memo[w] = d
                    return d
        r = self.oracle.achievement(w)
        self.counts["solve"] += 1
        if r["status"] == "failed":
            d = Decision(None, "failed")
            self.memo[w] = d
            return d
        k = self.add_point(r["f"], r["x"])
        if r["t"] <= 0:
            d = Decision(True, "solve", k, r["t"])
        else:
            L = float(sum(r["mu"][j] * w[j] for j in range(self.p) if r["mu"][j] > 0)) + r["t"]
            h = self.add_halfspace(r["mu"], L, r["x"], r["y_cap"], r["y_tier"], "achievement")
            d = Decision(False, "solve", h, r["t"])
        self.memo[w] = d
        return d


@dataclass
class SearchResult:
    status: str
    key: tuple = None
    frontier: list = None
    levels: list = None
    counts: dict = None
    oracle_calls: int = 0
    seconds: float = 0.0
    searcher: object = None


def lcs(oracle, model, use_points=True, use_halfspaces=True, time_limit=None, searcher=None):
    """Lexicographic coalition search. Returns the optimal key and the level records
    needed by the certificate."""
    s = searcher or Searcher(oracle, model, use_points, use_halfspaces, time_limit)
    t0 = time.perf_counter()
    if searcher is None:
        s.initialize()
    p = s.p
    frontier = [tuple([INF] * p)]
    key, levels = [], []
    unresolved = False
    for level in range(model.m):
        b = model.profiles[model.order[level]]
        cand = {}
        for u in frontier:
            attained = [j for j in range(p) if u[j] <= b[j]]
            free = [j for j in range(p) if u[j] > b[j]]
            for r in range(len(free) + 1):
                for R in combinations(free, r):
                    S = frozenset(attained) | frozenset(R)
                    w = tuple(min(u[j], b[j]) if j in S else u[j] for j in range(p))
                    cand[w] = model.importance(S)
        values = sorted(set(cand.values()), reverse=True)
        best, ties, record = None, [], []
        for v in values:
            group = [w for w in cand if cand[w] == v]
            group.sort(key=lambda w: (sum(math.isinf(x) for x in w), sum(x for x in w if not math.isinf(x))),
                       reverse=True)
            for w in group:
                if best is not None:
                    if any(weaker_or_equal(f, w) for f in ties):
                        record.append((w, v, Decision(True, "dominance")))
                        s.counts["dominance"] += 1
                        continue
                    if any(weaker_or_equal(w, f) for f in ties):
                        ties = [f for f in ties if not weaker_or_equal(w, f)]
                        ties.append(w)
                        record.append((w, v, Decision(True, "dominance")))
                        continue
                d = s.decide(w)
                record.append((w, v, d))
                if d.feasible is None:
                    unresolved = True
                elif d.feasible:
                    if best is None:
                        best = v
                    ties = [f for f in ties if not weaker_or_equal(w, f)]
                    if not any(weaker_or_equal(f, w) for f in ties):
                        ties.append(w)
                if s.timed_out():
                    return SearchResult("TIMEOUT", tuple(key), frontier, levels, s.counts,
                                        oracle.calls, time.perf_counter() - t0, s)
            if best is not None:
                break
        key.append(best)
        levels.append({"profile": model.order[level], "frontier": frontier, "best": best,
                       "decisions": record})
        frontier = ties
    status = "UNRESOLVED" if unresolved else "OPTIMAL"
    return SearchResult(status, tuple(key), frontier, levels, s.counts, oracle.calls,
                        time.perf_counter() - t0, s)


def enumerate_grid(oracle, model, time_limit=None):
    """Reference: decide every threshold requirement vector by its own convex call."""
    s = Searcher(oracle, model, use_points=False, use_halfspaces=False, time_limit=time_limit)
    t0 = time.perf_counter()
    s.initialize()
    grid = list(product(*model.threshold_levels()))
    feasible = {}
    unresolved = False
    for w in grid:
        d = s.decide(w)
        feasible[w] = d.feasible
        unresolved |= d.feasible is None
        if s.timed_out():
            return SearchResult("TIMEOUT", None, None, None, s.counts, oracle.calls,
                                time.perf_counter() - t0, s)
    keys = {w: model.requirement_key(w) for w in grid}
    best = max(keys[w] for w in grid if feasible[w])
    better_unresolved = any(feasible[w] is None and keys[w] > best for w in grid)
    opt = [w for w in grid if feasible[w] and keys[w] == best]
    frontier = [w for w in opt if not any(o != w and weaker_or_equal(o, w) for o in opt)]
    status = "UNRESOLVED" if better_unresolved else "OPTIMAL"
    return SearchResult(status, best, frontier, [{"grid": len(grid), "feasible": feasible}], s.counts,
                        oracle.calls, time.perf_counter() - t0, s)


def refine(oracle, frontier, weights=None, rel_slack=1e-7):
    """Pareto refinement: minimize sum_j weights_j f_j over the union of the frontier sets.

    Each requirement is tightened by rel_slack * s_j when possible (then by 1e-9 and 1e-11
    times s_j), so that the witness attains its thresholds with a margin that survives
    exact checking.
    Returns dict(x, f, value, u, solves, per_u) or None.
    """
    lam = np.asarray(weights if weights is not None else 1.0 / oracle.s, dtype=float)
    best, per_u = None, []
    for u in frontier:
        r = None
        for slack in (rel_slack, 1e-9, 1e-11, 0.0):
            w = tuple(v - slack * oracle.s[j] if not math.isinf(v) else v for j, v in enumerate(u))
            r = oracle.weighted(lam, w)
            if r["status"] not in ("failed", "infeasible"):
                break
        per_u.append((u, r))
        if r["status"] in ("failed", "infeasible"):
            continue
        if best is None or r["value"] < best["value"]:
            best = {"x": r["x"], "f": r["f"], "value": r["value"], "u": u}
    if best is None:
        return None
    best["per_u"] = per_u
    best["lam"] = lam
    return best
