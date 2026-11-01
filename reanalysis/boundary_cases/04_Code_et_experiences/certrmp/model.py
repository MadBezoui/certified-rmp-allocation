"""RMP preference models for criteria to minimize.

A model has m reference profiles b_h in R^p, an order sigma in which profiles are
consulted, and an importance value I(S) for every coalition S of criteria, monotone for
inclusion. Allocation x attains criterion j at profile h when f_j(x) <= b_hj (equality
counts as attained). Its key is (I(C_sigma1(x)), ..., I(C_sigmam(x))), compared
lexicographically; a larger key is preferred and equal keys are indifferent. Keys are
ordinal: no arithmetic is performed on them.
"""
from fractions import Fraction
from itertools import combinations
import math

import numpy as np


class RMPModel:
    def __init__(self, profiles, order, importance, p, kind="table", weights=None, table=None):
        self.profiles = np.atleast_2d(np.asarray(profiles, dtype=float))
        if any(isinstance(h, bool) or int(h) != h for h in order):
            raise ValueError("profile order must contain integer indices")
        self.order = [int(h) for h in order]
        self.importance = importance
        self.p = int(p)
        self.kind = kind
        self.weights = weights
        self.table = table
        if self.p <= 0 or self.profiles.ndim != 2 or self.profiles.shape[0] == 0 or self.profiles.shape[1] != self.p:
            raise ValueError("profile dimension differs from p")
        if not np.isfinite(self.profiles).all():
            raise ValueError("profiles must contain finite values")
        if sorted(self.order) != list(range(len(self.profiles))):
            raise ValueError("order must be a permutation of the profiles")
        if self.kind == "srmp":
            if self.weights is None or len(self.weights) != self.p or any(Fraction(x) < 0 for x in self.weights):
                raise ValueError("SRMP weights must be nonnegative and have one entry per criterion")
        elif self.kind == "table":
            if self.table is None or len(self.table) != 2 ** self.p:
                raise ValueError("coalition table must be complete and monotone")
            try:
                monotone = self.check_monotone()
            except (KeyError, IndexError) as exc:
                raise ValueError("coalition table must be complete and monotone") from exc
            if not monotone:
                raise ValueError("coalition table must be monotone")
        else:
            raise ValueError("unknown preference model")

    @property
    def m(self):
        return len(self.profiles)

    @classmethod
    def srmp(cls, profiles, order, weights):
        """Additive coalition importance with exact decimal weights."""
        w = [Fraction(str(float(x))) for x in weights]
        if any(x < 0 for x in w):
            raise ValueError("weights must be nonnegative")
        return cls(profiles, order, lambda C: sum((w[j] for j in C), Fraction(0)), len(w),
                   kind="srmp", weights=[str(x) for x in w])

    @classmethod
    def from_table(cls, profiles, order, table, p):
        """General monotone importance given as integer ranks for all 2^p coalitions."""
        if any(isinstance(v, bool) or int(v) != v for v in table.values()):
            raise ValueError("coalition ranks must be integers")
        tab = {frozenset(k): int(v) for k, v in table.items()}
        return cls(profiles, order, lambda C: tab[frozenset(C)], p, kind="table",
                   table={tuple(sorted(k)): v for k, v in tab.items()})

    def subsets(self):
        return [frozenset(c) for r in range(self.p + 1) for c in combinations(range(self.p), r)]

    def check_monotone(self):
        for S in self.subsets():
            for j in range(self.p):
                if j not in S and self.importance(S | {j}) < self.importance(S):
                    return False
        return True

    def coalitions(self, f):
        return [frozenset(j for j in range(self.p) if f[j] <= self.profiles[h, j]) for h in self.order]

    def key(self, f):
        return tuple(self.importance(C) for C in self.coalitions(f))

    def requirement_coalition(self, u, level):
        b = self.profiles[self.order[level]]
        return frozenset(j for j in range(self.p) if u[j] <= b[j])

    def requirement_key(self, u):
        return tuple(self.importance(self.requirement_coalition(u, l)) for l in range(self.m))

    def threshold_levels(self):
        return [sorted(set(self.profiles[:, j].tolist())) + [math.inf] for j in range(self.p)]

    def to_dict(self):
        return {"kind": self.kind, "profiles": self.profiles.tolist(), "order": self.order,
                "weights": self.weights,
                "table": None if self.table is None else [[list(k), v] for k, v in self.table.items()]}


def model_from_dict(d):
    if d["kind"] == "srmp":
        return _srmp_exact(d)
    return RMPModel.from_table(d["profiles"], d["order"], {tuple(k): v for k, v in d["table"]}, len(d["profiles"][0]))


def _srmp_exact(d):
    w = [Fraction(x) for x in d["weights"]]
    return RMPModel(d["profiles"], d["order"], lambda C: sum((w[j] for j in C), Fraction(0)), len(w),
                    kind="srmp", weights=d["weights"])


def random_monotone_table(p, rng, levels=None):
    """A random monotone weak order on coalitions, generally not additive.

    Coalitions are ranked by a random linear extension of inclusion, then adjacent
    ranks are merged at random to create ties.
    """
    subsets = [frozenset(c) for r in range(p + 1) for c in combinations(range(p), r)]
    remaining = set(subsets)
    order = []
    while remaining:
        minimal = [S for S in remaining if not any(T < S for T in remaining)]
        S = minimal[rng.integers(len(minimal))]
        order.append(S)
        remaining.remove(S)
    rank, current = {}, 0
    for i, S in enumerate(order):
        if i > 0 and rng.random() > 0.3:
            current += 1
        rank[S] = current
    # Repair monotonicity broken by merging: rank(S) >= rank(T) for T subset S holds since
    # ranks are nondecreasing along a linear extension.
    return {tuple(sorted(S)): v for S, v in rank.items()}
