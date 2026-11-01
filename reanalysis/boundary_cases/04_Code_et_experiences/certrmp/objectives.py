"""Convex objectives to minimize: service shortfall, allocation movement and disparity.

Every objective has the form f(x) = (1/a) * g(x) with a fixed positive scale a.
  shortfall  g = sum_i w_i (d_i - x_i)^2
  movement   g = sum_i v_i |x_i - x_prev_i|
  disparity  g = sum_g gamma_g / |J_g| * sum_{i in J_g} (x_i/d_i - mean_{J_g}(x/d))^2
Weights are nonnegative; a recipient with zero weight does not enter the objective.
Disparity groups contain only recipients whose request exceeds the group's floor.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

SHORTFALL, MOVEMENT, DISPARITY = "shortfall", "movement", "disparity"


@dataclass
class Objective:
    name: str
    kind: str
    weights: np.ndarray = None      # per recipient (shortfall, movement)
    groups: list = None             # index arrays (disparity)
    gammas: np.ndarray = None       # per group (disparity)
    scale: float = 1.0

    def active_groups(self, inst):
        return [(np.asarray(J), float(g)) for J, g in zip(self.groups, self.gammas) if len(J) >= 2 and g > 0]

    def value(self, inst, x):
        x = np.asarray(x, dtype=float)
        if self.kind == SHORTFALL:
            return float(np.sum(self.weights * (inst.d - x) ** 2)) / self.scale
        if self.kind == MOVEMENT:
            return float(np.sum(self.weights * np.abs(x - inst.x_prev))) / self.scale
        total = 0.0
        for J, g in self.active_groups(inst):
            s = x[J] / inst.d[J]
            total += g / len(J) * float(np.sum((s - s.mean()) ** 2))
        return total / self.scale

    def expression(self, inst, x):
        """Convex expression whose minimum over auxiliary variables equals the objective.

        It is exact wherever the objective is minimized or bounded above."""
        if self.kind == SHORTFALL:
            nz = np.flatnonzero(self.weights)
            return cp.sum(cp.multiply(self.weights[nz], cp.square(inst.d[nz] - x[nz]))) / self.scale
        if self.kind == MOVEMENT:
            nz = np.flatnonzero(self.weights)
            return cp.sum(cp.multiply(self.weights[nz], cp.abs(x[nz] - inst.x_prev[nz]))) / self.scale
        # The variance equals min over a free center c of sum (s_i - c)^2. Inside a
        # minimization or an upper-bound constraint the center can be a variable, which
        # keeps the conic model sparse.
        terms = []
        for J, g in self.active_groups(inst):
            center = cp.Variable()
            s = cp.multiply(x[J], 1.0 / inst.d[J])
            terms.append(g / len(J) * cp.sum_squares(s - center))
        if not terms:
            return cp.Constant(0.0)
        return cp.sum(cp.hstack(terms)) / self.scale

    def upper_bound(self, inst):
        """A valid upper bound on the objective over the box lo <= x <= hi."""
        lo, hi = inst.lo, inst.hi
        if self.kind == SHORTFALL:
            return float(np.sum(self.weights * (inst.d - lo) ** 2)) / self.scale
        if self.kind == MOVEMENT:
            dev = np.maximum(np.abs(lo - inst.x_prev), np.abs(hi - inst.x_prev))
            return float(np.sum(self.weights * dev)) / self.scale
        # Service fractions lie in [0, 1]; a variance of such numbers is at most 1/4.
        return sum(g / 4 for _, g in self.active_groups(inst)) / self.scale

    def to_dict(self):
        return {"name": self.name, "kind": self.kind, "scale": self.scale,
                "weights": None if self.weights is None else self.weights.tolist(),
                "groups": None if self.groups is None else [np.asarray(J).tolist() for J in self.groups],
                "gammas": None if self.gammas is None else np.asarray(self.gammas).tolist()}


def disparity_groups(inst, floor_ratio=1e-3):
    """Within-tier groups of recipients whose request exceeds floor_ratio times the mean request."""
    floor = floor_ratio * float(np.mean(inst.d)) if inst.n else 0.0
    return [np.asarray([i for i in t if inst.d[i] > floor], dtype=int) for t in inst.tiers]


def standard_objectives(inst, floor_ratio=1e-3):
    """The three core objectives with frozen scale conventions.

    Shortfall is divided by sum d_i^2, movement by sum d_i, disparity by the number of tiers,
    so that each lies in [0, 1] on the box (disparity in [0, 1/4]).
    """
    n = inst.n
    groups = disparity_groups(inst, floor_ratio)
    return [
        Objective("shortfall", SHORTFALL, weights=np.ones(n), scale=float(np.sum(inst.d ** 2))),
        Objective("movement", MOVEMENT, weights=np.ones(n), scale=float(np.sum(inst.d))),
        Objective("disparity", DISPARITY, groups=groups, gammas=np.ones(len(groups)),
                  scale=float(len(groups))),
    ]


def regional_objectives(inst, regions, floor_ratio=1e-3):
    """Objective vectors for dimension stress tests: one shortfall objective per region,
    then movement and disparity. With R regions this gives p = R + 2 objectives."""
    objs = []
    for k, reg in enumerate(regions):
        w = np.zeros(inst.n)
        w[reg] = 1.0
        objs.append(Objective(f"shortfall_r{k + 1}", SHORTFALL, weights=w,
                              scale=float(np.sum(inst.d[reg] ** 2))))
    core = standard_objectives(inst, floor_ratio)
    return objs + core[1:]


def values(objs, inst, x):
    return np.array([o.value(inst, x) for o in objs])
