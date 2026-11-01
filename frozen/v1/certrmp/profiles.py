"""Preference-model families used by the benchmark (frozen before the main runs).

uniform      dominance-ordered profiles at random levels of the payoff range
near_front   each profile is a computed Pareto point shifted by up to 5% of the range
easy         all profiles at or above the payoff nadir (everything attainable)
unattainable all profiles below the individual minima (nothing attainable)
adjacent     each profile equals a computed Pareto point (thresholds at the frontier)
"""
import math

import numpy as np

from .model import RMPModel, random_monotone_table

SPAN_FLOOR = 0.05          # criteria without trade-off get a span of 5% of their minimum


def _sig(v, digits=6):
    v = float(v)
    return 0.0 if v == 0 else float(f"{v:.{digits}g}")


def importance(p, rng, kind):
    if kind == "srmp":
        w = np.round(rng.dirichlet(np.ones(p)), 2)
        w[-1] = round(1 - w[:-1].sum(), 2)
        return np.maximum(w, 0.0).tolist()
    return random_monotone_table(p, rng)


def build_model(profiles, order, kind, imp, p):
    if kind == "srmp":
        return RMPModel.srmp(profiles, order, imp)
    return RMPModel.from_table(profiles, order, imp, p)


def generate(oracle, m, seed, family="uniform", kind="srmp"):
    """Returns (model, info). Uses only the oracle of the instance (payoff range and
    weighted-sum Pareto points), never the solver under test."""
    rng = np.random.default_rng(seed)
    p = oracle.p
    F = payoff_table(oracle)
    fmin, fnad = F.diagonal().copy(), F.max(axis=0)
    flat = (fnad - fmin) <= 1e-6 * (1.0 + np.abs(fmin))
    span = np.where(flat, SPAN_FLOOR * np.maximum(np.abs(fmin), 1e-6), fnad - fmin)
    if family == "uniform":
        levels = np.sort(rng.uniform(0.05, 0.95, (m, p)), axis=0)
        prof = fmin + levels * span
    elif family in ("near_front", "adjacent"):
        prof = []
        for h in range(m):
            lam = rng.dirichlet(np.ones(p)) / span
            fp = oracle.weighted(lam)["f"]
            shift = rng.uniform(-0.05, 0.05, p) * span if family == "near_front" else np.zeros(p)
            prof.append(fp + shift)
        prof = np.array(prof)
    elif family == "easy":
        prof = fnad + rng.uniform(0.0, 0.2, (m, p)) * span
    elif family == "unattainable":
        prof = fmin - rng.uniform(0.05, 0.5, (m, p)) * span
    else:
        raise ValueError(family)
    if family == "adjacent":
        profiles = prof.tolist()                       # exact float frontier values
    else:
        profiles = [[_sig(v) for v in row] for row in prof]
    order = rng.permutation(m).tolist()
    imp = importance(p, rng, kind)
    model = build_model(profiles, order, kind, imp, p)
    return model, {"fmin": fmin.tolist(), "fnad": fnad.tolist(), "family": family, "kind": kind,
                   "flat_criteria": np.flatnonzero(flat).tolist()}


def payoff_table(oracle):
    """Lexicographic payoff table: row j minimizes f_j, then the sum of the other
    (normalized) objectives on that minimum face, so that no row is only weakly efficient."""
    p = oracle.p
    rows = []
    for j in range(p):
        lam = np.zeros(p)
        lam[j] = 1.0
        first = oracle.weighted(lam)
        fj = float(first["f"][j])
        lam2 = 1.0 / oracle.s
        lam2[j] = 0.0
        w = [math.inf] * p
        w[j] = fj + 1e-9 * (1.0 + abs(fj))
        second = oracle.weighted(lam2, w)
        ok = second["status"] in ("optimal", "optimal_inaccurate")
        rows.append(second["f"] if ok else first["f"])
    return np.array(rows)
