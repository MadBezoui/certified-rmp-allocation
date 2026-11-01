"""Seeded instance and preference-model generators (frozen benchmark protocol).

All generated numbers are rounded to at most six decimals, so that each instance is an
exact rational instance when read from its decimal representation.
"""
import numpy as np

from .instance import Instance, SINGLE, LAMINAR, GENERAL
from .model import RMPModel, random_monotone_table

TIER_SHARES = (0.2, 0.3, 0.5)
SCARCITY = 0.75


def _r(v, k=4):
    return np.round(np.asarray(v, dtype=float), k)


def make_instance(n, structure, seed, ramps=False, scarcity=SCARCITY, ramp=None):
    """ramps=True means ramp limits of 0.3 d; ramp=r sets them to r d (sensitivity runs)."""
    rng = np.random.default_rng(seed)
    d = _r(np.exp(rng.normal(0.0, 0.5, n)))
    d = np.maximum(d, 0.01)
    d_prev = d * np.exp(rng.normal(0.0, 0.25, n))
    x_prev = _r(d_prev * rng.uniform(0.5, 1.0, n))
    perm = rng.permutation(n)
    cuts = np.cumsum([int(round(s * n)) for s in TIER_SHARES[:-1]])
    tiers = [np.sort(part) for part in np.split(perm, cuts)]
    rows, caps = [], []
    if structure == SINGLE:
        rows.append(np.ones(n))
        caps.append(scarcity * d.sum())
    elif structure == LAMINAR:
        rows.append(np.ones(n))
        caps.append(scarcity * d.sum())
        order = rng.permutation(n)
        for region in np.array_split(order, 4):
            row = np.zeros(n)
            row[region] = 1
            rows.append(row)
            caps.append(rng.uniform(0.7, 0.95) * d[region].sum())
            for feeder in np.array_split(region, max(1, int(np.ceil(len(region) / 10)))):
                if len(feeder) < 2:
                    continue
                row = np.zeros(n)
                row[feeder] = 1
                rows.append(row)
                caps.append(rng.uniform(0.75, 1.0) * d[feeder].sum())
    elif structure == GENERAL:
        # Two crossing partitions (network regions and supply contracts) under a global
        # capacity: 0/1 rows that do not form a laminar family.
        rows.append(np.ones(n))
        caps.append(scarcity * d.sum())
        for parts in (4, 3):
            for group in np.array_split(rng.permutation(n), parts):
                row = np.zeros(n)
                row[group] = 1
                rows.append(row)
                caps.append(rng.uniform(0.7, 0.95) * d[group].sum())
    elif structure == "weighted":
        # Heterogeneous resource coefficients: used only to illustrate degeneracy.
        q = 4
        A = np.zeros((q, n))
        for i in range(n):
            used = rng.choice(q, size=2, replace=False)
            A[used, i] = _r(rng.uniform(0.5, 1.5, 2), 3)
        for r in range(q):
            rows.append(A[r])
            caps.append(rng.uniform(0.6, 0.9) * float(A[r] @ d))
    else:
        raise ValueError(structure)
    A = np.array(rows)
    c = _r(caps)
    if ramp is None and ramps:
        ramp = 0.3
    r = None
    if ramp is not None:
        # With ramps the previous allocation must be reachable history: it is made feasible
        # for the current requests and capacities (2 % slack), so that the physical set
        # contains it and the ramps only limit how far the new allocation can move.
        xp = np.minimum(x_prev, d)
        load = A @ xp
        scale = min(1.0, float(np.min(0.98 * c[load > 0] / load[load > 0]))) if np.any(load > 0) else 1.0
        x_prev = np.floor(xp * scale * 1e4) / 1e4
        r = _r(ramp * d)
    return Instance(d=d, x_prev=x_prev, tiers=tiers, A=A, c=c,
                    structure=GENERAL if structure == "weighted" else structure,
                    r_down=r, r_up=r, meta={"n": n, "structure": structure, "seed": seed, "ramps": ramp is not None,
                                            "ramp": ramp, "scarcity": scarcity})


def payoff_range(oracle):
    """Individual minima and the payoff-table nadir estimate of each objective."""
    p = oracle.p
    F = []
    for j in range(p):
        lam = np.zeros(p)
        lam[j] = 1.0
        r = oracle.weighted(lam)
        F.append(r["f"])
    F = np.array(F)
    return F.diagonal().copy(), F.max(axis=0)


def _sig(v, digits=6):
    v = float(v)
    if v == 0:
        return 0.0
    return float(f"{v:.{digits}g}")


def make_model(fmin, fnad, m, seed, kind="srmp"):
    """Dominance-ordered profiles inside the payoff range, a random consultation order
    and random importance (additive weights or a nonadditive monotone table)."""
    rng = np.random.default_rng(seed)
    p = len(fmin)
    levels = np.sort(rng.uniform(0.05, 0.95, (m, p)), axis=0)       # increasing threshold = easier
    span = np.maximum(fnad - fmin, 0.0)
    profiles = np.array([[_sig(fmin[j] + levels[h, j] * span[j]) for j in range(p)] for h in range(m)])
    order = rng.permutation(m).tolist()
    if kind == "srmp":
        w = rng.dirichlet(np.ones(p))
        w = np.round(w, 2)
        w[-1] = round(1 - w[:-1].sum(), 2)
        w = np.maximum(w, 0.0)
        return RMPModel.srmp(profiles, order, w.tolist())
    table = random_monotone_table(p, rng)
    return RMPModel.from_table(profiles, order, table, p)
