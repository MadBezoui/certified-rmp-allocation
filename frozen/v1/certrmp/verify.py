"""Independent verifier for RMP optimality certificates.

Uses only exact rational arithmetic and the certificate itself; it shares no code with
the search. All numbers in a certificate are decimal or 'p/q' strings read as exact
rationals. The verifier checks:
  1. the priority totals (recomputed exactly for single and laminar capacities);
  2. that the witness lies in the priority set and attains the claimed key;
  3. that every supporting halfspace bound is valid (weak duality and convexity);
  4. that every coalition candidate better than the claimed key, at every level, is
     separated by a halfspace, and every tied candidate is separated or covered by the
     next frontier (this implies that no allocation has a larger key);
  5. the weighted-sum tolerance epsilon of the Pareto refinement.
"""
from fractions import Fraction as F
from itertools import combinations
import json
import math

GRID = 2 ** 100   # per-coordinate rounding grid for cheap exact sums


def q(s):
    if s is None or s == "inf":
        return None
    return F(s)


def floor_grid(v):
    return F(math.floor(v * GRID), GRID)


def ceil_grid(v):
    return F(math.ceil(v * GRID), GRID)


class Data:
    def __init__(self, cert):
        I = cert["instance"]
        self.n = I["n"]
        self.d = [F(v) for v in I["d"]]
        self.xp = [F(v) for v in I["x_prev"]]
        self.tiers = [list(t) for t in I["tiers"]]
        self.A = [[F(v) for v in row] for row in I["A"]]
        self.c = [F(v) for v in I["c"]]
        self.structure = I["structure"]
        lo = [F(0)] * self.n
        hi = list(self.d)
        if I.get("r_down") is not None:
            lo = [max(F(0), x - F(r)) for x, r in zip(self.xp, I["r_down"])]
        if I.get("r_up") is not None:
            hi = [min(dd, x + F(r)) for dd, x, r in zip(self.d, self.xp, I["r_up"])]
        self.lo, self.hi = lo, hi
        self.D = [sum((self.d[i] for i in t), F(0)) for t in self.tiers]
        self.objs = cert["objectives"]
        M = cert["model"]
        self.profiles = [[F(repr(float(v))) if not isinstance(v, str) else F(v) for v in row]
                         for row in M["profiles"]]
        self.order = M["order"]
        self.p = len(self.objs)
        if M["kind"] == "srmp":
            w = [F(v) for v in M["weights"]]
            self.importance = lambda S: sum((w[j] for j in S), F(0))
        else:
            tab = {frozenset(k): int(v) for k, v in M["table"]}
            self.importance = lambda S: tab[frozenset(S)]

    def physically_feasible(self):
        if any(a > b for a, b in zip(self.lo, self.hi)):
            return False
        return all(sum((a * x for a, x in zip(row, self.lo)), F(0)) <= cr for row, cr in zip(self.A, self.c))


# -- 1. priority totals --------------------------------------------------------------

def polymatroid_totals(D):
    """Lexicographically minimal tier shortages for 0/1 laminar rows (single included)."""
    sets = {}
    for row, cap in zip(D.A, D.c):
        if any(a not in (0, 1) for a in row):
            return None
        s = frozenset(i for i, a in enumerate(row) if a == 1)
        sets[s] = min(sets.get(s, cap), cap)
    keys = list(sets)
    for a in keys:
        for b in keys:
            if a & b and not (a <= b or b <= a):
                return None
    parent = {s: min((t for t in keys if s < t), key=len, default=None) for s in keys}
    children = {s: [t for t in keys if parent[t] == s] for s in keys}
    roots = [s for s in keys if parent[s] is None]
    cap = {s: sets[s] - sum((D.lo[i] for i in s), F(0)) for s in keys}
    room = [D.hi[i] - D.lo[i] for i in range(D.n)]
    in_root = set().union(*roots) if roots else set()

    def rank(mask):
        def node(s):
            covered = set().union(*children[s]) if children[s] else set()
            tot = sum((room[i] for i in s - covered if mask[i]), F(0))
            tot += sum((node(ch) for ch in children[s]), F(0))
            return min(cap[s], tot)
        tot = sum((node(s) for s in roots), F(0))
        return tot + sum((room[i] for i in range(D.n) if i not in in_root and mask[i]), F(0))

    mask, prev, out = [False] * D.n, F(0), []
    for k, t in enumerate(D.tiers):
        for i in t:
            mask[i] = True
        r = rank(mask)
        out.append(D.D[k] - (r - prev + sum((D.lo[i] for i in t), F(0))))
        prev = r
    return out


def lp_stage_totals(D, stages):
    """Check exact primal points and multipliers of the sequential tier LPs."""
    rows = [list(r) for r in D.A]
    row_lo = [None] * len(rows)
    row_hi = list(D.c)
    out = []
    for k, (t, st) in enumerate(zip(D.tiers, stages)):
        x = [F(v) for v in st["x"]]
        y = [F(v) for v in st["y"]]
        if len(y) != len(rows):
            return None
        cost = [F(0)] * D.n
        for i in t:
            cost[i] = F(-1)
        if not all(D.lo[i] <= x[i] <= D.hi[i] for i in range(D.n)):
            return None
        for r, row in enumerate(rows):
            val = sum((a * xi for a, xi in zip(row, x) if a != 0), F(0))
            if (row_lo[r] is not None and val < row_lo[r]) or (row_hi[r] is not None and val > row_hi[r]):
                return None
        red = list(cost)
        bound = F(0)
        for r, yr in enumerate(y):
            if yr == 0:
                continue
            b = row_lo[r] if yr > 0 else row_hi[r]
            if b is None:
                return None
            bound += yr * b
            for j, a in enumerate(rows[r]):
                if a != 0:
                    red[j] -= yr * a
        bound += sum((min(red[j] * D.lo[j], red[j] * D.hi[j]) for j in range(D.n)), F(0))
        primal = sum((cost[j] * x[j] for j in range(D.n)), F(0))
        if bound != primal:
            return None
        served = -primal
        out.append(D.D[k] - served)
        row = [F(0)] * D.n
        for i in t:
            row[i] = F(1)
        rows.append(row)
        row_lo.append(served)
        row_hi.append(None)
    return out


# -- 2. objective values -----------------------------------------------------------

def objective_upper(D, obj, x):
    """Exact upper bound (per-term rounding up) on one objective at a rational point."""
    a = F(obj["scale"])
    if obj["kind"] == "shortfall":
        tot = sum((ceil_grid(F(w) * (D.d[i] - x[i]) ** 2) for i, w in enumerate(obj["weights"]) if F(w) != 0), F(0))
        return tot / a
    if obj["kind"] == "movement":
        tot = sum((ceil_grid(F(w) * abs(x[i] - D.xp[i])) for i, w in enumerate(obj["weights"]) if F(w) != 0), F(0))
        return tot / a
    tot = F(0)
    for J, g in zip(obj["groups"], obj["gammas"]):
        g = F(g)
        if len(J) < 2 or g <= 0:
            continue
        s = [x[i] / D.d[i] for i in J]
        m = floor_grid(sum(s, F(0)) / len(J))           # any center gives an upper bound
        tot += g / len(J) * sum((ceil_grid((si - m) ** 2) for si in s), F(0))
    return tot / a


# -- 3. supporting halfspace bounds -------------------------------------------------

def halfspace_bound(D, sigma, weights, tangent_x, tangent_s, y_cap, y_tier):
    """Exact lower bound on min over the priority set of sum_j weights_j f_j(x).

    Each convex term is replaced by its tangent at the given point; the capacity and
    tier rows are dualized with y >= 0; the remaining separable problem over the box is
    solved coordinatewise. Every coordinate contribution is rounded down.
    """
    n = D.n
    alpha = [F(0)] * n          # shortfall curvature coefficients
    V = [F(0)] * n              # movement coefficients
    lin = [F(0)] * n
    const = F(0)
    for j, obj in enumerate(D.objs):
        a = weights[j]
        if a == 0:
            continue
        sc = F(obj["scale"])
        if obj["kind"] == "shortfall":
            for i, w in enumerate(obj["weights"]):
                if F(w) != 0:
                    alpha[i] += a * F(w) / sc
        elif obj["kind"] == "movement":
            for i, w in enumerate(obj["weights"]):
                if F(w) != 0:
                    V[i] += a * F(w) / sc
        else:
            for J, g in zip(obj["groups"], obj["gammas"]):
                g = F(g)
                if len(J) < 2 or g <= 0:
                    continue
                kappa = a * g / (sc * len(J))
                s = [tangent_s[i] for i in J]
                m = sum(s, F(0)) / len(J)
                for i, si in zip(J, s):
                    # kappa (s_i - m)^2 tangent: const kappa (m^2 - si^2), slope 2 kappa (si - m) / d_i
                    const += floor_grid(kappa * (m * m - si * si))
                    lin[i] += 2 * kappa * (si - m) / D.d[i]
    for i in range(n):
        if alpha[i] != 0:
            xb = tangent_x[i]
            val = alpha[i] * (D.d[i] - xb) ** 2
            slope = -2 * alpha[i] * (D.d[i] - xb)
            const += floor_grid(val - slope * xb)
            lin[i] += slope
    for r, yr in enumerate(y_cap):
        if yr != 0:
            for i, a in enumerate(D.A[r]):
                if a != 0:
                    lin[i] += yr * a
            const -= yr * D.c[r]
    for k, yk in enumerate(y_tier):
        if yk != 0:
            for i in D.tiers[k]:
                lin[i] -= yk
            const -= yk * (sigma[k] - D.D[k])
    total = const
    for i in range(n):
        cands = [D.lo[i], D.hi[i]]
        if D.lo[i] < D.xp[i] < D.hi[i]:
            cands.append(D.xp[i])
        total += floor_grid(min(lin[i] * z + V[i] * abs(z - D.xp[i]) for z in cands))
    return total


# -- 4. the full check ---------------------------------------------------------------

def verify(cert):
    D = Data(cert)
    out = {"checks": {}, "errors": []}
    claimed = cert["status"]
    if claimed == "PHYSICAL_INFEASIBLE":
        ok = not D.physically_feasible()
        out["checks"]["physical_infeasibility"] = ok
        out["verified"] = ok
        return out
    if not D.physically_feasible():
        out["errors"].append("instance is physically infeasible")
        out["verified"] = False
        return out

    sigma = [F(v) for v in cert["priority"]["totals"]]
    if D.structure in ("single", "laminar"):
        exact = polymatroid_totals(D)
        out["checks"]["priority_exact"] = exact is not None and exact == sigma
        if not out["checks"]["priority_exact"]:
            out["errors"].append("priority totals differ from the exact polymatroid totals")
    elif cert["priority"].get("method") == "lp-exact":
        exact = lp_stage_totals(D, cert["priority"]["stages"])
        out["checks"]["priority_exact"] = exact is not None and exact == sigma
        if not out["checks"]["priority_exact"]:
            out["errors"].append("sequential LP certificates do not confirm the priority totals")
    else:
        out["checks"]["priority_exact"] = False   # declared totals, not certified here

    # Witness.
    x = [F(v) for v in cert["witness"]["x"]]
    inside = all(D.lo[i] <= x[i] <= D.hi[i] for i in range(D.n))
    inside &= all(sum((a * xi for a, xi in zip(row, x) if a != 0), F(0)) <= cr for row, cr in zip(D.A, D.c))
    inside &= all(sum((x[i] for i in t), F(0)) >= D.D[k] - sigma[k] for k, t in enumerate(D.tiers))
    out["checks"]["witness_in_priority_set"] = inside
    fup = [objective_upper(D, obj, x) for obj in D.objs]
    key = []
    for h in D.order:
        C = frozenset(j for j in range(D.p) if fup[j] <= D.profiles[h][j])
        key.append(D.importance(C))
    claimed_key = [F(v) if D_is_srmp(cert) else int(v) for v in cert["key"]]
    out["checks"]["witness_key"] = key >= claimed_key
    out["witness_f_upper"] = [float(v) for v in fup]

    # Halfspaces.
    bounds = []
    for hs in cert["halfspaces"]:
        mu = [F(v) for v in hs["mu"]]
        if any(v < 0 for v in mu):
            bounds.append(None)
            continue
        L = halfspace_bound(D, sigma, mu, [F(v) for v in hs["tangent_x"]], [F(v) for v in hs["tangent_s"]],
                            [max(F(v), F(0)) for v in hs["y_cap"]], [max(F(v), F(0)) for v in hs["y_tier"]])
        bounds.append((mu, L))

    def separated(w):
        for b in bounds:
            if b is None:
                continue
            mu, L = b
            if any(mu[j] > 0 and w[j] is None for j in range(D.p)):
                continue
            if sum((mu[j] * w[j] for j in range(D.p) if mu[j] > 0), F(0)) < L:
                return True
        return False

    # Levels.
    frontier = [tuple([None] * D.p)]
    level_ok = True
    for lvl, rec in enumerate(cert["levels"]):
        b = D.profiles[D.order[lvl]]
        best = claimed_key[lvl]
        nxt = [tuple(q(v) for v in u) for u in rec["next_frontier"]]
        for u in frontier:
            att = [j for j in range(D.p) if u[j] is not None and u[j] <= b[j]]
            free = [j for j in range(D.p) if j not in att]
            for r in range(len(free) + 1):
                for R in combinations(free, r):
                    S = frozenset(att) | frozenset(R)
                    w = tuple((b[j] if u[j] is None else min(u[j], b[j])) if j in S else u[j] for j in range(D.p))
                    v = D.importance(S)
                    if v > best:
                        if not separated(w):
                            level_ok = False
                            out["errors"].append(f"level {lvl}: better candidate not separated")
                    elif v == best:
                        covered = any(all((uu is None) or (ww is not None and ww <= uu)
                                          for ww, uu in zip(w, u2)) for u2 in nxt)
                        if not covered and not separated(w):
                            level_ok = False
                            out["errors"].append(f"level {lvl}: tied candidate neither covered nor separated")
        frontier = nxt
    out["checks"]["levels"] = level_ok

    # Refinement tolerance.
    ref = cert.get("refinement")
    if ref:
        lam = [F(v) for v in ref["lambda"]]
        val = sum((l * f for l, f in zip(lam, fup)), F(0))
        lows = []
        for item in ref["bounds"]:
            u = [q(v) for v in item["u"]]
            nu = [F(v) for v in item["nu"]]
            if any(nu[j] < 0 or (nu[j] > 0 and u[j] is None) for j in range(D.p)):
                lows.append(None)
                continue
            agg = [l + n_ for l, n_ in zip(lam, nu)]
            L = halfspace_bound(D, sigma, agg, [F(v) for v in item["tangent_x"]], [F(v) for v in item["tangent_s"]],
                                [max(F(v), F(0)) for v in item["y_cap"]], [max(F(v), F(0)) for v in item["y_tier"]])
            lows.append(L - sum((nu[j] * u[j] for j in range(D.p) if nu[j] > 0), F(0)))
        covered_final = len(lows) == len(frontier) and all(l is not None for l in lows)
        out["checks"]["refinement_bounds"] = covered_final
        if covered_final:
            eps = val - min(lows)
            out["refinement_epsilon"] = float(eps)
            out["refinement_epsilon_relative"] = float(eps / max(abs(val), F(1, 10 ** 12)))
    out["verified"] = all(v for k, v in out["checks"].items() if k not in ("priority_exact", "refinement_bounds"))
    out["exact"] = out["verified"] and out["checks"]["priority_exact"]
    return out


def D_is_srmp(cert):
    return cert["model"]["kind"] == "srmp"


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as fh:
        cert = json.load(fh)
    res = verify(cert)
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["verified"] else 1)
