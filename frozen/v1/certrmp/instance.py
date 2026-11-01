"""Allocation instances: physical set, priority tiers and exact priority totals.

An instance describes one decision epoch. Allocations x satisfy
    lo <= x <= hi,  A x <= c,
with lo = max(0, x_prev - r_down) and hi = min(d, x_prev + r_up). The matrix A is
nonnegative. Tiers partition the recipients, highest priority first.
"""
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

SINGLE, LAMINAR, GENERAL = "single", "laminar", "general"


@dataclass
class Instance:
    d: np.ndarray
    x_prev: np.ndarray
    tiers: list
    A: np.ndarray
    c: np.ndarray
    structure: str = GENERAL
    r_down: np.ndarray = None
    r_up: np.ndarray = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.d = np.asarray(self.d, dtype=float)
        self.x_prev = np.asarray(self.x_prev, dtype=float)
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.c = np.asarray(self.c, dtype=float)
        self.tiers = [np.asarray(t, dtype=int) for t in self.tiers]
        if np.any(self.A < 0) or np.any(self.d < 0):
            raise ValueError("requests and resource coefficients must be nonnegative")
        seen = np.sort(np.concatenate(self.tiers))
        if not np.array_equal(seen, np.arange(self.n)):
            raise ValueError("tiers must partition the recipients")

    @property
    def n(self):
        return len(self.d)

    @property
    def lo(self):
        lo = np.zeros(self.n)
        if self.r_down is not None:
            lo = np.maximum(lo, self.x_prev - self.r_down)
        return lo

    @property
    def hi(self):
        hi = self.d.copy()
        if self.r_up is not None:
            hi = np.minimum(hi, self.x_prev + self.r_up)
        return hi

    def physically_feasible(self):
        """Exact test (A >= 0): the box is nonempty and its lowest corner fits."""
        lo, hi = self.exact_bounds()
        if any(a > b for a, b in zip(lo, hi)):
            return False
        return all(row_dot(self.A[r], lo) <= ex(self.c[r]) for r in range(len(self.c)))

    def exact_bounds(self):
        """Exact box bounds computed from the decimal data, not from rounded floats."""
        xp, d = exact_vec(self.x_prev), exact_vec(self.d)
        lo = [Fraction(0)] * self.n
        hi = list(d)
        if self.r_down is not None:
            lo = [max(Fraction(0), a - ex(r)) for a, r in zip(xp, self.r_down)]
        if self.r_up is not None:
            hi = [min(b, a + ex(r)) for b, a, r in zip(d, xp, self.r_up)]
        return lo, hi

    def tier_rows(self):
        """Rows G x <= h encoding S_k(x) <= sigma_k, given sigma, as -sum_{T_k} x <= sigma_k - D_k."""
        return [np.asarray(t) for t in self.tiers]


def ex(a):
    """Exact rational read from the shortest decimal representation of a binary64 value."""
    return Fraction(repr(float(a)))


def exact_vec(v):
    return [ex(a) for a in v]


def row_dot(row, xs):
    return sum((ex(a) * x for a, x in zip(row, xs) if a != 0), Fraction(0))


# ---------------------------------------------------------------------------
# Laminar capacity families and exact priority totals
# ---------------------------------------------------------------------------

def laminar_tree(inst):
    """Distinct sets, their capacities and parent links for 0/1 laminar rows of A.

    Identical rows are merged with the smallest capacity. parent[r] is the smallest set
    strictly containing set r, or -1. Raises ValueError if the rows are not laminar.
    """
    caps = {}
    for row, cap in zip(inst.A, inst.c):
        if not np.all((row == 0) | (row == 1)):
            raise ValueError("laminar rows must be 0/1 indicators")
        s = frozenset(np.flatnonzero(row).tolist())
        caps[s] = min(caps.get(s, ex(cap)), ex(cap))
    sets = list(caps)
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            sa, sb = sets[a], sets[b]
            if sa & sb and not (sa <= sb or sb <= sa):
                raise ValueError("rows are not laminar")
    parent = []
    for s in sets:
        sup = [r for r, t in enumerate(sets) if s < t]
        parent.append(min(sup, key=lambda r: len(sets[r])) if sup else -1)
    return sets, [caps[s] for s in sets], parent


def exact_priority_totals(inst):
    """Exact lexicographically minimal tier shortages for single or laminar capacities.

    The shifted set {0 <= y <= hi - lo, sum_{i in v} y_i <= c_v - sum_{i in v} lo_i} is a
    polymatroid. Serving tiers greedily in priority order attains the nested ranks
    r(T_1), r(T_1 u T_2) - r(T_1), ... Returns a list of Fractions, or None if the
    physical set is empty.
    """
    if not inst.physically_feasible():
        return None
    lo, hi = inst.exact_bounds()
    sets, caps, parent = laminar_tree(inst)
    cap = [caps[r] - sum((lo[i] for i in sets[r]), Fraction(0)) for r in range(len(sets))]
    children = {r: [] for r in range(len(sets))}
    roots = []
    for r, p in enumerate(parent):
        (children[p] if p >= 0 else roots).append(r)
    covered_by_child = {r: set().union(*[sets[ch] for ch in children[r]]) if children[r] else set()
                        for r in range(len(sets))}
    all_covered = set().union(*[sets[r] for r in roots]) if roots else set()
    free = [i for i in range(inst.n) if i not in all_covered]

    def rank(mask):
        def node(r):
            own = sets[r] - covered_by_child[r]
            total = sum((hi[i] - lo[i] for i in own if mask[i]), Fraction(0))
            total += sum((node(ch) for ch in children[r]), Fraction(0))
            return min(cap[r], total)
        total = sum((node(r) for r in roots), Fraction(0))
        return total + sum((hi[i] - lo[i] for i in free if mask[i]), Fraction(0))

    mask = [False] * inst.n
    prev_rank = Fraction(0)
    shortages = []
    for t in inst.tiers:
        for i in t:
            mask[i] = True
        r = rank(mask)
        served = r - prev_rank + sum((lo[i] for i in t), Fraction(0))
        demand = sum((ex(inst.d[i]) for i in t), Fraction(0))
        shortages.append(demand - served)
        prev_rank = r
    return shortages


def float_priority_totals(inst, tol=1e-10):
    """Sequential linear programs over the physical set (any nonnegative A).

    Returns float shortages, or None if the physical set is empty.
    """
    import highspy
    if not inst.physically_feasible():
        return None
    n = inst.n
    lo, hi = inst.lo, inst.hi
    fixed_rows = []   # (index array, lower bound on served total)
    shortages = []
    for t in inst.tiers:
        h = highspy.Highs()
        h.silent()
        inf = highspy.kHighsInf
        cost = np.zeros(n)
        cost[t] = -1.0           # maximize service of tier t
        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.col_cost_ = cost
        lp.col_lower_ = lo
        lp.col_upper_ = hi
        rows_lo, rows_hi, mat = [], [], []
        for r in range(len(inst.c)):
            rows_lo.append(-inf)
            rows_hi.append(inst.c[r])
            mat.append(inst.A[r])
        for idx, served in fixed_rows:
            row = np.zeros(n)
            row[idx] = 1.0
            rows_lo.append(served)
            rows_hi.append(inf)
            mat.append(row)
        M = np.array(mat)
        lp.num_row_ = M.shape[0]
        lp.row_lower_ = np.array(rows_lo)
        lp.row_upper_ = np.array(rows_hi)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        nz_start, nz_index, nz_value = [0], [], []
        for row in M:
            nz = np.flatnonzero(row)
            nz_index.extend(nz.tolist())
            nz_value.extend(row[nz].tolist())
            nz_start.append(len(nz_index))
        lp.a_matrix_.start_ = nz_start
        lp.a_matrix_.index_ = nz_index
        lp.a_matrix_.value_ = nz_value
        h.passModel(lp)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"priority LP status {h.getModelStatus()}")
        served = -h.getInfo().objective_function_value
        served_fixed = served - tol * (1 + abs(served))
        fixed_rows.append((t, served_fixed))
        shortages.append(float(inst.d[t].sum()) - served)
    return shortages


def priority_totals(inst):
    """Exact totals (Fractions) for single and laminar structures, floats otherwise.

    Returns (totals, exact_flag) or (None, None) when the physical set is empty.
    """
    if not inst.physically_feasible():
        return None, None
    if inst.structure in (SINGLE, LAMINAR):
        return exact_priority_totals(inst), True
    totals, stages = exact_lp_stages(inst)
    if totals is not None:
        inst.meta["priority_stages"] = stages
        return totals, True
    return float_priority_totals(inst), False


# ---------------------------------------------------------------------------
# Exact priority totals for a general nonnegative matrix A
# ---------------------------------------------------------------------------

def _exact_solve(M, rhs):
    """Solve the square rational system M z = rhs exactly; None if singular."""
    k = len(M)
    G = [list(M[r]) + [rhs[r]] for r in range(k)]
    for col in range(k):
        pr = next((i for i in range(col, k) if G[i][col] != 0), None)
        if pr is None:
            return None
        G[col], G[pr] = G[pr], G[col]
        pv = G[col][col]
        G[col] = [v / pv for v in G[col]]
        for i in range(k):
            if i != col and G[i][col] != 0:
                fct = G[i][col]
                G[i] = [a - fct * b for a, b in zip(G[i], G[col])]
    return [G[i][k] for i in range(k)]


def lp_dual_bound(cost, lo, hi, rows, row_lo, row_hi, y):
    """Exact weak-duality lower bound for min cost.x over {lo <= x <= hi, row_lo <= R x <= row_hi}.

    rows are dense lists of Fractions; None in row_lo/row_hi means infinite. Returns None
    if y uses an infinite row bound.
    """
    n = len(cost)
    red = list(cost)
    total = Fraction(0)
    for r, yr in enumerate(y):
        if yr == 0:
            continue
        bound = row_lo[r] if yr > 0 else row_hi[r]
        if bound is None:
            return None
        total += yr * bound
        for j, a in enumerate(rows[r]):
            if a != 0:
                red[j] -= yr * a
    for j in range(n):
        total += min(red[j] * lo[j], red[j] * hi[j])
    return total


def exact_lp_stages(inst):
    """Lexicographic tier service by sequential LPs, each solved exactly.

    HiGHS supplies an optimal basis; the basic solution and the row multipliers are then
    recomputed in rational arithmetic, and optimality is confirmed by equality of the
    primal value and the exact weak-duality bound. Returns (shortages, stages) where each
    stage holds the exact primal point and multipliers, or (None, None) on failure.
    """
    import highspy
    n = inst.n
    lo, hi = inst.exact_bounds()
    A = [[ex(a) for a in row] for row in inst.A]
    c = exact_vec(inst.c)
    rows, row_lo, row_hi = [list(r) for r in A], [None] * len(A), list(c)
    shortages, stages = [], []
    for t in inst.tiers:
        cost = [Fraction(0)] * n
        for i in t:
            cost[i] = Fraction(-1)
        h = highspy.Highs()
        h.silent()
        h.setOptionValue("primal_feasibility_tolerance", 1e-10)
        h.setOptionValue("dual_feasibility_tolerance", 1e-10)
        inf = highspy.kHighsInf
        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = len(rows)
        lp.col_cost_ = np.array([float(v) for v in cost])
        lp.col_lower_ = np.array([float(v) for v in lo])
        lp.col_upper_ = np.array([float(v) for v in hi])
        lp.row_lower_ = np.array([-inf if v is None else float(v) for v in row_lo])
        lp.row_upper_ = np.array([inf if v is None else float(v) for v in row_hi])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        start, index, value = [0], [], []
        for row in rows:
            for j, a in enumerate(row):
                if a != 0:
                    index.append(j)
                    value.append(float(a))
            start.append(len(index))
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = start, index, value
        h.passModel(lp)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None, None
        basis = h.getBasis()
        BS = highspy.HighsBasisStatus
        col_st = list(basis.col_status)
        row_st = list(basis.row_status)
        x = [None] * n
        for j in range(n):
            if col_st[j] == BS.kLower:
                x[j] = lo[j]
            elif col_st[j] == BS.kUpper:
                x[j] = hi[j]
            elif col_st[j] != BS.kBasic:
                x[j] = lo[j]
        basic = [j for j in range(n) if col_st[j] == BS.kBasic]
        tight = []
        for r in range(len(rows)):
            if row_st[r] == BS.kLower:
                tight.append((r, row_lo[r]))
            elif row_st[r] == BS.kUpper:
                tight.append((r, row_hi[r]))
        if len(tight) != len(basic) or any(b is None for _, b in tight):
            return None, None
        M = [[rows[r][j] for j in basic] for r, _ in tight]
        rhs = [b - sum((rows[r][j] * x[j] for j in range(n) if x[j] is not None and rows[r][j] != 0), Fraction(0))
               for r, b in tight]
        sol = _exact_solve(M, rhs) if basic else []
        if sol is None:
            return None, None
        for j, v in zip(basic, sol):
            x[j] = v
        # Multipliers: y on tight rows with B^T y = cost_B.
        MT = [[rows[r][j] for r, _ in tight] for j in basic]
        yt = _exact_solve(MT, [cost[j] for j in basic]) if basic else []
        if yt is None:
            return None, None
        y = [Fraction(0)] * len(rows)
        for (r, _), v in zip(tight, yt):
            y[r] = v
        feasible = all(lo[j] <= x[j] <= hi[j] for j in range(n))
        for r in range(len(rows)):
            val = sum((rows[r][j] * x[j] for j in range(n) if rows[r][j] != 0), Fraction(0))
            feasible &= (row_lo[r] is None or val >= row_lo[r]) and (row_hi[r] is None or val <= row_hi[r])
        primal = sum((cost[j] * x[j] for j in range(n)), Fraction(0))
        bound = lp_dual_bound(cost, lo, hi, rows, row_lo, row_hi, y)
        if not feasible or bound is None or bound != primal:
            return None, None
        served = -primal
        stages.append({"x": x, "y": y})
        shortages.append(sum((ex(inst.d[i]) for i in t), Fraction(0)) - served)
        row = [Fraction(0)] * n
        for i in t:
            row[i] = Fraction(1)
        rows.append(row)
        row_lo.append(served)
        row_hi.append(None)
    return shortages, stages
