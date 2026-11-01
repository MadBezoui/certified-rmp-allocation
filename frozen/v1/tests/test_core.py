import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import itertools
import numpy as np
import pytest
from rmp.core import (Instance, Objectives, RMPModel, solve, priority_constraints,
                      OPTIMAL_NUMERICAL, PHYSICAL_INFEASIBLE)


def two_recipients():
    inst = Instance(d=np.array([4.0, 2.0]), tiers=[[0, 1]], A=np.ones((1, 2)), c=np.array([3.0]),
                    x_prev=np.array([1.5, 1.5]))
    return inst, Objectives(w=np.ones(2), v=np.ones(2), gamma=np.array([4.0]))


def face_grid(inst, steps=121):
    """Points of the single-tier face {sum x = c, 0 <= x <= d} for n = 3."""
    c = inst.c[0]
    pts = []
    for a in np.linspace(0, inst.d[0], steps):
        for b in np.linspace(0, inst.d[1], steps):
            z = c - a - b
            if -1e-12 <= z <= inst.d[2] + 1e-12:
                pts.append(np.array([a, b, max(z, 0.0)]))
    return pts


def random_three(rng):
    d = rng.uniform(0.5, 3.0, 3)
    inst = Instance(d=d, tiers=[[0, 1, 2]], A=np.ones((1, 3)), c=np.array([0.6 * d.sum()]),
                    x_prev=rng.uniform(0, 1, 3) * d)
    obj = Objectives(w=np.ones(3), v=np.ones(3), gamma=np.array([1.0]))
    return inst, obj


def random_profiles(rng, inst, obj, m):
    vals = np.array([obj.values(inst, x) for x in face_grid(inst, 25)])
    lo, hi = vals.min(0), vals.max(0)
    return lo + rng.uniform(0.1, 0.9, (m, 3)) * (hi - lo)


def test_example_matches_plan():
    inst, obj = two_recipients()
    assert np.allclose(obj.values(inst, np.array([1.5, 1.5])), [6.5, 0, 0.140625])
    assert np.allclose(obj.values(inst, np.array([2.0, 1.0])), [5, 1, 0])
    assert np.allclose(obj.values(inst, np.array([2.5, 0.5])), [4.5, 2, 0.140625])
    model = RMPModel.srmp([[5.2, 1.2, 0.02]], [0], [0.4, 0.2, 0.4])
    res = solve(inst, obj, model)
    assert res.status == OPTIMAL_NUMERICAL and res.key == (1,)
    assert np.allclose(res.x, [2, 1], atol=1e-5)


def test_literal_cell_is_nonconvex():
    # Allocations (a, 3 - a) with f3 > 0.02 form two disjoint intervals.
    inst, obj = two_recipients()
    a = np.linspace(1, 3, 2001)
    above = np.array([obj.values(inst, np.array([t, 3 - t]))[2] > 0.02 for t in a])
    switches = np.count_nonzero(np.diff(above.astype(int)))
    assert above[0] and above[-1] and not above[1000] and switches == 2


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    inst, obj = random_three(rng)
    m = 1 + seed % 3
    model = RMPModel.srmp(random_profiles(rng, inst, obj, m), rng.permutation(m), rng.uniform(0, 1, 3).round(2))
    res = solve(inst, obj, model)
    grid_best = max(model.key(obj.values(inst, x)) for x in face_grid(inst))
    assert grid_best <= res.key
    assert model.key(res.f) == res.key or res.boundary_flags


@pytest.mark.parametrize("seed", range(6))
def test_refinement_is_pareto_efficient(seed):
    rng = np.random.default_rng(100 + seed)
    inst, obj = random_three(rng)
    model = RMPModel.srmp(random_profiles(rng, inst, obj, 2), [0, 1], [0.5, 0.3, 0.2])
    res = solve(inst, obj, model)
    for x in face_grid(inst, 81):
        f = obj.values(inst, x)
        assert not (np.all(f <= res.f - 1e-6) and np.any(f < res.f - 1e-5))


def test_nonadditive_model():
    # {f2} > {f1} needs w2 > w1, while {f1,f3} > {f2,f3} needs w1 > w2: no additive weights fit.
    rank = {(): 0, (0,): 1, (1,): 2, (2,): 1, (0, 1): 3, (1, 2): 4, (0, 2): 5, (0, 1, 2): 6}
    table = {frozenset(k): v for k, v in rank.items()}
    inst, obj = two_recipients()
    model = RMPModel.from_table([[4.6, 0.5, 0.05]], [0], table, 3)
    assert model.check_monotone()
    res = solve(inst, obj, model)
    a = np.linspace(1, 3, 4001)
    best = max(model.key(obj.values(inst, np.array([t, 3 - t]))) for t in a)
    assert res.key == best


def test_nonmonotone_model_is_detected():
    table = {frozenset(s): len(s) for r in range(4) for s in itertools.combinations(range(3), r)}
    table[frozenset({0, 1, 2})] = 0
    assert not RMPModel.from_table([[1, 1, 1]], [0], table, 3).check_monotone()


def test_duplicate_and_unattainable_profiles():
    inst, obj = two_recipients()
    base = RMPModel.srmp([[5.2, 1.2, 0.02]], [0], [0.4, 0.2, 0.4])
    dup = RMPModel.srmp([[5.2, 1.2, 0.02], [5.2, 1.2, 0.02]], [0, 1], [0.4, 0.2, 0.4])
    impossible = RMPModel.srmp([[-1, -1, -1], [5.2, 1.2, 0.02]], [0, 1], [0.4, 0.2, 0.4])
    r0, r1, r2 = solve(inst, obj, base), solve(inst, obj, dup), solve(inst, obj, impossible)
    assert r1.key == r0.key * 2 and np.allclose(r0.x, r1.x, atol=1e-5)
    assert r2.key == (0,) + r0.key and np.allclose(r0.x, r2.x, atol=1e-5)


def test_zero_weight_criterion():
    inst, obj = two_recipients()
    model = RMPModel.srmp([[4.6, 0.5, 0.05]], [0], [0.5, 0.5, 0.0])
    res = solve(inst, obj, model)
    a = np.linspace(1, 3, 4001)
    assert res.key == max(model.key(obj.values(inst, np.array([t, 3 - t]))) for t in a)


def test_unit_change_preserves_choice():
    inst, obj = two_recipients()
    prof = np.array([[5.2, 1.2, 0.02]])
    r0 = solve(inst, obj, RMPModel.srmp(prof, [0], [0.4, 0.2, 0.4]))
    s = np.array([10.0, 0.5, 100.0])
    obj2 = Objectives(w=np.ones(2), v=np.ones(2), gamma=np.array([4.0]), scale=tuple(1 / s))
    r1 = solve(inst, obj2, RMPModel.srmp(prof * s, [0], [0.4, 0.2, 0.4]), tie_weights=tuple(1 / s))
    assert r0.key == r1.key and np.allclose(r0.x, r1.x, atol=1e-4)


def test_physical_infeasibility_is_reported():
    inst = Instance(d=np.array([1.0, 1.0]), tiers=[[0], [1]], A=np.ones((1, 2)), c=np.array([1.0]),
                    x_prev=np.array([1.0, 1.0]), r_down=np.array([0.1, 0.1]), r_up=np.array([0.1, 0.1]))
    obj = Objectives(w=np.ones(2), v=np.ones(2), gamma=np.zeros(2))
    assert solve(inst, obj, RMPModel.srmp([[1, 1, 1]], [0], [1, 1, 1])).status == PHYSICAL_INFEASIBLE


def test_priorities_use_reachable_set():
    # Unconstrained priority optimum (1, 0) is unreachable from (0.2, 0.8) with ramp 0.1,
    # yet a reachable priority-optimal allocation (0.3, 0.7) exists.
    inst = Instance(d=np.array([1.0, 1.0]), tiers=[[0], [1]], A=np.ones((1, 2)), c=np.array([1.0]),
                    x_prev=np.array([0.2, 0.8]), r_down=np.array([0.1, 0.1]), r_up=np.array([0.1, 0.1]))
    build, short = priority_constraints(inst)
    assert build is not None and abs(short[0] - 0.7) < 1e-7
