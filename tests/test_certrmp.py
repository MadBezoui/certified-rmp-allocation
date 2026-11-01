"""Correctness suite for the certified RMP solver (mapped to the theorem assumptions)."""
import copy
import json
import math
import sys
import pathlib
import warnings
from itertools import product

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from certrmp.instance import Instance, SINGLE, LAMINAR, GENERAL, priority_totals, exact_priority_totals
from certrmp.objectives import Objective, standard_objectives, values
from certrmp.model import RMPModel, random_monotone_table
from certrmp.oracle import Oracle
from certrmp.search import lcs, enumerate_grid, refine
from certrmp.generator import make_instance, payoff_range, make_model
from certrmp.pipeline import solve
from certrmp import certify, verify
from rmp import core as reference


def example():
    inst = Instance(d=[4, 2], x_prev=[1.5, 1.5], tiers=[[0, 1]], A=np.ones((1, 2)), c=[3], structure=SINGLE)
    objs = [Objective("shortfall", "shortfall", weights=np.ones(2), scale=1.0),
            Objective("movement", "movement", weights=np.ones(2), scale=1.0),
            Objective("disparity", "disparity", groups=[np.array([0, 1])], gammas=np.array([4.0]), scale=1.0)]
    return inst, objs


def certificate_for(inst, objs, model):
    sigma, exact = priority_totals(inst)
    o = Oracle(inst, objs, sigma)
    r = lcs(o, model)
    ref = refine(o, r.frontier)
    return json.loads(json.dumps(certify.build(inst, objs, model, sigma, r, ref, exact)))


# -- worked example ----------------------------------------------------------------

def test_example_values_and_choice():
    inst, objs = example()
    assert np.allclose(values(objs, inst, np.array([1.5, 1.5])), [6.5, 0, 0.140625])
    assert np.allclose(values(objs, inst, np.array([2.0, 1.0])), [5, 1, 0])
    assert np.allclose(values(objs, inst, np.array([2.5, 0.5])), [4.5, 2, 0.140625])
    model = RMPModel.srmp([[5.2, 1.2, 0.02]], [0], [0.4, 0.2, 0.4])
    rec = solve(inst, objs, model)
    assert rec["status"] == "OPTIMAL_VERIFIED" and rec["key"] == ["1"]
    o = Oracle(inst, objs, priority_totals(inst)[0])
    ref = refine(o, [(5.2, 1.2, 0.02)], weights=[1.0, 1.0, 1.0])
    assert np.allclose(ref["x"], [2, 1], atol=1e-5)


def test_example_best_class():
    inst, objs = example()
    model = RMPModel.srmp([[5.2, 1.2, 0.02], [6.0, 1.8, 0.1]], [0, 1], [0.4, 0.2, 0.4])
    rec = solve(inst, objs, model)
    assert rec["status"] == "OPTIMAL_VERIFIED" and rec["key"] == ["1", "1"]
    key = lambda a: [str(v) for v in model.key(values(objs, inst, np.array([a, 3 - a])))]
    low = (5 - math.sqrt(1.4)) / 2
    assert key(low + 1e-6) == ["1", "1"] and key(low - 1e-6) != ["1", "1"]
    assert key(2.1 - 1e-9) == ["1", "1"] and key(2.1 + 1e-6) != ["1", "1"]   # 2.1 itself is on the boundary


def test_best_class_can_contain_dominated_allocations():
    # Example without movement: a = 1.95 has the best key, a = 2 is better on both criteria.
    inst, objs = example()
    objs2 = [objs[0], objs[2]]
    model = RMPModel.srmp([[5.2, 0.02], [6.0, 0.1]], [0, 1], [0.4, 0.4])
    rec = solve(inst, objs2, model)
    assert rec["status"] == "OPTIMAL_VERIFIED"
    f195 = values(objs2, inst, np.array([1.95, 1.05]))
    f2 = values(objs2, inst, np.array([2.0, 1.0]))
    assert [str(v) for v in model.key(f195)] == rec["key"]
    assert np.all(f2 < f195)
    assert rec["x"][0] >= 2 - 1e-6


def test_literal_cell_is_nonconvex():
    inst, objs = example()
    a = np.linspace(1, 3, 2001)
    above = np.array([objs[2].value(inst, np.array([t, 3 - t])) > 0.02 for t in a])
    assert above[0] and above[-1] and not above[1000]
    assert np.count_nonzero(np.diff(above.astype(int))) == 2


# -- agreement with independent references ------------------------------------------

CASES = [(st, seed, m, kind, ramps) for st in (SINGLE, LAMINAR, GENERAL) for seed in range(3)
         for m, kind in ((1, "srmp"), (2, "table"), (3, "srmp")) for ramps in (False,)]


@pytest.mark.parametrize("structure,seed,m,kind,ramps", CASES)
def test_lcs_matches_enumeration(structure, seed, m, kind, ramps):
    inst = make_instance(20, structure, seed, ramps=ramps)
    objs = standard_objectives(inst)
    sigma, _ = priority_totals(inst)
    fmin, fnad = payoff_range(Oracle(inst, objs, sigma))
    model = make_model(fmin, fnad, m, 50 + seed, kind=kind)
    a = lcs(Oracle(inst, objs, sigma), model)
    b = enumerate_grid(Oracle(inst, objs, sigma), model)
    assert a.status == b.status == "OPTIMAL" and a.key == b.key


@pytest.mark.parametrize("seed", range(6))
def test_tied_weights_match_enumeration(seed):
    inst = make_instance(15, LAMINAR, 100 + seed)
    objs = standard_objectives(inst)
    sigma, _ = priority_totals(inst)
    fmin, fnad = payoff_range(Oracle(inst, objs, sigma))
    base = make_model(fmin, fnad, 3, 200 + seed)
    model = RMPModel.srmp(base.profiles, base.order, [0.4, 0.2, 0.4])
    assert lcs(Oracle(inst, objs, sigma), model).key == enumerate_grid(Oracle(inst, objs, sigma), model).key


@pytest.mark.parametrize("seed", range(6))
def test_matches_separate_reference_code(seed):
    """rmp/core.py shares no control logic with certrmp."""
    inst = make_instance(8, SINGLE, 300 + seed)
    objs = standard_objectives(inst)
    sigma, _ = priority_totals(inst)
    fmin, fnad = payoff_range(Oracle(inst, objs, sigma))
    model = make_model(fmin, fnad, 2, 400 + seed)
    a = lcs(Oracle(inst, objs, sigma), model)
    rinst = reference.Instance(d=inst.d, tiers=[t.tolist() for t in inst.tiers], A=inst.A, c=inst.c,
                               x_prev=inst.x_prev)
    sc = tuple(o.scale for o in objs)
    robj = reference.Objectives(w=np.ones(inst.n), v=np.ones(inst.n), gamma=np.ones(len(inst.tiers)), scale=sc)
    from fractions import Fraction
    rmodel = reference.RMPModel.srmp(model.profiles, model.order, [float(Fraction(w)) for w in model.weights])
    r = reference.solve(rinst, robj, rmodel)
    assert r.key == a.key


def test_bruteforce_on_a_face():
    rng = np.random.default_rng(7)
    for trial in range(5):
        d = np.round(rng.uniform(0.5, 3.0, 3), 3)
        inst = Instance(d=d, x_prev=np.round(d * rng.uniform(0.2, 1, 3), 3), tiers=[[0, 1, 2]],
                        A=np.ones((1, 3)), c=[round(0.6 * d.sum(), 3)], structure=SINGLE)
        objs = standard_objectives(inst)
        sigma, _ = priority_totals(inst)
        fmin, fnad = payoff_range(Oracle(inst, objs, sigma))
        model = make_model(fmin, fnad, 2, 500 + trial)
        rec = solve(inst, objs, model)
        best = None
        for a, b in product(np.linspace(0, d[0], 81), np.linspace(0, d[1], 81)):
            z = inst.c[0] - a - b
            if -1e-12 <= z <= d[2]:
                k = model.key(values(objs, inst, np.array([a, b, max(z, 0)])))
                best = k if best is None or k > best else best
        assert rec["status"] == "OPTIMAL_VERIFIED"
        assert best <= tuple(model.importance(frozenset()) * 0 + type(best[0])(v) for v in rec["key"])


# -- certificates ------------------------------------------------------------------

@pytest.fixture(scope="module")
def cert_case():
    inst = make_instance(25, LAMINAR, 11)
    objs = standard_objectives(inst)
    sigma, _ = priority_totals(inst)
    fmin, fnad = payoff_range(Oracle(inst, objs, sigma))
    model = make_model(fmin, fnad, 3, 12)
    return certificate_for(inst, objs, model)


def test_certificate_verifies(cert_case):
    out = verify.verify(cert_case)
    assert out["verified"] and out["exact"]


def test_rejects_better_claimed_key(cert_case):
    bad = copy.deepcopy(cert_case)
    from fractions import Fraction
    bad["key"] = [str(Fraction(bad["key"][0]) + Fraction(1, 100))] + bad["key"][1:]
    assert not verify.verify(bad)["verified"]


def test_rejects_inflated_multipliers(cert_case):
    bad = copy.deepcopy(cert_case)
    for hs in bad["halfspaces"]:
        hs["tangent_x"] = ["0"] * len(hs["tangent_x"])     # still valid bounds, but much weaker
        hs["y_cap"] = ["1000"] * len(hs["y_cap"])
    out = verify.verify(bad)
    # Weak but valid bounds cannot certify the exclusions: the verifier must refuse.
    assert not out["checks"]["levels"] or len(cert_case["halfspaces"]) == 0


def test_rejects_moved_witness(cert_case):
    bad = copy.deepcopy(cert_case)
    bad["witness"]["x"] = ["0"] * len(bad["witness"]["x"])
    assert not verify.verify(bad)["verified"]


def test_rejects_changed_priority_totals(cert_case):
    bad = copy.deepcopy(cert_case)
    from fractions import Fraction
    bad["priority"]["totals"] = [str(Fraction(v) + Fraction(1, 2)) for v in bad["priority"]["totals"]]
    assert not verify.verify(bad)["exact"]


def test_rejects_missing_frontier(cert_case):
    bad = copy.deepcopy(cert_case)
    for lvl in bad["levels"]:
        lvl["next_frontier"] = []
    bad["halfspaces"] = []
    assert not verify.verify(bad)["verified"]


# -- statuses and semantics ---------------------------------------------------------

def test_physical_infeasibility_is_reported_and_verified():
    inst = Instance(d=[1, 1], x_prev=[1, 1], tiers=[[0], [1]], A=np.ones((1, 2)), c=[1], structure=SINGLE,
                    r_down=np.array([0.1, 0.1]), r_up=np.array([0.1, 0.1]))
    objs = standard_objectives(inst)
    rec = solve(inst, objs, RMPModel.srmp([[1, 1, 1]], [0], [1, 1, 1]))
    assert rec["status"] == "PHYSICAL_INFEASIBLE" and rec["verified"]


def test_priorities_use_reachable_set():
    inst = Instance(d=[1, 1], x_prev=[0.2, 0.8], tiers=[[0], [1]], A=np.ones((1, 2)), c=[1], structure=SINGLE,
                    r_down=np.array([0.1, 0.1]), r_up=np.array([0.1, 0.1]))
    s = exact_priority_totals(inst)
    assert s is not None and float(s[0]) == pytest.approx(0.7) and float(s[1]) == pytest.approx(0.3)


def test_refinement_is_not_dominated():
    inst = make_instance(6, SINGLE, 21)
    objs = standard_objectives(inst)
    sigma, _ = priority_totals(inst)
    o = Oracle(inst, objs, sigma)
    fmin, fnad = payoff_range(o)
    model = make_model(fmin, fnad, 2, 22)
    rec = solve(inst, objs, model)
    fstar = np.array(rec["f"])
    rng = np.random.default_rng(0)
    for _ in range(300):
        lam = rng.dirichlet(np.ones(3))
        r = o.weighted(lam)
        f = r["f"]
        assert not (np.all(f <= fstar - 1e-7) and np.any(f < fstar - 1e-6))


def test_nonmonotone_model_detected():
    table = {(): 0, (0,): 1, (1,): 1, (2,): 1, (0, 1): 2, (0, 2): 2, (1, 2): 2, (0, 1, 2): 0}
    with pytest.raises(ValueError, match='monotone'):
        RMPModel.from_table([[1, 1, 1]], [0], table, 3)
    assert RMPModel.from_table([[1, 1, 1]], [0], random_monotone_table(3, np.random.default_rng(1)), 3).check_monotone()


def test_recipient_renaming_preserves_result():
    inst = make_instance(12, LAMINAR, 31)
    objs = standard_objectives(inst)
    sigma, _ = priority_totals(inst)
    fmin, fnad = payoff_range(Oracle(inst, objs, sigma))
    model = make_model(fmin, fnad, 3, 32)
    perm = np.random.default_rng(3).permutation(inst.n)
    inv = np.argsort(perm)
    inst2 = Instance(d=inst.d[perm], x_prev=inst.x_prev[perm], tiers=[np.sort(inv[t]) for t in inst.tiers],
                     A=inst.A[:, perm], c=inst.c, structure=LAMINAR)
    a, b = solve(inst, objs, model), solve(inst2, standard_objectives(inst2), model)
    assert a["key"] == b["key"] and np.allclose(a["f"], b["f"], atol=1e-6)


def test_duplicate_profile_duplicates_key():
    inst, objs = example()
    m1 = RMPModel.srmp([[5.2, 1.2, 0.02]], [0], [0.4, 0.2, 0.4])
    m2 = RMPModel.srmp([[5.2, 1.2, 0.02], [5.2, 1.2, 0.02]], [0, 1], [0.4, 0.2, 0.4])
    a, b = solve(inst, objs, m1), solve(inst, objs, m2)
    assert b["key"] == a["key"] * 2 and np.allclose(a["f"], b["f"], atol=1e-6)


def test_unit_change_preserves_result():
    inst, objs = example()
    scale = np.array([10.0, 0.5, 100.0])
    objs2 = [Objective(o.name, o.kind, o.weights, o.groups, o.gammas, scale=o.scale / s) for o, s in zip(objs, scale)]
    prof = np.array([[5.2, 1.2, 0.02]])
    a = solve(inst, objs, RMPModel.srmp(prof, [0], [0.4, 0.2, 0.4]))
    b = solve(inst, objs2, RMPModel.srmp(prof * scale, [0], [0.4, 0.2, 0.4]))
    assert a["key"] == b["key"] and np.allclose(np.array(a["f"]) * scale, b["f"], rtol=1e-5)


def test_no_false_verification_at_boundaries():
    """Profiles placed exactly at computed individual minima: the solver may resolve or
    flag them, but a verified status must agree with enumeration."""
    for seed in range(4):
        inst = make_instance(10, SINGLE, 600 + seed)
        objs = standard_objectives(inst)
        sigma, _ = priority_totals(inst)
        fmin, fnad = payoff_range(Oracle(inst, objs, sigma))
        model = RMPModel.srmp([fmin.tolist(), ((fmin + fnad) / 2).tolist()], [0, 1], [0.3, 0.3, 0.4])
        rec = solve(inst, objs, model)
        e = enumerate_grid(Oracle(inst, objs, sigma), model)
        if rec["status"] == "OPTIMAL_VERIFIED":
            assert tuple(rec["key"]) == tuple(str(v) for v in e.key)
