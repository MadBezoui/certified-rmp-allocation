"""Tests of the comparators from the literature (comparators/methods.py) on small instances."""
import os
import pathlib
import sys
from fractions import Fraction

import cvxpy as cp
import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(os.environ.get("RMP_CODE_ROOT", HERE / "frozen" / "v1"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from certrmp.generator import make_instance  # noqa: E402
from certrmp.instance import priority_totals  # noqa: E402
from certrmp.model import RMPModel  # noqa: E402
from certrmp.objectives import standard_objectives, values  # noqa: E402
from certrmp.oracle import Oracle  # noqa: E402
from certrmp.pipeline import solve  # noqa: E402
from comparators import methods  # noqa: E402


def setup(structure, seed=3, n=30):
    inst = make_instance(n, structure, seed)
    objs = standard_objectives(inst)
    sigma, _ = priority_totals(inst)
    fmin, fnad, _, _ = methods.payoff_table(Oracle(inst, objs, sigma))
    profiles = [fmin + 0.35 * (fnad - fmin), fmin + 0.7 * (fnad - fmin)]
    model = RMPModel.srmp(profiles, [1, 0], [0.5, 0.3, 0.2])
    best = solve(inst, objs, model, method="lcs", do_verify=False)
    return inst, objs, sigma, model, tuple(Fraction(v) for v in best["key"])


@pytest.mark.parametrize("structure", ["single", "laminar", "general"])
def test_objective_matrix_matches_values(structure):
    inst = make_instance(30, structure, 5)
    objs = standard_objectives(inst)
    X = np.random.default_rng(0).uniform(inst.lo, inst.hi, size=(7, inst.n))
    F = methods.objective_matrix(objs, inst, X)
    for x, f in zip(X, F):
        assert np.allclose(f, values(objs, inst, x), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("structure", ["single", "laminar", "general"])
def test_projection_is_the_euclidean_projection(structure):
    inst = make_instance(30, structure, 11)
    sigma, _ = priority_totals(inst)
    proj = methods.Projection(inst, sigma)
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = rng.uniform(-0.5, 1.5, inst.n) * inst.d
        x = cp.Variable(inst.n)
        if proj.single:     # the frozen repair keeps every tier total at its priority total
            cons = [x >= inst.lo, x <= inst.hi] + [cp.sum(x[t]) == tg for t, tg in zip(inst.tiers, proj.targets)]
        else:
            cons = methods.x_constraints(inst, sigma, x)
        cp.Problem(cp.Minimize(cp.sum_squares(x - v)), cons).solve(solver="CLARABEL")
        y = proj(v)
        assert np.max(np.abs(y - x.value)) < 1e-5
        phys, prio = methods.residuals(inst, sigma, y)
        assert phys < 1e-7 and prio < 1e-6


@pytest.mark.parametrize("structure", ["single", "laminar", "general"])
def test_heuristics_never_beat_the_certified_key(structure):
    inst, objs, sigma, model, best = setup(structure)
    runs = [methods.asf_reference(inst, objs, model, sigma),
            methods.goal_programming(inst, objs, model, sigma),
            methods.eps_timed(inst, objs, model, sigma, time_limit=3.0),
            methods.emo(inst, objs, model, sigma, "nsga", time_limit=2.0),
            methods.emo(inst, objs, model, sigma, "rmpga", time_limit=2.0),
            methods.emo(inst, objs, model, sigma, "rnsga", time_limit=2.0)]
    for r in runs:
        assert r["status"] in ("HEURISTIC", "SAMPLE")
        assert r["key"] <= best
        assert model.key(values(objs, inst, r["x"])) == r["key"]
        phys, prio = methods.residuals(inst, sigma, r["x"])
        assert phys < 1e-7 and prio < 1e-6
        times = [t for t, _ in r["trajectory"]]
        keys = [k for _, k in r["trajectory"]]
        assert times == sorted(times) and all(a < b for a, b in zip(keys, keys[1:]))


@pytest.mark.parametrize("structure", ["single", "general"])
def test_indicator_formulation_claims_the_certified_key(structure):
    inst, objs, sigma, model, best = setup(structure)
    r = methods.miqcp_scip_indicator(inst, objs, model, sigma, time_limit=60.0)
    assert r["status"] == "OPTIMAL"
    assert r["claimed_key"] == best
