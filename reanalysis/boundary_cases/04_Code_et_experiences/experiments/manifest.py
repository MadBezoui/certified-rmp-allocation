"""Write a frozen benchmark manifest: instance specifications and preference models.

Usage: python manifest.py <design> <output.json>
Designs: pilot, main, stress_profiles, stress_dimension, sensitivity, preference.
The ablation runs reuse the main manifest with a filter (see the protocol).
"""
import hashlib
import json
import sys
import pathlib
import warnings

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from certrmp.generator import make_instance, SCARCITY
from certrmp.model import RMPModel, random_monotone_table
from certrmp.instance import priority_totals
from certrmp.objectives import standard_objectives, regional_objectives
from certrmp.oracle import Oracle
from certrmp import profiles
from fractions import Fraction

STRUCTURES = ("single", "laminar", "general")


def instance_hash(inst):
    h = hashlib.sha256()
    for arr in (inst.d, inst.x_prev, inst.A, inst.c):
        h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
    for t in inst.tiers:
        h.update(np.asarray(t, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def build_instance(e):
    return make_instance(e["n"], e["structure"], e["seed"], scarcity=e.get("scarcity", SCARCITY),
                         ramp=e.get("ramp"))


def objectives_for(inst, p):
    if p == 3:
        return standard_objectives(inst)
    if p == 2:
        return standard_objectives(inst)[:2]
    regions = np.array_split(np.arange(inst.n), p - 2)
    return regional_objectives(inst, regions)


def entries(design):
    if design == "pilot":
        for n in (10, 30, 100):
            for p in (2, 3):
                for m in (1, 3, 5):
                    for st in STRUCTURES:
                        for seed in range(3):
                            yield dict(n=n, p=p, m=m, structure=st, seed=seed, models=[("uniform", "srmp")])
    elif design == "main":
        for n in (30, 100, 300, 1000):
            for m in (2, 4, 8):
                for st in STRUCTURES:
                    for seed in range(20):
                        yield dict(n=n, p=3, m=m, structure=st, seed=1000 + seed,
                                   models=[("uniform", "srmp"), ("near_front", "srmp")])
    elif design == "stress_profiles":
        for st in STRUCTURES:
            for seed in range(20):
                yield dict(n=100, p=3, m=4, structure=st, seed=5000 + seed,
                           models=[("easy", "srmp"), ("unattainable", "srmp"), ("adjacent", "srmp"),
                                   ("near_front", "table")])
    elif design == "stress_dimension":
        for p in (2, 3, 5, 8):
            for seed in range(10):
                yield dict(n=100, p=p, m=4, structure="laminar", seed=7000 + seed,
                           models=[("uniform", "srmp"), ("near_front", "srmp")])
    elif design == "sensitivity":
        # resource scarcity x ramp tightness, plus a looser solver tolerance on the base cell
        for scarcity in (0.5, 0.75, 0.9):
            for ramp in (None, 0.3, 0.1):
                for seed in range(10):
                    yield dict(n=100, p=3, m=4, structure="laminar", seed=8000 + seed, scarcity=scarcity,
                               ramp=ramp, models=[("near_front", "srmp")])
        for seed in range(10):
            yield dict(n=100, p=3, m=4, structure="laminar", seed=8000 + seed, scarcity=0.75, ramp=None,
                       solver_tol=1e-8, models=[("near_front", "srmp")])
    elif design == "preference":
        # one physical instance, several preference models derived from one base model
        for st in STRUCTURES:
            for seed in range(10):
                yield dict(n=100, p=3, m=4, structure=st, seed=9000 + seed, models=[("near_front", "srmp")],
                           variants=("base", "perturbed", "reversed", "concentrated", "equal", "table"))
    else:
        raise ValueError(design)


def variant(model, name, rng, span):
    """Preference-model variants for the sensitivity study (same profiles unless stated).
    span is the payoff range of each criterion used when the profiles were generated."""
    d = model.to_dict()
    P = np.array(d["profiles"])
    if name == "base":
        return d
    if name == "perturbed":
        Q = P + rng.uniform(-0.02, 0.02, P.shape) * np.asarray(span)
        return dict(d, profiles=[[profiles._sig(v) for v in row] for row in Q])
    if name == "reversed":
        return dict(d, order=list(reversed(d["order"])))
    if name == "concentrated":
        w = ["1/10"] * model.p
        w[int(rng.integers(model.p))] = str(1 - (model.p - 1) * Fraction(1, 10))
        return dict(d, weights=w)
    if name == "equal":
        return dict(d, weights=[f"1/{model.p}"] * model.p)
    if name == "table":
        tab = random_monotone_table(model.p, rng)
        return RMPModel.from_table(d["profiles"], d["order"], tab, model.p).to_dict()
    raise ValueError(name)


def main(design, out):
    rows = []
    for k, e in enumerate(entries(design)):
        inst = build_instance(e)
        objs = objectives_for(inst, e["p"])
        sigma, _ = priority_totals(inst)
        row = dict(e, id=f"{design}-{k:05d}", instance_sha=instance_hash(inst), models=[])
        if sigma is None:
            row["physically_infeasible"] = True
            rows.append(row)
            continue
        oracle = Oracle(inst, objs, sigma)
        for j, (family, kind) in enumerate(e["models"]):
            model, info = profiles.generate(oracle, e["m"], seed=e["seed"] * 10 + j, family=family, kind=kind)
            if "variants" in e:
                rng = np.random.default_rng(e["seed"] + 77)
                span = np.array(info["fnad"]) - np.array(info["fmin"])
                span = np.where(span > 1e-6 * (1 + np.abs(info["fmin"])), span,
                                profiles.SPAN_FLOOR * np.maximum(np.abs(info["fmin"]), 1e-6))
                for name in e["variants"]:
                    md = variant(model, name, rng, span)
                    row["models"].append(dict(model=md, variant=name, **dict(info, kind=md["kind"])))
            else:
                row["models"].append(dict(model=model.to_dict(), **info))
        row.pop("variants", None)
        rows.append(row)
        if k % 50 == 0:
            print(design, k, flush=True)
    pathlib.Path(out).write_text(json.dumps({"design": design, "entries": rows}))
    print("wrote", len(rows), "entries to", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
