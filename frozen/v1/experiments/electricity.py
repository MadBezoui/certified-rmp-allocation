"""Demand-trace study: repeated RMP decisions on UCI electricity load diagrams.

Real normalized demand traces (32 clients, 15-minute resolution, 2013) are combined with
constructed capacities, feeder limits, service tiers, ramp limits and preference
profiles. This is a data-driven synthetic decision experiment, not a deployment.

Usage: python electricity.py <out_dir> [ramp] [workers]
"""
import json
import os
import sys
import pathlib
import time
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np

from certrmp.instance import Instance, LAMINAR, priority_totals
from certrmp.objectives import Objective, values
from certrmp.model import RMPModel
from certrmp.oracle import Oracle
from certrmp.dynamic import RMPController, BaselineController, emergency_instance

DATA = pathlib.Path(os.environ.get("ABO_DATA", "/mnt/project-files/ABO/code_experiment_results/data/processed/electricity.npz"))
BLOCK, NBLOCKS, DEV = 336, 30, 6
RHO, SHOCK = 0.9, (112, 224)
FEEDER_CAP = 7.6
WEIGHTS = (0.5, 0.25, 0.25)
BETAS = (0.25, 0.5, 0.75)        # demanding, moderate, lenient
ORDER = (1, 0, 2)                # consult moderate, then demanding, then lenient
RULES = ("myopic", "feedback", "smoothing", "absolute", "robd")


def load():
    data = np.load(DATA)
    D = np.round(data["demand"], 6)
    crit = np.asarray(data["critical"])
    n = D.shape[1]
    tiers = [np.sort(crit), np.setdiff1d(np.arange(n), crit)]
    A = [np.ones(n)]
    for g in range(4):
        row = np.zeros(n)
        row[g * 8:(g + 1) * 8] = 1
        A.append(row)
    starts = np.linspace(0, len(D) - BLOCK, NBLOCKS, dtype=int)
    return D, tiers, np.array(A), starts, [str(s) for s in data["timestamps"][starts]]


def objectives_for(inst):
    groups = [np.array([i for i in t if inst.d[i] > 1e-3], dtype=int) for t in inst.tiers]
    n = inst.n
    return [Objective("shortfall", "shortfall", weights=np.ones(n), scale=float(n)),
            Objective("movement", "movement", weights=np.ones(n), scale=float(n)),
            Objective("disparity", "disparity", groups=groups, gammas=np.ones(len(groups)), scale=float(len(groups)))]


def epoch_instance(d, x_prev, t, tiers, A, ramp):
    cap = RHO * 32.0 * (0.5 if SHOCK[0] <= t < SHOCK[1] else 1.0)
    c = np.r_[cap, np.full(4, FEEDER_CAP)]
    r = None if ramp is None else np.full(len(d), ramp)
    return Instance(d=d, x_prev=np.round(x_prev, 9), tiers=tiers, A=A, c=np.round(c, 6), structure=LAMINAR,
                    r_down=r, r_up=r)


def development_profiles(D, tiers, A, starts, ramp):
    """Profiles from development blocks only: medians of fmin + beta (nadir - fmin) along
    the myopic trajectory."""
    samples = []
    for b in range(DEV):
        ctl = BaselineController("myopic")
        x = D[starts[b]].copy()
        for t in range(0, BLOCK, 4):
            inst = epoch_instance(D[starts[b] + t], x, t, tiers, A, ramp)
            sigma, _ = priority_totals(inst)
            if sigma is None:
                inst = emergency_instance(inst)
                sigma, _ = priority_totals(inst)
            o = Oracle(inst, objectives_for(inst), sigma)
            F = np.array([o.weighted(np.eye(3)[j])["f"] for j in range(3)])
            samples.append((F.diagonal(), F.max(axis=0)))
            x, _ = ctl.step(inst)
    fmin = np.array([s[0] for s in samples])
    fnad = np.array([s[1] for s in samples])
    profiles = [[float(f"{np.median(fmin[:, j] + beta * (fnad[:, j] - fmin[:, j])):.6g}") for j in range(3)]
                for beta in BETAS]
    return profiles


def run_block(args):
    b, method, profiles, ramp = args[:4]
    horizon = args[4] if len(args) > 4 else BLOCK
    warnings.filterwarnings("ignore")
    D, tiers, A, starts, _ = load()
    model = RMPModel.srmp(profiles, list(ORDER), list(WEIGHTS))
    if method.startswith("rmp"):
        ctl = RMPController(model, objectives_for, reuse=(method in ("rmp_reuse", "rmp_warm")), certify_each=True,
                            warm=(method == "rmp_warm"))
    else:
        ctl = BaselineController(method)
    x = D[starts[b]].copy()
    rows = []
    for t in range(horizon):
        d = D[starts[b] + t]
        inst = epoch_instance(d, x, t, tiers, A, ramp)
        x_new, info = ctl.step(inst)
        objs = objectives_for(inst)
        f = values(objs, inst, x_new)
        row = {"block": b, "t": t, "method": method, "f": f.tolist(), "key": [str(v) for v in model.key(f)],
               "emergency": bool(info.get("emergency")), "seconds": info.get("seconds"),
               "served_tier1": float(x_new[tiers[0]].sum()), "demand_tier1": float(d[tiers[0]].sum()),
               "served_total": float(x_new.sum()), "capacity": float(inst.c[0])}
        for k in ("calls", "carried", "separated_by_reused", "verified", "key_switch"):
            if k in info:
                row[k] = info[k]
        rows.append(row)
        x = x_new
    return rows


def main():
    out = pathlib.Path(sys.argv[1])
    ramp = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "none" else None
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    out.mkdir(parents=True, exist_ok=True)
    D, tiers, A, starts, stamps = load()
    tag = "none" if ramp is None else f"{ramp:g}"
    prof_path = out / f"profiles_ramp{tag}.json"
    if prof_path.exists():
        profiles = json.loads(prof_path.read_text())["profiles"]
    else:
        profiles = development_profiles(D, tiers, A, starts, ramp)
        prof_path.write_text(json.dumps({"profiles": profiles, "order": ORDER, "weights": WEIGHTS,
                                         "betas": BETAS, "dev_blocks": list(range(DEV)),
                                         "block_starts": stamps}, indent=2))
    print("profiles", profiles, flush=True)
    res_path = out / f"trajectories_ramp{tag}.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["block"], r["method"]))
    tasks = [(b, m, profiles, ramp) for b in range(DEV, NBLOCKS)
             for m in ("rmp_warm", "rmp_reuse", "rmp_cold") + RULES if (b, m) not in done]
    print(len(tasks), "block runs", flush=True)
    with Pool(workers, maxtasksperchild=4) as pool, res_path.open("a") as fh:
        for rows in pool.imap_unordered(run_block, tasks):
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print("done", rows[0]["block"], rows[0]["method"], flush=True)


if __name__ == "__main__":
    main()
