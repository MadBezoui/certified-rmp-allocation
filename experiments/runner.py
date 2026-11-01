"""Run benchmark tasks from a frozen manifest; restartable and append-only.

Usage: python runner.py <manifest.json> <results.jsonl> <methods,comma,separated> [workers] [time_limit]
Each output line is one (entry, model, method) record. Existing records are skipped on
restart, and every line carries the hashes of the code and of the manifest.
"""
import hashlib
import json
import os
import platform
import sys
import pathlib
import time
import traceback
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
warnings.filterwarnings("ignore")

import numpy as np

from certrmp.generator import make_instance
from certrmp.instance import priority_totals
from certrmp.model import model_from_dict
from certrmp.oracle import Oracle
from certrmp.pipeline import solve
from certrmp import baselines
from manifest import objectives_for, build_instance
import certrmp.oracle as oracle_mod

DEFAULT_TOL = oracle_mod.SOLVER_OPTS["tol_feas"]


def code_hash():
    h = hashlib.sha256()
    for f in sorted((ROOT / "certrmp").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def keystr(k):
    return None if k is None else [str(v) for v in k]


def residuals(inst, sigma, x):
    """Largest violation of the physical constraints and of the priority totals (float)."""
    x = np.asarray(x, dtype=float)
    phys = max(float(np.max(inst.A @ x - inst.c)), float(np.max(x - inst.hi)), float(np.max(inst.lo - x)), 0.0)
    prio = max([float(inst.d[t].sum() - x[t].sum() - float(s)) for t, s in zip(inst.tiers, sigma)] + [0.0])
    return phys, prio


def run_task(args):
    entry, mi, method, tl = args
    warnings.filterwarnings("ignore")
    rec = {"id": entry["id"], "model": mi, "method": method, "n": entry["n"], "p": entry["p"], "m": entry["m"],
           "structure": entry["structure"], "seed": entry["seed"]}
    for k in ("scarcity", "ramp", "solver_tol"):
        if k in entry:
            rec[k] = entry[k]
    try:
        oracle_mod.SOLVER_OPTS.update(tol_gap_abs=entry.get("solver_tol", DEFAULT_TOL),
                                      tol_gap_rel=entry.get("solver_tol", DEFAULT_TOL),
                                      tol_feas=entry.get("solver_tol", DEFAULT_TOL))
        inst = build_instance(entry)
        objs = objectives_for(inst, entry["p"])
        if entry.get("physically_infeasible"):
            rec["status"] = "PHYSICAL_INFEASIBLE"
            return rec
        md = entry["models"][mi]
        model = model_from_dict(md["model"])
        rec["family"], rec["kind"] = md["family"], md["kind"]
        if "variant" in md:
            rec["variant"] = md["variant"]
        t0 = time.perf_counter()
        if method in ("lcs", "lcs_nopoints", "lcs_nohalf", "lcs_nocache", "enum"):
            base = "enum" if method == "enum" else "lcs"
            r = solve(inst, objs, model, method=base, time_limit=tl,
                      use_points=method not in ("lcs_nopoints", "lcs_nocache"),
                      use_halfspaces=method not in ("lcs_nohalf", "lcs_nocache"),
                      do_verify=(method == "lcs"))
            r.pop("x", None)
            rec.update(r)
        elif method in ("scip", "gurobi"):
            sigma, _ = priority_totals(inst)
            fn = baselines.miqcp_scip if method == "scip" else baselines.miqcp_gurobi
            r = fn(inst, objs, model, sigma, time_limit=tl)
            rec.update(status=r["status"], claimed_key=keystr(r["claimed_key"]), key=keystr(r.get("key")),
                       f=None if r.get("f") is None else list(map(float, r["f"])), seconds=r["seconds"])
            if r.get("x") is not None:
                rec["residual_physical"], rec["residual_priority"] = residuals(inst, sigma, r["x"])
        elif method == "eps":
            sigma, _ = priority_totals(inst)
            r = baselines.eps_constraint(Oracle(inst, objs, sigma), model)
            rec.update(status=r["status"], key=keystr(r["key"]), f=list(map(float, r["f"])),
                       seconds=r["seconds"], points=r["points"])
        elif method in ("nsga", "rmp_ga"):
            sigma, _ = priority_totals(inst)
            r = baselines.nsga2(inst, objs, model, sigma, pop=100, gens=200, seed=entry["seed"],
                                guided=(method == "rmp_ga"))
            rec.update(status=r["status"], key=keystr(r["key"]), f=list(map(float, r["f"])),
                       seconds=r["seconds"], evaluations=r["evaluations"])
            rec["residual_physical"], rec["residual_priority"] = residuals(inst, sigma, r["x"])
        else:
            raise ValueError(method)
        rec["wall"] = time.perf_counter() - t0
    except Exception as exc:
        rec["status"] = "ERROR"
        rec["error"] = repr(exc)[:300]
        rec["trace"] = traceback.format_exc()[-800:]
    for k in ("key", "claimed_key"):
        if k in rec and rec[k] is not None and not isinstance(rec[k], list):
            rec[k] = keystr(rec[k])
    return rec


MEMORY_LIMIT = int(os.environ.get("RMP_MEMORY_LIMIT_GB", "6")) * 2 ** 30


def limit_memory():
    """Address-space limit per worker; a task exceeding it is recorded as an ERROR."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))


def applicable(entry, method):
    if method == "gurobi":
        return entry["n"] <= 30
    if method in ("nsga", "rmp_ga"):
        return entry["structure"] == "single"
    return True


def main():
    manifest_path, out_path, methods = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    tl = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
    only = sys.argv[6] if len(sys.argv) > 6 else None          # optional filter "key=value;key=value"
    raw = pathlib.Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    mhash = hashlib.sha256(raw).hexdigest()[:16]
    chash = code_hash()
    done = set()
    out = pathlib.Path(out_path)
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") != "ERROR":
                done.add((r["id"], r["model"], r["method"]))
    filt = {}
    if only:
        for kv in only.split(";"):
            k, v = kv.split("=")
            filt[k] = set(v.split("|"))
    tasks = []
    for e in manifest["entries"]:
        if any(str(e.get(k)) not in vs for k, vs in filt.items()):
            continue
        nmod = max(1, len(e.get("models", [])))
        for mi in range(nmod):
            for method in methods:
                if not applicable(e, method) or (e["id"], mi, method) in done:
                    continue
                tasks.append((e, mi, method, tl))
    print(f"{len(tasks)} tasks, {len(done)} already done", flush=True)
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
           "numpy": np.__version__, "workers": workers, "time_limit": tl, "code_sha": chash, "manifest_sha": mhash}
    for mod in ("scipy", "cvxpy", "clarabel", "highspy", "pyscipopt", "pymoo"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    env["memory_limit_bytes"] = MEMORY_LIMIT
    with Pool(workers, maxtasksperchild=20, initializer=limit_memory) as pool, out.open("a") as fh:
        for k, rec in enumerate(pool.imap_unordered(run_task, tasks)):
            rec.update(code_sha=chash, manifest_sha=mhash, time_limit=tl)
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            if k % 25 == 0:
                print(k, rec["id"], rec["method"], rec.get("status"), round(rec.get("wall", 0) or 0, 2), flush=True)
    (out.parent / (out.stem + "_environment.json")).write_text(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
