"""Comparators from the literature under the budget of the frozen protocol (protocol v1, section 8).

Usage: python run_literature.py <manifest.json> <results.jsonl> <methods> [workers] [time_limit] [filter]
Arguments are those of experiments/runner.py, whose loop, records and restart rule are reused.
Methods: the frozen ones (lcs, scip, ...) and asf, lgp, eps60, nsga60, rmpga60, rnsga60, scip_ind
from comparators/methods.py, on every structure. Each record also carries extra_sha, a hash of
the added code. The frozen code root is frozen/v1, or RMP_CODE_ROOT when set.
"""
import hashlib
import os
import pathlib
import sys
import time
import traceback
import warnings

HERE = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(os.environ.get("RMP_CODE_ROOT", HERE / "frozen" / "v1"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))                    # the frozen certrmp comes first

import runner  # noqa: E402
from certrmp.instance import priority_totals  # noqa: E402
from certrmp.model import model_from_dict  # noqa: E402
from comparators import methods  # noqa: E402
from manifest import build_instance, objectives_for  # noqa: E402

ADDED = ("asf", "lgp", "eps60", "nsga60", "rmpga60", "rnsga60", "scip_ind")
EXTRA_SHA = hashlib.sha256(b"".join(p.read_bytes() for p in (HERE / "comparators" / "methods.py",
                                                            pathlib.Path(__file__).resolve()))).hexdigest()[:16]
_frozen_task = runner.run_task
_frozen_applicable = runner.applicable


def run_added(entry, mi, method, tl):
    rec = {"id": entry["id"], "model": mi, "method": method, "n": entry["n"], "p": entry["p"], "m": entry["m"],
           "structure": entry["structure"], "seed": entry["seed"]}
    try:
        inst = build_instance(entry)
        objs = objectives_for(inst, entry["p"])
        md = entry["models"][mi]
        model = model_from_dict(md["model"])
        rec["family"], rec["kind"] = md["family"], md["kind"]
        t0 = time.perf_counter()
        sigma, _ = priority_totals(inst)
        if method == "scip_ind":
            r = methods.miqcp_scip_indicator(inst, objs, model, sigma, time_limit=tl)   # as the frozen scip
            rec["claimed_key"] = runner.keystr(r["claimed_key"])
        elif method == "asf":
            r = methods.asf_reference(inst, objs, model, sigma, time_limit=tl)
        elif method == "lgp":
            r = methods.goal_programming(inst, objs, model, sigma, time_limit=tl)
        elif method == "eps60":
            r = methods.eps_timed(inst, objs, model, sigma, time_limit=tl, seed=entry["seed"])
        else:
            r = methods.emo(inst, objs, model, sigma, method[:-2], time_limit=tl, seed=entry["seed"])
        x = r.pop("x", None)
        for k, v in r.items():
            if k == "key":
                rec[k] = runner.keystr(v)
            elif k == "f":
                rec[k] = [float(u) for u in v]
            elif k == "trajectory":
                rec[k] = [[t, runner.keystr(key)] for t, key in v]
            elif k != "claimed_key":
                rec[k] = v
        rec["seconds_with_priority"] = time.perf_counter() - t0
        if x is not None:
            rec["residual_physical"], rec["residual_priority"] = methods.residuals(inst, sigma, x)
        rec["wall"] = time.perf_counter() - t0
    except Exception as exc:
        rec["status"] = "ERROR"
        rec["error"] = repr(exc)[:300]
        rec["trace"] = traceback.format_exc()[-800:]
    return rec


def run_task(args):
    warnings.filterwarnings("ignore")
    entry, mi, method, tl = args
    rec = run_added(entry, mi, method, tl) if method in ADDED else _frozen_task(args)
    rec["extra_sha"] = EXTRA_SHA
    return rec


def applicable(entry, method):
    return method in ADDED or _frozen_applicable(entry, method)


runner.run_task = run_task
runner.applicable = applicable

if __name__ == "__main__":
    runner.main()
