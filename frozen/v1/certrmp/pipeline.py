"""End-to-end solve: physical check, exact priority totals, search, refinement,
certificate and independent verification, with explicit statuses.

Statuses
  OPTIMAL_VERIFIED     every claim checked in exact rational arithmetic
  OPTIMAL_NUMERICAL    search completed; some exact check failed or priority totals are
                       only numerical (the failed checks are listed)
  FEASIBLE_UNCERTIFIED a feasible allocation is returned but optimality is not established
  PHYSICAL_INFEASIBLE  the physical set is empty (verified exactly)
  UNRESOLVED           a convex call failed or the time limit was reached
"""
import json
import math
import time

import numpy as np

from .instance import priority_totals
from .oracle import Oracle
from .search import lcs, enumerate_grid, refine
from . import certify, verify as verifier


def solve(inst, objs, model, method="lcs", time_limit=None, do_verify=True,
          use_points=True, use_halfspaces=True, keep_certificate=False):
    rec = {"method": method, "n": inst.n, "p": len(objs), "m": model.m}
    t0 = time.perf_counter()
    sigma, exact = priority_totals(inst)
    rec["t_priority"] = time.perf_counter() - t0
    if sigma is None:
        rec.update(status="PHYSICAL_INFEASIBLE", seconds=time.perf_counter() - t0)
        if do_verify:
            cert = {"status": "PHYSICAL_INFEASIBLE", "instance": certify.instance_dict(inst),
                    "objectives": certify.objectives_dict(objs), "model": model.to_dict()}
            rec["verified"] = verifier.verify(cert)["verified"]
        return rec
    oracle = Oracle(inst, objs, sigma)
    t1 = time.perf_counter()
    if method == "lcs":
        res = lcs(oracle, model, use_points=use_points, use_halfspaces=use_halfspaces, time_limit=time_limit)
    elif method == "enum":
        res = enumerate_grid(oracle, model, time_limit=time_limit)
    else:
        raise ValueError(method)
    rec["t_search"] = time.perf_counter() - t1
    rec["counts"] = dict(res.counts)
    rec["search_calls"] = oracle.calls
    rec["solver_failures"] = oracle.failures
    rec["solver_retries"] = oracle.stats["retries"]
    if res.status != "OPTIMAL":
        rec.update(status="UNRESOLVED", reason=res.status, seconds=time.perf_counter() - t0)
        return rec
    rec["key"] = [str(v) for v in res.key]
    t2 = time.perf_counter()
    ref = refine(oracle, res.frontier)
    rec["t_refine"] = time.perf_counter() - t2
    rec["calls"] = oracle.calls
    rec["t_solver"] = oracle.solve_time
    if ref is None:
        rec.update(status="UNRESOLVED", reason="refinement failed", seconds=time.perf_counter() - t0)
        return rec
    rec["f"] = ref["f"].tolist()
    rec["x"] = ref["x"]
    rec["frontier_size"] = len(res.frontier)
    rec["seconds_to_key"] = rec["t_priority"] + rec["t_search"]
    if method != "lcs" or not do_verify:
        rec.update(status="OPTIMAL_NUMERICAL", seconds=time.perf_counter() - t0)
        return rec
    t3 = time.perf_counter()
    cert = certify.build(inst, objs, model, sigma, res, ref, exact)
    rec["t_certify"] = time.perf_counter() - t3
    rec["halfspaces"] = len(cert["halfspaces"])
    rec["certificate_bytes"] = len(json.dumps(cert))
    t4 = time.perf_counter()
    out = verifier.verify(json.loads(json.dumps(cert)))
    rec["t_verify"] = time.perf_counter() - t4
    if not out["checks"].get("witness_key") and out["checks"].get("levels"):
        # The refined allocation sits on a threshold within rounding. Fall back to the most
        # interior point of a frontier requirement set (achievement problem, t* < 0).
        for u in res.frontier:
            r = oracle.achievement(u)
            if r["status"] == "failed" or r["t"] >= 0:
                continue
            alt = dict(ref, x=r["x"], f=r["f"])
            cert2 = certify.build(inst, objs, model, sigma, res, alt, exact)
            out2 = verifier.verify(json.loads(json.dumps(cert2)))
            if out2["checks"].get("witness_key"):
                cert, out, ref = cert2, out2, alt
                rec["f"], rec["x"] = alt["f"].tolist(), alt["x"]
                rec["witness_fallback"] = True
                break
        rec["t_verify"] = time.perf_counter() - t4
    rec["checks"] = out["checks"]
    rec["errors"] = out["errors"]
    rec["refinement_epsilon_relative"] = out.get("refinement_epsilon_relative")
    if out["verified"] and out["exact"]:
        rec["status"] = "OPTIMAL_VERIFIED"
    elif out["checks"].get("witness_in_priority_set") and out["checks"].get("witness_key") \
            and not out["checks"].get("levels"):
        rec["status"] = "FEASIBLE_UNCERTIFIED"
    else:
        rec["status"] = "OPTIMAL_NUMERICAL"
    rec["seconds"] = time.perf_counter() - t0
    if keep_certificate:
        rec["certificate"] = cert
    return rec
