"""Comparators from the literature under equal budgets (exploratory, protocol v1, section 8).

Reads results/v1/literature/literature.jsonl (the added comparators and LCS run again on the
same pairs and machine) and results/v1/main.jsonl (the frozen runs of the same pairs), and
writes numbers_literature.tex, tab_literature.tex, literature.csv and literature_summary.json.
The reference is the certified key of LCS on each pair. A comparator reaches it when the key of
its returned allocation, evaluated in floating point, equals that key.
Usage: python analyze_literature.py <results_dir> <out_dir>
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_benchmark as ab  # noqa: E402

ANYTIME = ("asf", "lgp", "eps60", "nsga60", "rmpga60", "rnsga60")
MIQCP = ("scip", "scip_ind")
CAMEL = {"asf": "Asf", "lgp": "Lgp", "eps60": "Eps", "nsga60": "Nsga", "rmpga60": "Guided", "rnsga60": "Rnsga",
         "scip": "Scip", "scip_ind": "ScipInd"}
LABEL = {"asf": "Reference point method", "lgp": "Lexicographic goal programming",
         "eps60": "$\\varepsilon$-constraint, refined", "nsga60": "NSGA-II", "rmpga60": "RMP-guided NSGA-II",
         "rnsga60": "R-NSGA-II", "scip": "MIQCP, big-M, SCIP", "scip_ind": "MIQCP, indicators, SCIP"}


def load_lit(path):
    """Records of the added runs. A resumed run retries the tasks that ended in ERROR and appends
    the new record, so the last record of each task is kept and the retries are counted."""
    df = pd.DataFrame([json.loads(line) for line in ab.read_lines(path) if line.strip()])
    for col in ("manifest_sha", "code_sha", "extra_sha"):
        if df[col].nunique() > 1:
            raise SystemExit(f"{path}: records of several {col} values must not be pooled")
    retried = int(df.duplicated(subset=["id", "model", "method"]).sum())
    df = df.drop_duplicates(subset=["id", "model", "method"], keep="last").copy()
    df["t"] = df["seconds"].astype(float)
    return df, retried


def key_at(trajectory, t):
    """Best key held at time t, from the improvement times of an anytime run."""
    best = None
    for s, k in trajectory if isinstance(trajectory, list) else []:
        if s > t:
            break
        best = k
    return best


def compare(ref_keys, keys):
    """+1 when the reference key is better, 0 on a tie, -1 when the comparator is better."""
    return np.array([np.nan if y is None else ab.cmp_keys(ab.key_of(x), ab.key_of(y))
                     for x, y in zip(ref_keys, keys)], dtype=float)


def main():
    res = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tex = ab.Tex()
    lit, retried = load_lit(res / "literature" / "literature.jsonl")
    main_df = ab.load(res / "main.jsonl")
    if set(lit.manifest_sha) != set(main_df.manifest_sha) or set(lit.code_sha) != set(main_df.code_sha):
        raise SystemExit("the added runs used another manifest or frozen code version")
    errors = int((lit.status == "ERROR").sum())
    lcs = lit[lit.method == "lcs"].set_index(["id", "model"])
    if not lcs.status.isin(["OPTIMAL_VERIFIED", "OPTIMAL_NUMERICAL"]).all():
        raise SystemExit("LCS did not establish the key on every added pair")
    pairs = lcs.index
    frozen = main_df[main_df.method.isin(["lcs", "scip"])].set_index(["id", "model"])
    flcs = frozen[frozen.method == "lcs"].loc[pairs]
    same = int(sum(ab.cmp_keys(ab.key_of(a), ab.key_of(b)) == 0 for a, b in zip(lcs["key"], flcs["key"])))
    tex("LitPairs", ab.count(len(pairs)))
    tex("LitInstances", ab.count(len(set(i for i, _ in pairs))))
    tex("LitRecords", ab.count(len(lit)))
    tex("LitErrors", ab.count(errors))
    tex("LitRetried", ab.count(retried))
    tex("LitLcsKeySame", ab.count(same))
    tex("LitLcsSecMedian", ab.num(lcs.t.median(), 2))
    tex("LitLcsSecMax", ab.num(lcs.t.max(), 1))
    for n in sorted(lcs.n.unique()):
        tex(f"LitLcsSecMedianN{ab.NWORD[int(n)]}", ab.num(lcs.t[(lcs.n == n).values].median(), 2))
    summary = {"pairs": len(pairs), "records": int(len(lit)), "errors": errors, "retried": retried,
               "lcs_key_same_as_frozen": same}

    rows, cells = [], []
    ns = sorted(lcs.n.unique())
    for m in ANYTIME + MIQCP:
        src = frozen[frozen.method == "scip"] if m == "scip" else lit[lit.method == m].set_index(["id", "model"])
        d = src.reindex(pairs)
        c60 = compare(lcs["key"], d["key"].where(d["key"].notna(), None))
        r = {"method": m, "pairs": int(d.status.notna().sum()), "lcs_better": int(np.nansum(c60 == 1)),
             "tie": int(np.nansum(c60 == 0)), "comparator_better": int(np.nansum(c60 == -1)),
             "no_allocation": int(np.isnan(c60).sum())}
        tag = f"Lit{CAMEL[m]}"
        reach = np.where(np.isnan(c60), 0.0, (c60 == 0).astype(float))
        ids = [i for i, _ in pairs]
        est, lo, hi, _ = ab.cluster_boot_mean(ids, reach)
        r.update(reach_all=est, reach_low=lo, reach_high=hi)
        tex(f"{tag}ReachAll", ab.pct(est))
        tex(f"{tag}ReachLow", ab.pct(lo))
        tex(f"{tag}ReachHigh", ab.pct(hi))
        tex(f"{tag}Better", ab.count(r["comparator_better"]))
        tex(f"{tag}Worse", ab.pct(np.nansum(c60 == 1) / len(pairs)))
        tex(f"{tag}NoAlloc", ab.count(r["no_allocation"]))
        by_n = {}
        for n in ns:
            mask = (lcs.n == n).values
            by_n[n] = float(reach[mask].mean())
            tex(f"{tag}ReachN{ab.NWORD[int(n)]}", ab.pct(by_n[n]))
            r[f"reach_n{int(n)}"] = by_n[n]
        if m in ANYTIME:
            at = [key_at(tr, t) for tr, t in zip(d["trajectory"], lcs.t)]
            cl = compare(lcs["key"], at)
            r["reach_at_lcs_time"] = float(np.where(np.isnan(cl), 0.0, (cl == 0).astype(float)).mean())
            tex(f"{tag}ReachAtLcs", ab.pct(r["reach_at_lcs_time"]))
            hit = d[(c60 == 0)]
            r["median_seconds_to_key"] = float(hit["seconds_to_key"].median()) if len(hit) else np.nan
            tex(f"{tag}SecToKeyMedian", ab.num(r["median_seconds_to_key"], 1))
            tex(f"{tag}SecMedian", ab.num(d["t"].median(), 2))
            if "generations" in d and d["generations"].notna().any():
                for n in ns:
                    g = d[(lcs.n == n).values]["generations"]
                    tex(f"{tag}GenMedianN{ab.NWORD[int(n)]}", ab.count(g.median()))
            if "points" in d and d["points"].notna().any():
                for n in ns:
                    g = d[(lcs.n == n).values]["points"]
                    tex(f"{tag}PointsMedianN{ab.NWORD[int(n)]}", ab.count(g.median()))
        else:
            ok = (d.status == "OPTIMAL").values
            r["established"] = float(ok.mean())
            tex(f"{tag}SolvedAll", ab.pct(ok.mean()))
            for n in ns:
                mask = (lcs.n == n).values
                tex(f"{tag}SolvedN{ab.NWORD[int(n)]}", ab.pct(ok[mask].mean()))
                r[f"established_n{int(n)}"] = float(ok[mask].mean())
            claim = compare(lcs["key"], d["claimed_key"].where(d["claimed_key"].notna(), None))
            agree = int(np.nansum(claim[ok] == 0))
            miss = int(sum(1 for x, y in zip(d["claimed_key"][ok], d["key"][ok])
                           if ab.cmp_keys(ab.key_of(x), ab.key_of(y)) == 1))
            tex(f"{tag}ClaimAgree", ab.count(agree))
            tex(f"{tag}SolvedPairs", ab.count(int(ok.sum())))
            tex(f"{tag}ClaimMiss", ab.count(miss))
            tex(f"{tag}ClaimMissPct", ab.pct(miss / ok.sum()) if ok.sum() else "n/a")
            r.update(claim_agree=agree, claim_miss=miss)
        rows.append(r)
        cell = [LABEL[m]] + [ab.pct(by_n[n]) for n in ns]
        cell.append(f"{ab.pct(est)} [{ab.pct(lo)}, {ab.pct(hi)}]")
        cell.append(ab.pct(r["reach_at_lcs_time"]) if m in ANYTIME else "--")
        cells.append(" & ".join(cell) + " \\\\")
    table = pd.DataFrame(rows)
    better = int(table.comparator_better.sum())
    tex("LitComparatorBetter", ab.count(better))
    tex("LitBestReachAll", ab.pct(table[table.method.isin(ANYTIME)].reach_all.max()))
    tex("LitBestReachAtLcs", ab.pct(table[table.method.isin(ANYTIME)].reach_at_lcs_time.max()))
    (out / "tab_literature.tex").write_text("\n".join(cells[:len(ANYTIME)]) + "\n\\midrule\n"
                                            + "\n".join(cells[len(ANYTIME):]) + "\n")
    table.to_csv(out / "literature.csv", index=False)
    summary["methods"] = rows
    (out / "literature_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    tex.write(out / "numbers_literature.tex")
    text = (out / "numbers_literature.tex").read_text().replace("analyze_benchmark.py", "analyze_literature.py")
    (out / "numbers_literature.tex").write_text(text)
    print(f"{len(pairs)} pairs, {len(lit)} records, {errors} errors, {retried} retried, "
          f"LCS keys as frozen: {same}, comparator better than LCS: {better}")


if __name__ == "__main__":
    main()
