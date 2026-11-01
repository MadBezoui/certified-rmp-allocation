"""Prespecified analysis of the frozen benchmark (protocol v1, section 5).

Reads the immutable JSON-lines records and writes tables (CSV, JSON and LaTeX), figures (PDF)
and a LaTeX file of number macros used by the manuscript. Nothing is computed by hand.
Usage: python analyze_benchmark.py <results_dir> <out_dir>
"""
import gzip
import json
import math
import pathlib
import sys
from fractions import Fraction

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from certrmp.model import model_from_dict  # noqa: E402

BLUE, RED, BLACK = (0, 0, 1), (1, 0, 0), (0, 0, 0)
GRAY = (0.5, 0.5, 0.5)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "serif",
                     "mathtext.fontset": "cm", "axes.linewidth": 0.6, "lines.linewidth": 1.0})
WIDTH = 6.3
STYLE = {  # color, linestyle, marker: color is never the only encoding
    "lcs": (BLUE, "-", "o"), "enum": (BLACK, "--", "s"), "scip": (RED, "-.", "^"),
    "gurobi": (RED, ":", "v"), "eps": (GRAY, "-", "D"), "nsga": (GRAY, "--", "x"), "rmp_ga": (GRAY, ":", "+"),
    "lcs_nopoints": (BLUE, "--", "s"), "lcs_nohalf": (BLUE, ":", "^"), "lcs_nocache": (BLUE, "-.", "x"),
}
LABEL = {"lcs": "LCS (proposed)", "enum": "enumeration", "scip": "MIQCP, SCIP", "gurobi": "MIQCP, Gurobi",
         "eps": "epsilon-constraint", "nsga": "NSGA-II", "rmp_ga": "RMP-guided NSGA-II",
         "lcs_nopoints": "LCS without point cache", "lcs_nohalf": "LCS without halfspace cache",
         "lcs_nocache": "LCS without caches"}
TEXNAME = {"lcs": "LCS", "enum": "Enumeration", "scip": "MIQCP, SCIP", "gurobi": "MIQCP, Gurobi",
           "eps": "$\\varepsilon$-constraint", "nsga": "NSGA-II", "rmp_ga": "RMP-guided NSGA-II"}
LCS_FAMILY = ("lcs", "lcs_nopoints", "lcs_nohalf", "lcs_nocache")
NWORD = {30: "Thirty", 100: "Hundred", 300: "ThreeHundred", 1000: "Thousand"}
MWORD = {2: "Two", 4: "Four", 8: "Eight"}
CAMEL = {"lcs": "Lcs", "enum": "Enum", "scip": "Scip", "gurobi": "Gurobi", "eps": "Eps", "nsga": "Nsga",
         "rmp_ga": "Guided"}
BOOT = 2000
RNG_SEED = 20261030
LIMIT = 60.0


# ----------------------------------------------------------------------------- loading

def read_lines(path):
    """Lines of a JSON-lines file, read from its gzip-compressed copy when only that exists."""
    path = pathlib.Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        return gzip.decompress(path.with_name(path.name + ".gz").read_bytes()).decode().splitlines()
    return path.read_text().splitlines()


def exists(path):
    path = pathlib.Path(path)
    return path.exists() or path.with_name(path.name + ".gz").exists()


def load(path):
    rows = [json.loads(line) for line in read_lines(path) if line.strip()]
    df = pd.DataFrame(rows)
    for col in ("manifest_sha", "code_sha"):
        if col in df and df[col].nunique() > 1:
            raise SystemExit(f"{path}: records of several {col} values must not be pooled")
    dup = df.duplicated(subset=["id", "model", "method"]).sum()
    if dup:
        raise SystemExit(f"{path}: {dup} duplicate (id, model, method) records")
    df["ok"] = df.apply(solved, axis=1)
    df["t"] = df["seconds"].astype(float)
    df["ok60"] = df.ok & (df.t <= LIMIT)
    return df


def evaluate_enum_keys(df, manifest_path):
    """Threshold enumeration records the key of its grid search. As for the MIQCP comparators,
    that key is kept as the claimed key, and the key of the returned allocation is evaluated
    here in floating point from its objective vector."""
    rows = df.index[(df.method == "enum") & df["f"].apply(lambda v: isinstance(v, list))]
    if len(rows) == 0:
        return df
    entries = {e["id"]: e for e in json.loads(pathlib.Path(manifest_path).read_text())["entries"]}
    if "claimed_key" not in df.columns:
        df["claimed_key"] = None
    df["claimed_key"] = df["claimed_key"].astype(object)
    df["key"] = df["key"].astype(object)
    for i in rows:
        model = model_from_dict(entries[df.at[i, "id"]]["models"][int(df.at[i, "model"])]["model"])
        df.at[i, "claimed_key"] = df.at[i, "key"]
        df.at[i, "key"] = [str(v) for v in model.key(df.at[i, "f"])]
    return df


def solved(row):
    m, st = row["method"], row.get("status")
    if m in LCS_FAMILY:
        return st in ("OPTIMAL_VERIFIED", "OPTIMAL_NUMERICAL")
    if m == "enum":
        return st == "OPTIMAL_NUMERICAL"
    if m in ("scip", "gurobi"):
        return st == "OPTIMAL"
    return False


def key_of(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return tuple(Fraction(x) for x in v)


def cmp_keys(a, b):
    if a is None or b is None:
        return None
    return (a > b) - (a < b)


# ----------------------------------------------------------------------------- statistics

def cluster_boot_mean(ids, values, n=BOOT, seed=RNG_SEED):
    """Mean of values with a percentile interval from resampling whole clusters (instances)."""
    d = pd.DataFrame({"id": list(ids), "v": np.asarray(values, dtype=float)}).dropna()
    g = d.groupby("id")["v"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(s), np.full(len(s), 1 / len(s)), size=n)
    vals = (w @ s) / (w @ c)
    return float(s.sum() / c.sum()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def p_one_sided(vals):
    """Legacy uncentred bootstrap tail, retained only to reproduce historical outputs.

    This is not a calibrated null p-value. Use the separate post-audit sign-flip
    sensitivity analysis and its stated exchangeability assumption for inference.
    """
    return float((np.sum(vals <= 0) + 1) / (len(vals) + 1))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired(df, a, b):
    x = df[df.method == a].set_index(["id", "model"])
    y = df[df.method == b].set_index(["id", "model"])
    common = x.index.intersection(y.index)
    return x.loc[common], y.loc[common]


# ----------------------------------------------------------------------------- figures

def solved_curve(df, method, budgets):
    d = df[df.method == method]
    if d.empty:
        return np.full(len(budgets), np.nan)
    return np.array([np.mean(d.ok.values & (d.t.values <= b)) for b in budgets])


def figure_solved(main, out, budgets):
    ns = sorted(main.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(WIDTH, 2.6), sharey=True, layout="constrained")
    for ax, n in zip(np.atleast_1d(axes), ns):
        d = main[main.n == n]
        for m in ("lcs", "enum", "scip", "gurobi"):
            if (d.method == m).any():
                c, ls, mk = STYLE[m]
                ax.plot(budgets, solved_curve(d, m, budgets), color=c, linestyle=ls, marker=mk, markevery=8,
                        markersize=3, label=LABEL[m])
        ax.set_xscale("log")
        ax.set_title(f"$n = {n}$")
        ax.set_xlabel("time budget (s)")
        ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("fraction of pairs solved")
    # one legend above the panels, so that no curve is hidden
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels), frameon=False)
    fig.savefig(out / "fig4_solved_fraction.pdf")
    plt.close(fig)


def figure_scaling(main, out):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    ax = axes[0]
    for m in ("lcs", "enum", "scip"):
        d = main[main.method == m]
        if d.empty:
            continue
        t = pd.Series(np.where(d.ok60, d.t, np.inf), index=d.index)
        med = t.groupby(d.n).median()
        c, ls, mk = STYLE[m]
        ax.plot(med.index, med.replace(np.inf, np.nan).values, color=c, linestyle=ls, marker=mk, markersize=3,
                label=LABEL[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ns = sorted(main.n.unique())
    ax.set_xticks(ns)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("recipients $n$")
    ax.set_ylabel("median time (s)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    for m in ("lcs", "enum"):
        d = main[(main.method == m) & main.ok]
        if d.empty:
            continue
        med = d.groupby("m")["search_calls"].median()
        c, ls, mk = STYLE[m]
        ax.plot(med.index, med.values, color=c, linestyle=ls, marker=mk, markersize=3, label=f"{LABEL[m]}, median")
    ms = np.array(sorted(main.m.unique()))
    p = 3
    ax.plot(ms, (ms + 1) ** p + p, color=BLACK, linestyle=":", label=r"$(m+1)^p + p$")
    ax.plot(ms, p + ms * 2 ** p, color=BLUE, linestyle=":", label=r"$p + m 2^p$")
    ax.set_yscale("log")
    ax.set_xticks(ms)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("profiles $m$")
    ax.set_ylabel("convex calls of the search")
    ax.legend(frameon=False, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig5_scaling.pdf")
    plt.close(fig)


def figure_ablation(abl, out):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    methods = [m for m in ("lcs", "lcs_nopoints", "lcs_nohalf", "lcs_nocache", "enum") if (abl.method == m).any()]
    for ax, col, lab in ((axes[0], "search_calls", "convex calls of the search"), (axes[1], "t_search", "search time (s)")):
        data = [abl[(abl.method == m) & abl.ok][col].astype(float).dropna().values for m in methods]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor((0, 0, 0, 0.1))
            patch.set_edgecolor(STYLE[m][0])
            patch.set_linestyle(STYLE[m][1])
        for med in bp["medians"]:
            med.set_color(BLACK)
        ax.set_yticks(range(1, len(methods) + 1))
        ax.set_yticklabels([LABEL[m] for m in methods] if ax is axes[0] else [""] * len(methods))
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig6_ablation.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- comparisons

def key_table(df, ref="lcs", methods=("enum", "scip", "gurobi", "eps", "nsga", "rmp_ga")):
    """Wins, ties and losses of the reference key against each comparator, on pairs where the
    reference established its key. Claimed keys (solver status) and evaluated keys (key of the
    returned allocation in floating point) are counted separately."""
    rows = []
    for m in methods:
        a, b = paired(df, ref, m)
        keep = a.ok.values
        a, b = a[keep], b[keep]
        if len(a) == 0:
            continue
        r = {"method": m, "pairs": len(a)}
        for kind, tag in (("claimed_key", "claimed"), ("key", "evaluated")):
            if kind not in b.columns:
                continue
            c = [cmp_keys(key_of(x), key_of(y)) for x, y in zip(a["key"], b[kind])]
            r.update({f"{tag}_lcs_better": sum(1 for v in c if v == 1), f"{tag}_tie": sum(1 for v in c if v == 0),
                      f"{tag}_lcs_worse": sum(1 for v in c if v == -1),
                      f"{tag}_missing": sum(1 for v in c if v is None)})
        rows.append(r)
    return pd.DataFrame(rows)


def dominates(fa, fb, tol=1e-9):
    fa, fb = np.asarray(fa, float), np.asarray(fb, float)
    eps = tol * (1 + np.abs(fa))
    return bool(np.all(fa <= fb + eps) and np.any(fa < fb - eps))


def dominance_table(df, ref="lcs", methods=("enum", "scip", "gurobi", "eps", "nsga", "rmp_ga")):
    rows = []
    for m in methods:
        a, b = paired(df, ref, m)
        keep = np.array([isinstance(x, list) and isinstance(y, list) for x, y in zip(a["f"], b["f"])], bool)
        a, b = a[keep], b[keep]
        if len(a) == 0:
            continue
        rows.append({"method": m, "pairs": len(a),
                     "lcs_dominates": sum(dominates(x, y) for x, y in zip(a["f"], b["f"])),
                     "comparator_dominates": sum(dominates(y, x) for x, y in zip(a["f"], b["f"]))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- LaTeX helpers

class Tex:
    def __init__(self):
        self.lines = []

    def __call__(self, name, value):
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}\n")

    def write(self, path):
        pathlib.Path(path).write_text("% Generated by scripts/analyze_benchmark.py. Do not edit.\n" + "".join(self.lines))


def finite(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def pct(x, nd=1):
    return f"{100 * float(x):.{nd}f}" if finite(x) else "n/a"


def num(x, nd=2):
    return f"{float(x):.{nd}f}" if finite(x) else "n/a"


def count(x):
    return f"{int(x):,}".replace(",", "{,}") if finite(x) else "n/a"


def sci(x):
    if not finite(x) or float(x) == 0:
        return "0"
    e = int(math.floor(math.log10(abs(float(x)))))
    return f"{float(x) / 10 ** e:.1f} \\times 10^{{{e}}}"


def subset(d, groups, values):
    for g, v in zip(groups, values):
        d = d[d[g] == v]
    return d


# ----------------------------------------------------------------------------- main

FAMILY_LABEL = {"easy": "all attainable", "unattainable": "none attainable", "adjacent": "on efficient points",
                "near_front": "nonadditive table"}
VARIANT_LABEL = {"base": "base model", "perturbed": "perturbed profiles", "reversed": "reversed order",
                 "concentrated": "concentrated weights", "equal": "equal weights", "table": "nonadditive table"}
ORDER = {**{k: i for i, k in enumerate(FAMILY_LABEL)}, **{k: i for i, k in enumerate(VARIANT_LABEL)}, "None": -1}


def group_order(design, grp):
    """Families and variants in the order of the design text, absent values first, numbers
    ascending except ramp limits, which go from loose to tight."""
    out = []
    for pos, v in enumerate(grp):
        if v in ORDER:
            out.append((0, ORDER[v]))
        else:
            loose_first = design == "sensitivity" and pos == 1
            out.append((1, -float(v) if loose_first else float(v)))
    return out


def group_label(design, grp):
    """Row label of the design table, in the words of the design section."""
    if design == "stress_profiles":
        return FAMILY_LABEL.get(grp[0], grp[0])
    if design == "preference":
        return VARIANT_LABEL.get(grp[0], grp[0])
    if design == "stress_dimension":
        return f"$p = {grp[0]}$"
    scarcity, ramp, tol = grp
    parts = [f"${100 * float(scarcity):.0f}\\%$", "no ramp" if ramp == "None" else f"${float(ramp):g}\\, d$"]
    if tol != "None":
        parts.append(f"$10^{{{int(round(__import__('math').log10(float(tol))))}}}$")
    return ", ".join(parts)


def main():
    res = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tex = Tex()
    summary = {}
    main_df = evaluate_enum_keys(load(res / "main.jsonl"), res / "manifest_main.json")
    budgets = np.logspace(-2, math.log10(LIMIT), 60)
    p = 3
    tex("NumMainInstances", count(main_df.id.nunique()))
    tex("NumMainPairs", count(main_df[["id", "model"]].drop_duplicates().shape[0]))
    tex("NumMainRuns", count(len(main_df)))

    # 1. statuses
    status = main_df.groupby(["method", "status"]).size().unstack(fill_value=0)
    status.to_csv(out / "main_status.csv")
    summary["status"] = {m: {k: int(v) for k, v in r.items() if v} for m, r in status.iterrows()}

    # 2. solved fractions at 60 s, overall and by n, with cluster bootstrap intervals
    rows = []
    for m in ("lcs", "enum", "scip", "gurobi"):
        d = main_df[main_df.method == m]
        for n in [None] + sorted(d.n.unique()):
            dd = d if n is None else d[d.n == n]
            if dd.empty:
                continue
            est, lo, hi, _ = cluster_boot_mean(dd.id, dd.ok60)
            rows.append({"method": m, "n": "all" if n is None else int(n), "pairs": len(dd), "solved_60s": est,
                         "ci_low": lo, "ci_high": hi,
                         "median_seconds_solved": float(dd[dd.ok60].t.median()) if dd.ok60.any() else np.nan})
            tex(f"Solved{CAMEL[m]}{'All' if n is None else 'N' + NWORD[int(n)]}", pct(est))
    solved_tab = pd.DataFrame(rows)
    solved_tab.to_csv(out / "main_solved.csv", index=False)
    lines = []
    for n in sorted(main_df.n.unique()) + ["all"]:
        cells = [str(int(n)) if n != "all" else "all"]     # n as in the text, without separator
        for m in ("lcs", "enum", "scip", "gurobi"):
            r = solved_tab[(solved_tab.method == m) & (solved_tab.n.astype(str) == str(n))]
            if r.empty:
                cells += ["--", "--"]
                continue
            r = r.iloc[0]
            cells += [f"{pct(r.solved_60s)} [{pct(r.ci_low)}, {pct(r.ci_high)}]", num(r.median_seconds_solved, 2)]
        lines.append(" & ".join(cells) + " \\\\")
    (out / "tab_solved.tex").write_text("\n".join(lines) + "\n")

    # 3. LCS statuses and certificates
    l = main_df[main_df.method == "lcs"]
    tex("LcsVerifiedPct", pct((l.status == "OPTIMAL_VERIFIED").mean()))
    tex("LcsVerifiedCount", count((l.status == "OPTIMAL_VERIFIED").sum()))
    tex("LcsNumericalCount", count((l.status == "OPTIMAL_NUMERICAL").sum()))
    tex("LcsUncertifiedCount", count((l.status == "FEASIBLE_UNCERTIFIED").sum()))
    tex("LcsUnresolvedCount", count((l.status == "UNRESOLVED").sum()))
    fb = l["witness_fallback"] if "witness_fallback" in l else pd.Series(False, index=l.index)
    tex("LcsFallbackCount", count(fb.fillna(False).astype(bool).sum()))
    tex("LcsRetries", count(l.solver_retries.fillna(0).sum()))
    tex("LcsFailures", count(l.solver_failures.fillna(0).sum()))
    lv = l[l.ok]
    kb = lv.certificate_bytes.astype(float) / 1024
    tex("CertKBMedian", num(kb.median(), 1))
    tex("CertKBMax", num(kb.max(), 1))
    for n in sorted(lv.n.unique()):
        dn = lv[lv.n == n]
        w = NWORD[int(n)]
        tex(f"CertKBMedianN{w}", num(dn.certificate_bytes.median() / 1024, 1))
        tex(f"VerifySecMedianN{w}", num(dn.t_verify.median(), 3))
        tex(f"LcsSecMedianN{w}", num(dn.t.median(), 2))
        tex(f"LcsKeySecMedianN{w}", num(dn.seconds_to_key.median(), 2))
        tex(f"CertShareMedianN{w}", pct(((dn.t_certify + dn.t_verify) / dn.t).median(), 0))
    tex("VerifySecMax", num(lv.t_verify.max(), 2))
    tex("HalfspacesMedian", num(lv.halfspaces.median(), 0))
    tex("HalfspacesMax", count(lv.halfspaces.max()))
    tex("RefineEpsMax", sci(lv.refinement_epsilon_relative.astype(float).max()))
    tex("RefineEpsMedian", sci(lv.refinement_epsilon_relative.astype(float).median()))
    tex("FrontierMax", count(lv.frontier_size.max()))
    tex("FrontierOnePct", pct((lv.frontier_size == 1).mean()))
    tex("LcsSearchCallsMax", count(lv.search_calls.max()))
    bound = p + lv.m * 2 ** p
    tex("LcsBoundExceed", count((lv.search_calls > bound).sum()))
    for m_ in sorted(main_df.m.unique()):
        for meth in ("lcs", "enum"):
            dm = main_df[(main_df.method == meth) & main_df.ok & (main_df.m == m_)]
            tex(f"{CAMEL[meth]}CallsMedianM{MWORD[int(m_)]}", num(dm.search_calls.median(), 0) if len(dm) else "n/a")
            tex(f"{CAMEL[meth]}CallsMaxM{MWORD[int(m_)]}", count(dm.search_calls.max()) if len(dm) else "n/a")
    counts = pd.DataFrame(list(lv["counts"])).sum()
    decided = counts.sum()
    for k in ("solve", "point", "halfspace", "dominance", "memo"):
        tex(f"Decided{k.capitalize()}Pct", pct(counts.get(k, 0) / decided))
    summary["certificates"] = {"verified": int((l.status == "OPTIMAL_VERIFIED").sum()), "pairs": len(l),
                               "kb_median": float(kb.median()), "kb_max": float(kb.max()),
                               "verify_max": float(lv.t_verify.max()), "decisions": counts.to_dict(),
                               "bound_exceed": int((lv.search_calls > bound).sum())}
    per_nm = lv.groupby(["n", "m"]).agg(calls=("search_calls", "median"), seconds=("t", "median"),
                                        verify=("t_verify", "median"), kb=("certificate_bytes", "median"))
    per_nm["kb"] = per_nm["kb"] / 1024
    per_nm.to_csv(out / "lcs_by_n_m.csv")

    # 4. confirmatory hypotheses (Holm family H1-H3)
    hyp = {}
    nmax = int(main_df.n.max())
    big = main_df[main_df.n == nmax]
    a, b = paired(big, "lcs", "scip")
    if len(a):
        d = a.ok60.astype(float).values - b.ok60.astype(float).values
        est, lo, hi, vals = cluster_boot_mean([i for i, _ in a.index], d)
        hyp["H1"] = {"n": nmax, "diff": est, "ci": [lo, hi], "p": p_one_sided(vals), "pairs": len(a),
                     "lcs_solved": float(a.ok60.mean()), "scip_solved": float(b.ok60.mean())}
    a, b = paired(main_df, "lcs", "enum")
    ca = a.calls.where(a.calls.notna(), a.search_calls).astype(float).values
    cb = b.calls.where(b.calls.notna(), b.search_calls).astype(float).values
    good = (ca > 0) & (cb > 0)
    if good.any():
        lr = np.log(cb[good]) - np.log(ca[good])
        est, lo, hi, vals = cluster_boot_mean([i for (i, _), g in zip(a.index, good) if g], lr)
        hyp["H2"] = {"mean_log_ratio": est, "ratio": math.exp(est), "ci_ratio": [math.exp(lo), math.exp(hi)],
                     "p": p_one_sided(vals), "pairs": int(good.sum())}
    a, b = paired(main_df, "lcs", "eps")
    c = [cmp_keys(key_of(x), key_of(y)) for x, y in zip(a["key"], b["key"])]
    keep = [v is not None for v in c]
    if any(keep):
        d = np.array([v for v in c if v is not None], float)
        ids = [i for (i, _), k in zip(a.index, keep) if k]
        est, lo, hi, vals = cluster_boot_mean(ids, d)
        hyp["H3"] = {"wins_minus_losses_rate": est, "ci": [lo, hi], "p": p_one_sided(vals),
                     "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum()),
                     "missing": int(len(c) - len(d))}
    names = [h for h in ("H1", "H2", "H3") if h in hyp]
    adj = holm(np.array([hyp[h]["p"] for h in names])) if names else []
    for h, v in zip(names, adj):
        hyp[h]["p_holm"] = float(v)
    summary["hypotheses"] = hyp
    if "H1" in hyp:
        h = hyp["H1"]
        tex("HOneLcs", pct(h["lcs_solved"]))
        tex("HOneScip", pct(h["scip_solved"]))
        tex("HOneDiff", pct(h["diff"]))
        tex("HOneLow", pct(h["ci"][0]))
        tex("HOneHigh", pct(h["ci"][1]))
        tex("HOnePHolm", num(h["p_holm"], 4))
        tex("HOnePairs", count(h["pairs"]))
    if "H2" in hyp:
        h = hyp["H2"]
        tex("HTwoRatio", num(h["ratio"], 1))
        tex("HTwoLow", num(h["ci_ratio"][0], 1))
        tex("HTwoHigh", num(h["ci_ratio"][1], 1))
        tex("HTwoPHolm", num(h["p_holm"], 4))
        tex("HTwoPairs", count(h["pairs"]))
    if "H3" in hyp:
        h = hyp["H3"]
        tex("HThreeWins", count(h["wins"]))
        tex("HThreeTies", count(h["ties"]))
        tex("HThreeLosses", count(h["losses"]))
        tex("HThreePHolm", num(h["p_holm"], 4))
        tex("HThreeMissing", count(h["missing"]))
    tex("BootMinP", num(1 / (BOOT + 1), 4))

    # 5. time ratios on pairs solved by both (geometric means), always next to solved fractions
    ratios = []
    for other in ("scip", "enum", "gurobi"):
        for n in sorted(main_df.n.unique()):
            a, b = paired(main_df[main_df.n == n], "lcs", other)
            both = a.ok60.values & b.ok60.values
            if both.sum() == 0:
                continue
            r = float(np.exp(np.mean(np.log(b.t.values[both]) - np.log(a.t.values[both]))))
            ratios.append({"other": other, "n": int(n), "pairs_both": int(both.sum()), "geo_ratio_other_over_lcs": r})
            tex(f"Ratio{CAMEL[other]}N{NWORD[int(n)]}", num(r, 1))
    pd.DataFrame(ratios).to_csv(out / "time_ratios.csv", index=False)

    # 6. key agreement, dominance and residuals
    frames = [main_df]
    if exists(res / "main_ga.jsonl"):
        ga = load(res / "main_ga.jsonl")
        frames.append(ga)
        tex("GuidedPairs", count((ga.method == "rmp_ga").sum()))
        tex("NsgaPairs", count((ga.method == "nsga").sum()))
    allm = pd.concat(frames, ignore_index=True)
    kt = key_table(allm)
    kt.to_csv(out / "main_keys.csv", index=False)
    dt = dominance_table(allm)
    dt.to_csv(out / "main_dominance.csv", index=False)
    lines = []
    for m in ("enum", "scip", "gurobi", "eps", "nsga", "rmp_ga"):
        k = kt[kt.method == m]
        dd = dt[dt.method == m]
        if k.empty:
            continue
        k = k.iloc[0]
        has_claim = m in ("enum", "scip", "gurobi") and "claimed_tie" in k.index and finite(k["claimed_tie"])
        claimed = (f"{count(k.claimed_lcs_better)} / {count(k.claimed_tie)} / {count(k.claimed_lcs_worse)}"
                   if has_claim else "--")
        ev = f"{count(k.evaluated_lcs_better)} / {count(k.evaluated_tie)} / {count(k.evaluated_lcs_worse)}"
        dom = f"{count(dd.iloc[0].lcs_dominates)} / {count(dd.iloc[0].comparator_dominates)}" if len(dd) else "--"
        lines.append(f"{TEXNAME[m]} & {count(k.pairs)} & {claimed} & {ev} & {dom} \\\\")
        for kind in ("claimed", "evaluated"):
            col = f"{kind}_tie"
            if col in k.index and finite(k[col]) and (kind == "evaluated" or has_claim):
                tex(f"Key{CAMEL[m]}{kind.capitalize()}Better", count(k[f"{kind}_lcs_better"]))
                tex(f"Key{CAMEL[m]}{kind.capitalize()}Tie", count(k[col]))
                tex(f"Key{CAMEL[m]}{kind.capitalize()}Worse", count(k[f"{kind}_lcs_worse"]))
                tex(f"Key{CAMEL[m]}{kind.capitalize()}Missing", count(k[f"{kind}_missing"]))
        tex(f"Key{CAMEL[m]}Pairs", count(k.pairs))
        tex(f"Key{CAMEL[m]}TiePct", pct(k.evaluated_tie / k.pairs))
        if len(dd):
            tex(f"Dom{CAMEL[m]}ByLcs", count(dd.iloc[0].lcs_dominates))
            tex(f"Dom{CAMEL[m]}OverLcs", count(dd.iloc[0].comparator_dominates))
    (out / "tab_quality.tex").write_text("\n".join(lines) + "\n")
    for m in ("enum", "scip", "gurobi"):
        d = main_df[(main_df.method == m) & main_df.ok]
        if len(d):
            if m != "enum":
                tex(f"Resid{CAMEL[m]}Physical", sci(d.residual_physical.astype(float).max()))
                tex(f"Resid{CAMEL[m]}Priority", sci(d.residual_priority.astype(float).max()))
            miss = sum(1 for x, y in zip(d.claimed_key, d.key) if cmp_keys(key_of(x), key_of(y)) == 1)
            tex(f"{CAMEL[m]}ClaimMiss", count(miss))
            tex(f"{CAMEL[m]}ClaimMissPct", pct(miss / len(d)))
            tex(f"{CAMEL[m]}SolvedPairs", count(len(d)))
            a, b = paired(main_df, "lcs", m)
            keep = a.ok.values & b.ok.values
            same = sum(1 for x, y in zip(a["key"][keep], b["claimed_key"][keep]) if cmp_keys(key_of(x), key_of(y)) == 0)
            tex(f"{CAMEL[m]}SolvedClaimAgree", count(same))
            tex(f"{CAMEL[m]}SolvedBoth", count(int(keep.sum())))
    for m in ("eps", "nsga", "rmp_ga"):
        d = allm[allm.method == m]
        if len(d):
            tex(f"{CAMEL[m]}SecMedian", num(d.t.median(), 2))
            tex(f"{CAMEL[m]}SecMax", num(d.t.max(), 1))

    # 7. figures
    figure_solved(main_df, out, budgets)
    figure_scaling(main_df, out)

    # 8. ablation
    abl_path = res / "main_ablation.jsonl"
    if exists(abl_path):
        abl = load(abl_path)
        keys = abl[["id", "model"]].drop_duplicates()
        base = main_df[main_df.method.isin(["lcs", "enum"])].merge(keys, on=["id", "model"])
        abl = pd.concat([abl, base], ignore_index=True)
        figure_ablation(abl, out)
        # ablations stop after the refinement and build no certificate: compare search times
        at = abl.groupby(["method", "n"]).agg(pairs=("id", "size"), solved=("ok60", "mean"),
                                             calls=("search_calls", "median"), seconds=("t_search", "median"))
        at.to_csv(out / "ablation.csv")
        summary["ablation"] = {f"{a}|{b}": v for (a, b), v in at.to_dict(orient="index").items()}
        for meth in ("lcs", "lcs_nopoints", "lcs_nohalf", "lcs_nocache", "enum"):
            for n in sorted(abl.n.unique()):
                if (meth, n) in at.index:
                    nm = "Abl" + "".join(w.capitalize() for w in meth.split("_")) + "N" + NWORD[int(n)]
                    tex(nm + "Calls", num(at.loc[(meth, n), "calls"], 0))
                    tex(nm + "Sec", num(at.loc[(meth, n), "seconds"], 2))
                    tex(nm + "Solved", pct(at.loc[(meth, n), "solved"]))
        # an ablation removes caches only, so it must reach the key of LCS on every pair
        for meth in ("lcs_nopoints", "lcs_nohalf", "lcs_nocache"):
            a, b = paired(abl, "lcs", meth)
            keep = a.ok.values & b.ok.values
            a, b = a[keep], b[keep]
            c = [cmp_keys(key_of(x), key_of(y)) for x, y in zip(a["key"], b["key"])]
            ratio = np.exp(np.mean(np.log(b.search_calls.astype(float).values / a.search_calls.astype(float).values))) \
                if len(a) else float("nan")
            nm = "Abl" + "".join(w.capitalize() for w in meth.split("_"))
            tex(nm + "KeyPairs", count(len(c)))
            tex(nm + "KeyDiffer", count(sum(1 for v in c if v != 0)))
            tex(nm + "CallRatio", num(ratio, 2))
            summary.setdefault("ablation_keys", {})[meth] = {"pairs": len(c), "differ": sum(1 for v in c if v != 0),
                                                             "call_ratio": ratio}

    # 9. stress, sensitivity and preference designs
    for design in ("stress_profiles", "stress_dimension", "sensitivity", "preference"):
        path = res / f"{design}.jsonl"
        if not exists(path):
            continue
        d = evaluate_enum_keys(load(path), res / f"manifest_{design}.json")
        groups = {"stress_profiles": ["family"], "stress_dimension": ["p"],
                  "sensitivity": ["scarcity", "ramp", "solver_tol"], "preference": ["variant"]}[design]
        for g in groups:
            if g not in d.columns:
                d[g] = None
        # absent values (no ramp limit, default tolerance) form their own group: pandas drops
        # missing keys in groupby, and its string dtype keeps them missing
        d[groups] = d[groups].astype(object).where(d[groups].notna(), "None").astype(str)
        tab = d.groupby(groups + ["method"]).agg(
            pairs=("id", "size"), solved=("ok60", "mean"), calls=("search_calls", "median"),
            seconds=("t", "median"), verified=("status", lambda s: float((s == "OPTIMAL_VERIFIED").mean())),
            infeasible=("status", lambda s: float((s == "PHYSICAL_INFEASIBLE").mean())))
        tab.to_csv(out / f"{design}.csv")
        kt = key_table(d, methods=("enum", "scip", "eps"))
        kt.to_csv(out / f"{design}_keys.csv", index=False)
        summary[design] = {"status": {f"{a}|{b}": int(v) for (a, b), v in d.groupby(["method", "status"]).size().items()},
                           "keys": kt.to_dict(orient="records")}

        def cell(grp, m, col, fmt):
            k = tuple(grp) + (m,)
            return fmt(tab.loc[k, col]) if k in tab.index else "--"

        rows_tex = []
        for grp in sorted((g if isinstance(g, tuple) else (g,) for g, _ in d.groupby(groups)), key=lambda g: group_order(design, g)):
            cells = [group_label(design, grp)]
            cells += [cell(grp, m, "solved", lambda v: pct(v, 0)) for m in ("lcs", "enum", "scip")]
            cells += [cell(grp, "lcs", "verified", lambda v: pct(v, 0)), cell(grp, "lcs", "calls", lambda v: num(v, 0)),
                      cell(grp, "enum", "calls", lambda v: num(v, 0)), cell(grp, "lcs", "seconds", lambda v: num(v, 2)),
                      cell(grp, "scip", "seconds", lambda v: num(v, 2))]
            a = subset(d[d.method == "lcs"], groups, grp).set_index(["id", "model"])
            b = subset(d[d.method == "eps"], groups, grp).set_index(["id", "model"])
            common = a.index.intersection(b.index)
            c = [cmp_keys(key_of(a.loc[i, "key"]), key_of(b.loc[i, "key"])) for i in common]
            both = sum(1 for v in c if v is not None)
            cells.append(f"{sum(1 for v in c if v == 0)}/{both}" if both else "--")
            rows_tex.append(" & ".join(cells) + " \\\\")
        (out / f"tab_{design}.tex").write_text("\n".join(rows_tex) + "\n")
        tag = "".join(w.capitalize() for w in design.split("_"))
        lcs_d = d[d.method == "lcs"]
        tex(f"{tag}LcsSolved", pct(lcs_d.ok60.mean()))
        tex(f"{tag}LcsVerified", pct((lcs_d.status == "OPTIMAL_VERIFIED").mean()))
        tex(f"{tag}Pairs", count(len(lcs_d)))
        tex(f"{tag}Infeasible", count((lcs_d.status == "PHYSICAL_INFEASIBLE").sum()))
        sc = d[d.method == "scip"]
        if len(sc):
            tex(f"{tag}ScipSolved", pct(sc.ok60.mean()))
        for m in ("enum", "scip", "eps"):
            k = kt[kt.method == m] if len(kt) else kt
            if len(k):
                k = k.iloc[0]
                tex(f"{tag}Key{CAMEL[m]}Worse", count(k.evaluated_lcs_worse))
                tex(f"{tag}Key{CAMEL[m]}Tie", count(k.evaluated_tie))
                tex(f"{tag}Key{CAMEL[m]}Better", count(k.evaluated_lcs_better))
                if m == "enum" and "claimed_tie" in k.index and finite(k["claimed_tie"]):
                    tex(f"{tag}KeyEnumClaimedWorse", count(k.claimed_lcs_worse))
                    tex(f"{tag}KeyEnumClaimedTie", count(k.claimed_tie))
                    tex(f"{tag}KeyEnumClaimedBetter", count(k.claimed_lcs_better))
        # values of single rows quoted in the text
        words = {"stress_profiles": {"easy": "Easy", "unattainable": "Unattainable", "adjacent": "Adjacent",
                                     "near_front": "Table"},
                 "stress_dimension": {"2": "PTwo", "3": "PThree", "5": "PFive", "8": "PEight"}}.get(design, {})
        for g, w in words.items():
            for m, col, name, fmt in (("lcs", "solved", "LcsSolved", lambda v: pct(v, 0)),
                                      ("lcs", "verified", "LcsVerified", lambda v: pct(v, 0)),
                                      ("enum", "solved", "EnumSolved", lambda v: pct(v, 0)),
                                      ("scip", "solved", "ScipSolved", lambda v: pct(v, 0)),
                                      ("lcs", "calls", "LcsCalls", lambda v: num(v, 0)),
                                      ("enum", "calls", "EnumCalls", lambda v: num(v, 0)),
                                      ("lcs", "seconds", "LcsSec", lambda v: num(v, 2))):
                if (g, m) in tab.index:
                    tex(f"{tag}{w}{name}", fmt(tab.loc[(g, m), col]))
        if design in ("sensitivity", "preference") and len(sc):
            s = tab.xs("scip", level="method")["solved"]
            tex(f"{tag}ScipSolvedMin", pct(s.min(), 0))
            tex(f"{tag}ScipSolvedMax", pct(s.max(), 0))
            c = tab.xs("lcs", level="method")["calls"]
            tex(f"{tag}LcsCallsMin", num(c.min(), 0))
            tex(f"{tag}LcsCallsMax", num(c.max(), 0))
            v = tab.xs("lcs", level="method")["verified"]
            tex(f"{tag}LcsVerifiedMin", pct(v.min(), 0))
        if design == "sensitivity":
            # the looser conic tolerance repeats one cell on the same seeds
            lc = d[d.method == "lcs"]
            both = lc[lc.solver_tol != "None"].merge(lc[lc.solver_tol == "None"], on=["seed", "model", "scarcity", "ramp"],
                                                    suffixes=("", "_base"))
            same = sum(1 for x, y in zip(both.key, both.key_base) if cmp_keys(key_of(x), key_of(y)) == 0)
            tex("SensitivityTolPairs", count(len(both)))
            tex("SensitivityTolKeySame", count(same))
        # comparator allocations whose floating-point key beats an exactly verified LCS key
        lv = d[(d.method == "lcs") & (d.status == "OPTIMAL_VERIFIED")].set_index(["id", "model"])
        beat = []
        for m in ("enum", "scip", "eps"):
            b = d[d.method == m].set_index(["id", "model"])
            for i in lv.index.intersection(b.index):
                if cmp_keys(key_of(lv.loc[i, "key"]), key_of(b.loc[i, "key"])) == -1:
                    row = b.loc[i]
                    beat.append({"method": m, "id": i[0], "model": int(i[1]),
                                 "residual_physical": row.get("residual_physical"),
                                 "residual_priority": row.get("residual_priority")})
        summary[design]["beat_verified"] = beat
        tex(f"{tag}BeatVerified", count(len(beat)))
        # claimed keys above a verified key: decisions taken within a solver tolerance
        claims = []
        for m in ("enum", "scip"):
            b = d[d.method == m].set_index(["id", "model"])
            if "claimed_key" not in b.columns:
                continue
            for i in lv.index.intersection(b.index):
                if cmp_keys(key_of(lv.loc[i, "key"]), key_of(b.loc[i, "claimed_key"])) == -1:
                    claims.append({"method": m, "id": i[0], "model": int(i[1]), "status": b.loc[i, "status"],
                                   "evaluated_beats": cmp_keys(key_of(lv.loc[i, "key"]), key_of(b.loc[i, "key"])) == -1})
        summary[design]["claims_above_verified"] = claims
        tex(f"{tag}ClaimAboveVerified", count(len(claims)))
        tex(f"{tag}ClaimAboveVerifiedAttained", count(sum(1 for c in claims if c["evaluated_beats"])))
        res_prio = [float(x["residual_priority"]) for x in beat if finite(x.get("residual_priority"))]
        if res_prio:
            tex(f"{tag}BeatVerifiedPriorityResid", sci(max(res_prio)))

    # 10. peak memory of single tasks, each in a fresh process (protocol v1, section 7)
    mem_path = res / "memory.jsonl"
    if exists(mem_path):
        mem = pd.DataFrame([json.loads(line) for line in read_lines(mem_path) if line.strip()])
        tex("MemErrors", count((mem.status == "ERROR").sum()))
        mem = mem[mem.status != "ERROR"]
        summary["memory"] = {}
        for (m, n), g in mem.groupby(["method", "n"]):
            nm = f"Mem{CAMEL[m]}N{NWORD[int(n)]}"
            tex(nm + "Median", num(g.peak_mib.median(), 0))
            tex(nm + "Max", num(g.peak_mib.max(), 0))
            summary["memory"][f"{m}|{n}"] = {"tasks": len(g), "median_mib": float(g.peak_mib.median()),
                                             "max_mib": float(g.peak_mib.max())}

    tex.write(out / "numbers_benchmark.tex")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
