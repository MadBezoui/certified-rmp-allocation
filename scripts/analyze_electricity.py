"""Analysis of the demand-trace study (repeated decisions on UCI electricity load diagrams).

Reads trajectories_ramp<tag>.jsonl and profiles_ramp<tag>.json, writes per-method tables
(CSV and JSON), Figure 7 (PDF) and LaTeX number macros. Nothing is computed by hand.
Usage: python analyze_electricity.py <results_dir> <out_dir>
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

BLUE, RED, BLACK, GRAY = (0, 0, 1), (1, 0, 0), (0, 0, 0), (0.5, 0.5, 0.5)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "serif",
                     "mathtext.fontset": "cm", "axes.linewidth": 0.6, "lines.linewidth": 1.0})
WIDTH = 6.3
METHODS = ("rmp_cold", "rmp_reuse", "rmp_warm", "myopic", "feedback", "smoothing", "absolute", "robd")
LABEL = {"rmp_cold": "RMP, cold", "rmp_reuse": "RMP, reuse", "rmp_warm": "RMP, warm start",
         "myopic": "myopic", "feedback": "projected feedback", "smoothing": "quadratic smoothing",
         "absolute": "absolute movement", "robd": "R-OBD"}
MARK = {"rmp_cold": (BLUE, "o"), "rmp_reuse": (BLUE, "s"), "rmp_warm": (BLUE, "^"), "myopic": (BLACK, "x"),
        "feedback": (RED, "D"), "smoothing": (RED, "v"), "absolute": (GRAY, "P"), "robd": (GRAY, "*")}
BOOT = 2000
RNG_SEED = 20261030


def key_of(v):
    return tuple(Fraction(x) for x in v)


def cmp(a, b):
    return (a > b) - (a < b)


def read_lines(path):
    """Lines of a JSON-lines file, read from its gzip-compressed copy when only that exists."""
    path = pathlib.Path(path)
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        return gzip.decompress(path.with_name(path.name + ".gz").read_bytes()).decode().splitlines()
    return path.read_text().splitlines()


def load(path):
    rows = [json.loads(line) for line in read_lines(path) if line.strip()]
    df = pd.DataFrame(rows)
    df[["f1", "f2", "f3"]] = pd.DataFrame(df.pop("f").tolist(), index=df.index)
    df["key_t"] = df["key"].apply(key_of)
    return df


def block_boot(values_by_block, n=BOOT, seed=RNG_SEED):
    """Percentile interval of the mean of per-block values, resampling blocks."""
    v = np.asarray(values_by_block, dtype=float)
    rng = np.random.default_rng(seed)
    means = [v[rng.integers(len(v), size=len(v))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def switches(g):
    k = g.sort_values("t")["key_t"].tolist()
    return sum(1 for a, b in zip(k, k[1:]) if a != b)


def method_table(df):
    rows = []
    for m in METHODS:
        d = df[df.method == m]
        if d.empty:
            continue
        per_block = d.groupby("block")
        r = {"method": m, "blocks": d.block.nunique(), "epochs": len(d),
             "shortfall": d.f1.mean(), "movement": d.f2.mean(), "disparity": d.f3.mean(),
             "movement_p95": d.f2.quantile(0.95),
             "tier1_service": d.served_tier1.sum() / d.demand_tier1.sum(),
             "emergencies": int(d.emergency.sum()),
             "key_switches_per_block": float(np.mean([switches(g) for _, g in per_block])),
             "seconds_mean": d.seconds.mean(), "seconds_p95": d.seconds.quantile(0.95)}
        lo, hi = block_boot(per_block.f2.mean().values)
        r.update(movement_ci_low=lo, movement_ci_high=hi)
        if "calls" in d and d.calls.notna().any():
            r["calls_mean"] = d.calls.mean()
            lo, hi = block_boot(per_block.calls.mean().values)
            r.update(calls_ci_low=lo, calls_ci_high=hi)
        if "verified" in d and d.verified.notna().any():
            r["verified_share"] = float(d.verified.fillna(False).astype(bool).mean())
        if "separated_by_reused" in d and d.separated_by_reused.notna().any():
            r["separated_by_reused_mean"] = d.separated_by_reused.mean()
            r["carried_mean"] = d.carried.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def paired_keys(df, ref="rmp_cold"):
    """Realized keys of ref against each method at the same (block, epoch). The states differ
    after the first epoch, so this compares realized trajectories, not decisions at a common state."""
    base = df[df.method == ref].set_index(["block", "t"])["key_t"]
    rows = []
    for m in METHODS:
        if m == ref or not (df.method == m).any():
            continue
        other = df[df.method == m].set_index(["block", "t"])["key_t"]
        common = base.index.intersection(other.index)
        c = np.array([cmp(base[k], other[k]) for k in common])
        rows.append({"method": m, "epochs": len(common), "ref_better": int((c > 0).sum()),
                     "tie": int((c == 0).sum()), "ref_worse": int((c < 0).sum())})
    return pd.DataFrame(rows)


def figure(df, out, block=None):
    """One test block under the tight ramp limit: the supply event, movement, the RMP key and
    the work saved by carried evidence, epoch by epoch."""
    block = int(df.block.min()) if block is None else block
    d = {m: df[(df.method == m) & (df.block == block)].sort_values("t") for m in METHODS}
    cold, warm = d["rmp_cold"], d["rmp_warm"]
    if cold.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 7.0), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1.1, 1.1, 0.9, 0.9]})
    above = dict(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize=6.5)
    ax = axes[0]
    ax.step(cold.t, cold.capacity, where="post", color=BLACK, linewidth=0.8, label="total capacity")
    ax.plot(cold.t, cold.demand_tier1, color=GRAY, linewidth=0.8, label="critical-tier request")
    ax.plot(cold.t, cold.served_tier1, color=BLUE, linewidth=0.8, linestyle="--", label="critical-tier allocation")
    ax.set_ylabel("units")
    ax.legend(**above)
    ax = axes[1]
    for m, ls in (("myopic", ":"), ("feedback", "--"), ("rmp_cold", "-")):
        if not d[m].empty:
            ax.plot(d[m].t, d[m].f2, color=MARK[m][0], linestyle=ls, linewidth=0.8, label=LABEL[m])
    ax.set_ylabel("movement $f_2$")
    ax.legend(**above)
    ax = axes[2]
    keys = np.array([[float(v) for v in k] for k in cold.key_t])
    for j, (ls, lab) in enumerate(zip(("-", "--", ":"), ("first consulted profile", "second", "third"))):
        ax.step(cold.t, keys[:, j] + 0.03 * (1 - j), where="post", color=BLUE, linestyle=ls, linewidth=0.8, label=lab)
    ax.set_ylabel("RMP key")
    ax.set_ylim(-0.08, 1.12)
    ax.legend(**above)
    ax = axes[3]
    ax.step(cold.t, cold.calls, where="post", color=BLACK, linewidth=0.8, label="calls, cold")
    ax.step(warm.t, warm.calls, where="post", color=BLUE, linewidth=0.8, label="calls, warm start")
    ax.step(warm.t, warm.separated_by_reused, where="post", color=RED, linestyle="--", linewidth=0.8,
            label="candidates excluded by carried halfspaces")
    ax.set_ylabel("count")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("epoch (15 minutes)")
    ax.legend(**above)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        e = cold[cold.emergency]
        for t in e.t:
            ax.axvspan(t - 0.5, t + 0.5, color=(0, 0, 0, 0.08), linewidth=0)
    fig.savefig(out / "fig7_demand_traces.pdf")
    plt.close(fig)


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main():
    res = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    paths = [res / f"trajectories_ramp{tag}.jsonl" for tag in ("0.5", "2")]
    missing = [str(path) for path in paths if not (path.is_file() or path.with_name(path.name + ".gz").is_file())]
    if missing:
        raise SystemExit("Missing required trajectory files: " + ", ".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    summary, dfs, tex, table_rows = {}, [], [], []
    for tag, filled, prefix in (("0.5", True, "ElecTight"), ("2", False, "ElecLoose")):
        path = res / f"trajectories_ramp{tag}.jsonl"
        if not (path.exists() or path.with_name(path.name + ".gz").exists()):
            continue
        df = load(path)
        dfs.append((tag, df, filled))
        tab = method_table(df)
        tab.to_csv(out / f"electricity_ramp{tag}.csv", index=False)
        pk = paired_keys(df)
        pk.to_csv(out / f"electricity_keys_ramp{tag}.csv", index=False)
        # agreement of the three exact controllers
        piv = df[df.method.str.startswith("rmp")].pivot_table(index=["block", "t"], columns="method",
                                                              values="f2", aggfunc="first")
        same, same_f = {}, {}
        for m in ("rmp_reuse", "rmp_warm"):
            if m in piv and "rmp_cold" in piv:
                kk = df[df.method.isin([m, "rmp_cold"])].pivot_table(index=["block", "t"], columns="method",
                                                                     values="key_t", aggfunc="first")
                kk = kk.dropna()
                same[m] = float(np.mean([a == b for a, b in zip(kk[m], kk["rmp_cold"])]))
                # the same decision also means the same objective vector, compared exactly
                ff = df[df.method.isin([m, "rmp_cold"])].set_index(["block", "t", "method"])[["f1", "f2", "f3"]]
                ff = ff.unstack("method").dropna()
                same_f[m] = float(np.mean(np.all([ff[(c, m)].values == ff[(c, "rmp_cold")].values
                                                  for c in ("f1", "f2", "f3")], axis=0)))
        t = tab.set_index("method")
        summary[tag] = {"table": tab.to_dict(orient="records"), "keys": pk.to_dict(orient="records"),
                        "key_agreement_with_cold": same}
        if {"rmp_cold", "rmp_reuse", "rmp_warm"} <= set(t.index):
            cold, reuse, warm = (t.loc[m, "calls_mean"] for m in ("rmp_cold", "rmp_reuse", "rmp_warm"))
            tex += [macro(prefix + "CallsCold", f"{cold:.2f}"), macro(prefix + "CallsReuse", f"{reuse:.2f}"),
                    macro(prefix + "CallsWarm", f"{warm:.2f}"),
                    macro(prefix + "WarmSaving", f"{100 * (1 - warm / cold):.0f}"),
                    macro(prefix + "ReuseSaving", f"{100 * (1 - reuse / cold):.0f}"),
                    macro(prefix + "Verified", f"{100 * t.loc['rmp_warm', 'verified_share']:.1f}")]
        tex.append(macro(prefix + "Epochs", f"{int(t.loc[t.index[0], 'epochs']):,}".replace(",", "{,}")))
        tex.append(macro(prefix + "Blocks", str(int(t.loc[t.index[0], "blocks"]))))
        word = {"rmp_cold": "Rmp", "myopic": "Myopic", "feedback": "Feedback", "smoothing": "Smoothing",
                "absolute": "Absolute", "robd": "Robd"}
        for m, w in word.items():
            if m not in t.index:
                continue
            tex += [macro(f"{prefix}Shortfall{w}", f"{t.loc[m, 'shortfall']:.3f}"),
                    macro(f"{prefix}Movement{w}", f"{t.loc[m, 'movement']:.3f}"),
                    macro(f"{prefix}Disparity{w}", f"{t.loc[m, 'disparity']:.3f}"),
                    macro(f"{prefix}Emerg{w}", str(int(t.loc[m, "emergencies"]))),
                    macro(f"{prefix}Switch{w}", f"{t.loc[m, 'key_switches_per_block']:.0f}"),
                    macro(f"{prefix}Ms{w}", f"{1000 * t.loc[m, 'seconds_mean']:.0f}")]
        rules = [m for m in ("myopic", "feedback", "smoothing", "absolute", "robd") if m in t.index]
        if rules and "rmp_cold" in t.index:
            tex += [macro(prefix + "EmergRuleMin", str(int(t.loc[rules, "emergencies"].min()))),
                    macro(prefix + "EmergRuleMax", str(int(t.loc[rules, "emergencies"].max()))),
                    macro(prefix + "MovementRuleMin", f"{t.loc[rules, 'movement'].min():.3f}"),
                    macro(prefix + "ShortfallRuleMax", f"{t.loc[rules, 'shortfall'].max():.3f}"),
                    macro(prefix + "ShortfallRuleMin", f"{t.loc[rules, 'shortfall'].min():.3f}"),
                    macro(prefix + "MovementRuleMax", f"{t.loc[rules, 'movement'].max():.3f}"),
                    macro(prefix + "DisparityRuleMin", f"{t.loc[rules, 'disparity'].min():.3f}"),
                    macro(prefix + "DisparityRuleMax", f"{t.loc[rules, 'disparity'].max():.3f}"),
                    macro(prefix + "SwitchRuleMin", f"{t.loc[rules, 'key_switches_per_block'].min():.0f}"),
                    macro(prefix + "SwitchRuleMax", f"{t.loc[rules, 'key_switches_per_block'].max():.0f}"),
                    macro(prefix + "MsRuleMax", f"{1000 * t.loc[rules, 'seconds_mean'].max():.0f}"),
                    macro(prefix + "TierOneMin", f"{t['tier1_service'].min():.3f}"),
                    macro(prefix + "TierOneMax", f"{t['tier1_service'].max():.3f}")]
        exact = [m for m in ("rmp_cold", "rmp_reuse", "rmp_warm") if m in t.index]
        if exact:
            tex.append(macro(prefix + "MsExactMax", f"{1000 * t.loc[exact, 'seconds_mean'].max():.0f}"))
        for m, w in (("rmp_reuse", "Reuse"), ("rmp_warm", "Warm")):
            if m in t.index:
                tex += [macro(prefix + "Separated" + w, f"{t.loc[m, 'separated_by_reused_mean']:.3f}"),
                        macro(prefix + "Carried" + w, f"{t.loc[m, 'carried_mean']:.1f}")]
        pr = pk[pk.method.isin(rules)]
        if len(pr):
            tex += [macro(prefix + "KeyBetterRuleMin", f"{100 * (pr.ref_better / pr.epochs).min():.0f}"),
                    macro(prefix + "KeyBetterRuleMax", f"{100 * (pr.ref_better / pr.epochs).max():.0f}"),
                    macro(prefix + "KeyTieRuleMin", f"{100 * (pr.tie / pr.epochs).min():.0f}"),
                    macro(prefix + "KeyTieRuleMax", f"{100 * (pr.tie / pr.epochs).max():.0f}"),
                    macro(prefix + "KeyWorseRuleMax", f"{100 * (pr.ref_worse / pr.epochs).max():.1f}")]
        for _, r in pk.iterrows():
            if r.method in word:
                w = word[r.method]
                tex += [macro(f"{prefix}KeyBetter{w}", f"{100 * r.ref_better / r.epochs:.0f}"),
                        macro(f"{prefix}KeyTie{w}", f"{100 * r.tie / r.epochs:.0f}"),
                        macro(f"{prefix}KeyWorse{w}", f"{100 * r.ref_worse / r.epochs:.1f}")]
        for m in ("rmp_reuse", "rmp_warm"):
            if m in same:
                tex.append(macro(prefix + "Agree" + ("Reuse" if m == "rmp_reuse" else "Warm"), f"{100 * same[m]:.1f}"))
                tex.append(macro(prefix + "SameObjective" + ("Reuse" if m == "rmp_reuse" else "Warm"),
                                 f"{100 * same_f[m]:.1f}"))
        rows = []
        for m in METHODS:
            if m not in t.index:
                continue
            r = t.loc[m]
            calls = f"{r.calls_mean:.2f}" if "calls_mean" in r and pd.notna(r.get("calls_mean")) else "--"
            rows.append(f"{LABEL[m]} & {r.shortfall:.3f} & {r.movement:.3f} & {r.disparity:.3f} & "
                        f"{r.key_switches_per_block:.0f} & {int(r.emergencies)} & {calls} & {1000 * r.seconds_mean:.0f} \\\\")
        table_rows.append((tag, rows))
    tight = [df for tag, df, _ in dfs if tag == "0.5"]
    if tight:
        figure(tight[0], out)
    (out / "electricity_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "numbers_electricity.tex").write_text("% Generated by scripts/analyze_electricity.py. Do not edit.\n" + "".join(tex))
    lines = []
    for k, (tag, rows) in enumerate(table_rows):
        if k:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{8}}{{l}}{{Ramp limit {tag} per client and epoch}} \\\\")
        lines += rows
    (out / "tab_electricity.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
