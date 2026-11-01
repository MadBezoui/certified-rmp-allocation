"""Illustrative figures computed from the two-recipient example (not experiment results).

Figure 1  conflict among the three objectives along the priority-optimal segment
Figure 2  reference profiles in objective space and the attained coalitions
Figure 3  lexicographic coalition search and its certificate for a two-criterion version

Every curve is evaluated from the objective definitions, and Figure 3 plots the decisions,
halfspaces and witness returned by the solver itself.
Usage: python figures_illustrative.py <out_dir>
"""
import json
import math
import pathlib
import sys
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

from certrmp.instance import Instance, SINGLE, priority_totals
from certrmp.objectives import Objective
from certrmp.model import RMPModel
from certrmp.oracle import Oracle
from certrmp.search import lcs, refine
from certrmp import certify, verify

BLUE, RED, BLACK = (0, 0, 1), (1, 0, 0), (0, 0, 0)
GRAY = (0, 0, 0, 0.15)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0})
WIDTH = 6.3

A_GRID = np.linspace(1.0, 3.0, 2001)
NAMED = {"previous (1.5, 1.5)": 1.5, "equal fractions (2, 1)": 2.0, "least shortfall (2.5, 0.5)": 2.5}
MARKERS = {"previous (1.5, 1.5)": "s", "equal fractions (2, 1)": "o", "least shortfall (2.5, 0.5)": "^"}
PROFILES = np.array([[5.2, 1.2, 0.02], [6.0, 1.8, 0.1]])
WEIGHTS = (0.4, 0.2, 0.4)


def f_example(a):
    """Unscaled example objectives at x = (a, 3 - a)."""
    a = np.asarray(a, dtype=float)
    f1 = (4 - a) ** 2 + (2 - (3 - a)) ** 2
    f2 = np.abs(a - 1.5) + np.abs(3 - a - 1.5)
    f3 = (a / 4 - (3 - a) / 2) ** 2
    return np.vstack([f1, f2, f3])


def example_instance():
    return Instance(d=np.array([4.0, 2.0]), x_prev=np.array([1.5, 1.5]), tiers=[np.array([0, 1])],
                    A=np.ones((1, 2)), c=np.array([3.0]), structure=SINGLE)


def example_objectives(which=(0, 1, 2)):
    objs = [Objective("shortfall", "shortfall", weights=np.ones(2), scale=1.0),
            Objective("movement", "movement", weights=np.ones(2), scale=1.0),
            # gamma/|J| sum (s_i - mean)^2 = (s1 - s2)^2 when gamma = 4
            Objective("disparity", "disparity", groups=[np.array([0, 1])], gammas=np.array([4.0]), scale=1.0)]
    return [objs[j] for j in which]


def figure1(out):
    F = f_example(A_GRID)
    labels = [r"$f_1$ squared shortfall", r"$f_2$ absolute movement", r"$f_3$ squared fraction gap"]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.6), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(A_GRID, F[j], color=BLUE)
        for name, a in NAMED.items():
            ax.plot([a], f_example([a])[j], marker=MARKERS[name], color=BLACK, markersize=4, linestyle="none",
                    label=name if j == 0 else None)
        ax.set_ylabel(labels[j], rotation=0, ha="left", va="center", labelpad=0)
        ax.yaxis.set_label_coords(1.01, 0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.35))
    axes[-1].set_xlabel(r"allocation $x = (a,\ 3 - a)$, value of $a$")
    fig.subplots_adjust(right=0.78, hspace=0.25, top=0.9)
    fig.savefig(out / "fig1_example_conflict.pdf")
    plt.close(fig)


def coalition_intervals(F, b):
    return F <= b[:, None]


def figure2(out):
    F = f_example(A_GRID)
    model = RMPModel.srmp(PROFILES, [0, 1], WEIGHTS)
    keys = [model.key(F[:, k]) for k in range(F.shape[1])]
    best = max(keys)
    opt = np.array([k == best for k in keys])
    fig = plt.figure(figsize=(WIDTH, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.45, wspace=0.3)
    for col, (i, j) in enumerate(((0, 1), (0, 2))):
        ax = fig.add_subplot(gs[0, col])
        for h, (b, ls, mk) in enumerate(zip(PROFILES, ("-", "--"), ("D", "P"))):
            ax.fill_between([0, b[i]], 0, b[j], color=GRAY, linewidth=0)
            ax.plot([0, b[i], b[i]], [b[j], b[j], 0], color=BLACK, linestyle=ls, linewidth=0.7)
            ax.plot([b[i]], [b[j]], marker=mk, color=BLACK, markersize=4, linestyle="none",
                    label=f"profile $b_{h + 1}$")
        ax.plot(F[i], F[j], color=BLUE, label=r"$f(X)$")
        ax.plot(F[i][opt], F[j][opt], color=BLUE, linewidth=3.0, alpha=0.5, label="best class")
        for name, a in NAMED.items():
            v = f_example([a])[:, 0]
            ax.plot([v[i]], [v[j]], marker=MARKERS[name], color=BLACK, markersize=4, linestyle="none")
        ax.set_xlabel(f"$f_{i + 1}$")
        ax.set_ylabel(f"$f_{j + 1}$")
        ax.set_xlim(4.3, 8.2)
        ax.set_ylim(0, 2.1 if j == 1 else 0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.legend(loc="upper right", frameon=False)
    ax = fig.add_subplot(gs[1, :])
    rows = []
    for h in range(2):
        att = coalition_intervals(F, PROFILES[h])
        for j in range(3):
            rows.append((f"$b_{h + 1}$: $f_{j + 1}$", att[j]))
    for r, (lab, mask) in enumerate(rows):
        y = len(rows) - r
        ax.fill_between(A_GRID, y - 0.35, y + 0.35, where=mask, color=GRAY, linewidth=0, step="mid")
        ax.text(0.985, y, lab, ha="right", va="center", fontsize=7)
    ax.fill_between(A_GRID, -0.45, 0.25, where=opt, color=BLUE, alpha=0.5, linewidth=0, step="mid")
    ax.text(0.985, -0.1, "best key", ha="right", va="center", fontsize=7)
    for name, a in NAMED.items():
        ax.axvline(a, color=BLACK, linestyle=":", linewidth=0.6)
        ax.plot([a], [len(rows) + 0.8], marker=MARKERS[name], color=BLACK, markersize=4, clip_on=False)
    ax.set_xlim(0.75, 3.0)
    ax.set_ylim(-0.6, len(rows) + 1)
    ax.set_yticks([])
    ax.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel(r"value of $a$; gray: criterion attained at the profile")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(out / "fig2_profiles_coalitions.pdf")
    plt.close(fig)
    lo, hi = A_GRID[opt].min(), A_GRID[opt].max()
    return {"best_key": [str(v) for v in best], "best_class": [float(lo), float(hi)]}


def search_example():
    """Two criteria (shortfall, fraction gap), profiles consulted in order, SRMP weights 0.6, 0.4."""
    inst = example_instance()
    objs = example_objectives((0, 2))
    sigma, _ = priority_totals(inst)
    model = RMPModel.srmp([[4.7, 0.005], [5.2, 0.05]], [0, 1], [0.6, 0.4])
    oracle = Oracle(inst, objs, sigma)
    res = lcs(oracle, model)
    ref = refine(oracle, res.frontier)
    cert = certify.build(inst, objs, model, sigma, res, ref, True)
    out = verify.verify(json.loads(json.dumps(cert)))
    return inst, objs, model, res, ref, cert, out


def figure3(out):
    inst, objs, model, res, ref, cert, ver = search_example()
    F = f_example(A_GRID)[[0, 2]]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    # upper image Y = f(X) + R^2_+ restricted to the window
    xs = np.r_[F[0], 9.0, 9.0]
    ys = np.r_[F[1], F[1].max(), F[1].min()]
    order = np.argsort(F[0])
    env_x, env_y = [], []
    best = math.inf
    for k in order[::-1]:
        best = min(best, F[1][k])
    # lower-left boundary of Y: for each f1 level, the smallest f3 among allocations with f1 below it
    grid = np.linspace(F[0].min(), 7.2, 600)
    bnd = np.array([F[1][F[0] <= g].min() for g in grid])
    ax.fill_between(grid, bnd, 0.32, color=BLUE, alpha=0.12, linewidth=0, label=r"attainable $Y$")
    ax.plot(F[0], F[1], color=BLUE, label=r"$f(X)$")
    for h in range(model.m):
        b = model.profiles[h]
        ax.axvline(b[0], color=BLACK, linewidth=0.5, linestyle="-" if h == 0 else "--")
        ax.axhline(b[1], color=BLACK, linewidth=0.5, linestyle="-" if h == 0 else "--")
        ax.text(b[0] + 0.03, 0.2 - 0.03 * h, f"$b_{{{h + 1},1}}$", fontsize=7, va="top")
        ax.text(7.18, b[1] + 0.004, f"$b_{{{h + 1},2}}$", fontsize=7, ha="right", va="bottom")
    s = res.searcher
    shown = set()
    for lvl in res.levels:
        for (w, v, d) in lvl["decisions"]:
            wx = 7.2 if math.isinf(w[0]) else w[0]
            wy = 0.29 if math.isinf(w[1]) else w[1]
            if math.isinf(w[1]):
                ax.annotate(r"$u_2=\infty$", (wx, wy), xytext=(6, -2), textcoords="offset points", fontsize=7,
                            va="center")
            if d.feasible:
                lab = None if "feasible" in shown else "attainable requirement"
                shown.add("feasible")
                ax.plot([wx], [wy], marker="o", markerfacecolor="none", color=BLUE, markersize=6,
                        linestyle="none", label=lab)
            else:
                lab = None if "excluded" in shown else "excluded requirement"
                shown.add("excluded")
                ax.plot([wx], [wy], marker="x", color=RED, markersize=6, linestyle="none", label=lab)
                hs = s.caches.halfspaces[d.ref]
                mu, L = hs.mu, hs.L
                gx = np.linspace(4.3, 7.2, 50)
                if mu[1] > 0:
                    gy = (L - mu[0] * gx) / mu[1]
                    ax.fill_between(gx, 0, np.clip(gy, 0, 0.32), color=RED, alpha=0.12, linewidth=0)
                    ax.plot(gx, gy, color=RED, linestyle="-.", linewidth=0.8,
                            label="separating halfspace")
    fw = ref["f"]
    ax.plot([fw[0]], [fw[1]], marker="*", color=BLACK, markersize=8, linestyle="none", label="witness")
    ax.set_xlim(4.3, 7.2)
    ax.set_ylim(0, 0.31)
    ax.set_xlabel(r"$f_1$ squared shortfall (requirement $u_1$)")
    ax.set_ylabel(r"$f_3$ squared fraction gap ($u_2$)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1.02, 0.5))
    fig.subplots_adjust(left=0.1, right=0.66, bottom=0.15, top=0.97)
    fig.savefig(out / "fig3_search_certificate.pdf")
    plt.close(fig)
    return {"key": [str(v) for v in res.key], "calls": res.oracle_calls, "counts": res.counts,
            "verified": bool(ver["verified"] and ver["exact"]), "witness_f": [float(v) for v in fw],
            "halfspaces": len(cert["halfspaces"]),
            "decisions": [[["inf" if math.isinf(x) else x for x in w], str(v), d.feasible, d.how]
                          for lvl in res.levels for (w, v, d) in lvl["decisions"]]}


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    figure1(out)
    info2 = figure2(out)
    info3 = figure3(out)
    (out / "illustrative_values.json").write_text(json.dumps({"figure2": info2, "figure3": info3}, indent=1))
    print(json.dumps({"figure2": info2, "figure3": info3}, indent=1))


if __name__ == "__main__":
    main()
