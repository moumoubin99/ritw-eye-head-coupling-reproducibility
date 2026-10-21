"""Fig 2 — M3: unmasked gait-band gain (kappa_raw) and slow-phase gain (kappa_sp) by condition, and
fast-phase fraction vs kappa_raw. Sources: results/m3_measures_main.csv, results/m3_stats.json."""
import json
import pandas as pd
from paper_plot_style import *

d = pd.read_csv(path("results", "m3_measures_main.csv"))
d = d[d.gait_ok.astype(str).str.lower() == "true"]
st = json.load(open(path("results", "m3_stats.json")))
desc = st["H_S5_descriptive"]

fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.1), gridspec_kw=dict(width_ratios=[1, 1, 1.35], wspace=0.45))
for ax, col, lab, pl in [(axs[0], "gait_kappa_raw", r"Unmasked gait-band gain $\kappa_{\mathrm{raw}}$", "a"),
                         (axs[1], "gait_kappa_sp", r"Slow-phase gain $\kappa_{\mathrm{sp}}$", "b")]:
    for i, task in enumerate((11, 12)):
        s = d[d.task == task]
        ax.scatter(i + jitter(len(s), seed=task), s[col], s=7, color=C["read" if task == 12 else "walk"],
                   alpha=0.55, lw=0, zorder=2)
        m, ci, ncl, nrec = cluster_mean_ci(s[col], s.pid)
        ref = desc[col][f"task{task}"]
        assert abs(m - ref["mean"]) < 1e-12 and ncl == ref["n_clusters"], (col, task, m, ref)
        ax.errorbar(i + 0.3, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="o", ms=4, color="k", mfc="white",
                    capsize=2, lw=0.9, zorder=3)
        ax.text(i + 0.15, -0.88, f"{nrec} rec / {ncl} cl", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color=C["grey"], lw=0.5, ls=":")
    ax.axhline(1, color=C["grey"], lw=0.5, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Not\nreading", "Reading"])
    ax.set_xlim(-0.5, 1.6); ax.set_ylim(-0.92, 1.3)
    ax.set_ylabel(lab)
    panel_label(ax, pl, x=-0.32)

ax = axs[2]
for task, mk in ((11, "o"), (12, "^")):
    s = d[d.task == task]
    ax.scatter(s.gait_fast_frac, s.gait_kappa_raw, s=10, marker=mk, color=C["read" if task == 12 else "walk"],
               alpha=0.75, lw=0, label=f"{COND[task]} ({len(s)} rec)")
ax.axhline(0, color=C["grey"], lw=0.5, ls=":")
ax.set_xlabel("Fast-phase fraction of gait-band support\n" r"$|F\cap S|/|S|$")
ax.set_ylabel(r"$\kappa_{\mathrm{raw}}$")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1, handletextpad=0.2, borderaxespad=0.1)
panel_label(ax, "c", x=-0.22)
save(fig, "fig2_m3_gains")
