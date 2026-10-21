"""Supplementary Fig. S2 -- per-recording H-T1 contrast b - a (task 12, H-T1 recordings) by reading medium
and by gaze calibration. Descriptive only: no CI or p. Groups with < 5 recordings are drawn hollow and carry no
summary mark. Sources: results/m3t/m3t_measures_main.csv, results/m3t/m3t_stats.json,
data/ritw_meta/reading_in_the_wild_seattle/ritw_annotations.csv (reading medium)."""
import json
import pandas as pd
from paper_plot_style import *

d = pd.read_csv(path("results", "m3t", "m3t_measures_main.csv"))
d = d[(d.task == 12) & (d.J_ok.astype(str).str.lower() == "true")].copy()
ann = pd.read_csv(path("data", "ritw_meta", "reading_in_the_wild_seattle", "ritw_annotations.csv"))
d = d.merge(ann[["id", "medium", "calibration_success"]], left_on="rid", right_on="id", how="left")
assert d.medium.notna().all() and ((d.gaze_kind == "personalized") == d.calibration_success.astype(bool)).all()
d["dba"] = d.J_b_hand - d.J_a_world
t1 = json.load(open(path("results", "m3t", "m3t_stats.json")))["T1_primary_b_minus_a"]
cm = d.groupby("pid").dba.mean().mean()
assert len(d) == t1["n_rec"] and d.pid.nunique() == t1["n_clusters"] and abs(cm - t1["mean"]) < 1e-12

MIN_N = 5
fig, axs = plt.subplots(1, 2, figsize=(SINGLE * 1.25, 1.9), sharey=True,
                        gridspec_kw=dict(width_ratios=[3, 2], wspace=0.12))
for ax, col, groups, labels, pl, col_c in [
        (axs[0], "medium", ["Object", "Digital", "Print"], ["Object", "Digital", "Print"], "a", C["read"]),
        (axs[1], "gaze_kind", ["personalized", "general"], ["Personalised", "General"], "b", C["purple"])]:
    for i, g in enumerate(groups):
        s = d[d[col] == g]
        big = len(s) >= MIN_N
        ax.scatter(i + jitter(len(s), w=0.13, seed=i + 7), s.dba, s=12, lw=0.8,
                   facecolor=col_c if big else "white", edgecolor=col_c, zorder=2)
        if big:
            ax.plot([i - 0.25, i + 0.25], [s.dba.median()] * 2, color="k", lw=1.0, zorder=3)
        ax.text(i, -0.6, f"{len(s)} rec\n{s.pid.nunique()} cl", ha="center", va="bottom", fontsize=6)
    ax.axhline(0, color=C["grey"], lw=0.5, ls=":")
    ax.axhline(t1["mean"], color=C["diff"], lw=0.7, ls="--", zorder=1)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("Reading medium" if col == "medium" else "Gaze calibration")
    panel_label(ax, pl, x=-0.22 if pl == "a" else -0.12)
axs[0].set_ylim(-0.62, 0.85)
axs[0].set_ylabel(r"$b-a$ (per recording)")
axs[1].text(1.62, t1["mean"], "H-T1\nestimate", color=C["diff"], fontsize=6, va="center", ha="left", clip_on=False)
save(fig, "suppS2_ba_by_medium_calibration")
print("rec", len(d), "range", d.dba.min().round(3), d.dba.max().round(3),
      d.groupby("medium").dba.median().round(3).to_dict(), d.groupby("gaze_kind").dba.median().round(3).to_dict())
