"""Fig 4 — POST HOC decomposition of the registered H-T1 contrast (task 12, J_ok, in-sample).
(a) FWL identity kappa_W = a + b*gamma and kappa_H = b + omitted-variable term, as cluster means of
recording-level terms; (b) standardised coefficients beta_W, beta_H; (c) unique Delta R^2 of each predictor.
Sources: results/posthoc/m3t_decomposition_checks.{csv,json}."""
import json
import pandas as pd
from paper_plot_style import *

d = pd.read_csv(path("results", "posthoc", "m3t_decomposition_checks.csv"))
J = d[d.J_ok.astype(str).str.lower() == "true"].copy()
js = json.load(open(path("results", "posthoc", "m3t_decomposition_checks.json")))
cmj = js["cluster_means"]
J["b_gamma"] = J.b * J.gamma
assert (J.kW_Ssp - J.a - J.b_gamma).abs().max() < 1e-12 and (J.kH_SspW - J.b - J.ovb_H).abs().max() < 1e-12
cm = lambda c: float(J.groupby("pid")[c].mean().mean())
for c in ("a", "b", "kW_Ssp", "kH_SspW", "ovb_H", "beta_W", "beta_H", "dR2_W", "dR2_H", "ratio_WH"):
    assert abs(cm(c) - cmj[c]) < 1e-12, c
nrec, ncl = len(J), J.pid.nunique()

fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.25), gridspec_kw=dict(width_ratios=[1.5, 1, 1], wspace=0.45))
ax = axs[0]
bars = [(0, [("a", cm("a"), C["world"], r"$a$ (world, joint fit)"),
             ("bg", cm("b_gamma"), C["wrist"], r"$b\gamma$ (wrist proxy $\times$ projection)")]),
        (1, [("kW", cm("kW_Ssp"), C["grey"], None)]),
        (2.5, [("b", cm("b"), C["wrist"], None),
               ("ovb", cm("ovb_H"), C["sky"], r"omitted-variable term of $\kappa_H$")]),
        (3.5, [("kH", cm("kH_SspW"), C["grey"], r"single-predictor fit (same samples)")])]
for x, stack in bars:
    y0 = 0
    for key, v, col, lab in stack:
        ax.bar(x, v, bottom=y0, width=0.7, color=col, edgecolor="k", lw=0.4, label=lab,
               hatch="////" if key in ("bg", "ovb") else None)
        ax.text(x, y0 + v / 2, f"{v:.3f}", ha="center", va="center", fontsize=6,
                color="k", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        y0 += v
ax.set_xticks([0, 1, 2.5, 3.5])
ax.set_xticklabels([r"$a + b\gamma$", r"$\kappa_W$", r"$b$ + omitted", r"$\kappa_H$"])
ax.set_ylabel("Coefficient or slope (cluster mean)")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper left", bbox_to_anchor=(-0.05, -0.2), ncol=2, fontsize=6, handlelength=1.2, columnspacing=0.8)
ax.text(0.02, 0.97, f"RMS($x_W$)/RMS($x_H$) = {cm('ratio_WH'):.2f}\n(cluster mean of recording ratios)", transform=ax.transAxes, ha="left", va="top",
        fontsize=6)
panel_label(ax, "a", x=-0.2)


def paired(ax, cw, ch, test, ylab, pl):
    for _, q in J.iterrows():
        ax.plot([0, 1], [q[cw], q[ch]], color=C["grey"], lw=0.5, alpha=0.7)
    ax.scatter(np.zeros(nrec), J[cw], s=10, color=C["world"], lw=0, zorder=2)
    ax.scatter(np.ones(nrec), J[ch], s=10, color=C["wrist"], lw=0, zorder=2)
    for x, c in ((-0.25, cw), (1.25, ch)):
        m, ci, _, _ = cluster_mean_ci(J[c], J.pid)
        ax.errorbar(x, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="o", ms=4, color="k", mfc="white", capsize=2, lw=0.8)
    t = js["tests"][test]
    ax.text(0.5, 1.0, (f"wrist − world: {t['mean']:+.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]".replace("-", "−") + "\n"
                      f"exact sign-flip p = {t['p_signflip']:.2f}; {nrec} rec / {ncl} cl"),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["World\nrotation", "Wrist\nproxy"]); ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(ylab)
    panel_label(ax, pl, x=-0.3, y=1.12)


paired(axs[1], "beta_W", "beta_H", "d_beta", r"Standardised coefficient $\beta$", "b")
paired(axs[2], "dR2_W", "dR2_H", "d_dR2", r"Unique $\Delta R^2$ (uncentred)", "c")
for a_ in axs[1:]:
    lo, hi = a_.get_ylim(); a_.set_ylim(min(0, lo), hi + 0.08 * (hi - min(0, lo)))
save(fig, "fig4_decomposition")
