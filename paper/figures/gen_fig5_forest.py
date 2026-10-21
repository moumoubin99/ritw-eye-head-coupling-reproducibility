"""Fig 5 — forest plot of the wrist-minus-world contrast b - a: primary (registered H-T1), the 11
registered sensitivities, and POST HOC checks (visually separated). Rows with < 10 clusters are hollow,
point estimate only (no interval, no p). Sources: results/m3t/m3t_stats.json,
results/posthoc/m3t_decomposition_checks.json."""
import json
from paper_plot_style import *

st = json.load(open(path("results", "m3t", "m3t_stats.json")))
ph = json.load(open(path("results", "posthoc", "m3t_decomposition_checks.json")))
S = st["sensitivities_primary"]
REG = [("rigid", "Rigid wrist predictor"), ("depth", "Vergence depth"),
       ("thr30", r"Fast-phase threshold 30 deg s$^{-1}$"), ("thr60", r"Fast-phase threshold 60 deg s$^{-1}$"),
       ("other_hand", "Other (non-primary) hand"), ("mid", "Left–right wrist midpoint"),
       ("J_rho_le_0.9", r"Collinearity $|\rho| \leq 0.9$"), ("cov_ge_0.7", r"Hand coverage $\geq 0.7$"),
       ("pid_date_clusters", "Date-only pseudo-participant clusters"), ("personalized_only", "Personalised calibration only"),
       ("mask_calibrated_1.8Hz", "Mask-calibrated (1.8 Hz)")]
assert len(REG) == 11 and set(k for k, _ in REG) == set(S)
PH = [(ph["tests"]["d_ht3"], "Head-translation regressor added"),
      (ph["tests"]["d_rig3"], r"Rigid split ($b_{\mathrm{tr}} - a$)"),
      (ph["tests"]["d_k0"], r"$k = 0$ linear bridge"),
      (ph["tests"]["d_near"], "Near-world target"),
      (ph["inclusion"]["all_finite"], "All finite task-12 fits"),
      (ph["inclusion"]["all_finite_excluding_not_converged"], "All finite, excluding not converged")]


def norm(r):
    p = r.get("p", r.get("p_signflip"))
    ncl = r["n_clusters"]
    kind = r.get("p_kind", "signflip_exact" if ncl <= 16 else "signflip_mc")
    return dict(mean=r["mean"], ci=r.get("ci95"), p=p, nrec=r["n_rec"], ncl=ncl, kind=kind)


rows = [("head", "Registered primary (H-T1)"), ("row", "Primary", norm(st["T1_primary_b_minus_a"]))]
rows += [("head", "Registered sensitivities")] + [("row", lab, norm(S[k])) for k, lab in REG]
rows += [("head", "Post hoc (in-sample)")] + [("row", lab, norm(r)) for r, lab in PH]

fig, ax = plt.subplots(figsize=(DOUBLE, 4.1))
n = len(rows)
for i, r in enumerate(rows):
    yy = n - i
    if r[0] == "head":
        ax.text(-0.62, yy, r[1], fontsize=7, fontweight="bold", va="center", ha="left")
        if i:
            ax.axhline(yy + 0.5, color=C["grey"], lw=0.4)
        continue
    _, lab, v = r
    few = v["ncl"] < 10
    post = r in rows[rows.index(("head", "Post hoc (in-sample)")):]
    col = C["purple"] if post else (C["black"] if lab == "Primary" else C["diff"])
    if few:
        ax.plot(v["mean"], yy, "o", ms=4, mfc="white", mec=col, mew=0.9)
        txt = f"{v['mean']:+.2f}".replace("-", "−") + "   (descriptive, < 10 clusters)"
    else:
        ax.errorbar(v["mean"], yy, xerr=[[v["mean"] - v["ci"][0]], [v["ci"][1] - v["mean"]]], fmt="o", ms=3.5,
                    color=col, capsize=1.5, lw=0.9)
        p = v["p"]
        ps = (lambda m, e: f"{m} × 10$^{{{int(e)}}}$")(*f"{p:.1e}".split("e")) if p < 0.001 else f"{p:.2g}"
        txt = f"{v['mean']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]".replace("-", "−") + f"   p = {ps}{' (MC)' if v['kind'] == 'signflip_mc' else ''}"
    ax.text(-0.58, yy, lab, fontsize=6.5, va="center", ha="left")
    ax.text(1.02, yy, f"{v['nrec']}/{v['ncl']}", fontsize=6.5, va="center", ha="left")
    ax.text(1.17, yy, txt, fontsize=6.5, va="center", ha="left")
ax.axvline(0, color=C["grey"], lw=0.6, ls="--")
prim = st["T1_primary_b_minus_a"]["mean"]
ax.axvline(prim, color=C["black"], lw=0.4, ls=":")
ax.set_xlim(-0.62, 1.0)
ax.set_ylim(0.4, n + 0.8)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel(r"Wrist-proxy minus world-rotation coefficient, $b - a$ (cluster mean, 95% percentile cluster bootstrap)")
ax.text(1.02, n + 0.75, "rec/cl", fontsize=6.5, fontweight="bold", va="center")
ax.text(1.17, n + 0.75, "mean [95% CI]   sign-flip p", fontsize=6.5, fontweight="bold", va="center")
# label region for row names lies left of the data area; draw x-axis only under the data range
ax.spines["bottom"].set_bounds(-0.2, 0.8)
save(fig, "fig5_forest")
