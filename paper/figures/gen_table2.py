"""Table 2: M3-T registered endpoints (H-T1-H-T4) and the 11 registered sensitivities (upper block);
POST HOC in-sample decomposition and inclusion checks (visually separated lower block).
Rows with < 10 clusters are point estimates only. Incremental R^2 (+0.075) carries no p.

Reads only: results/m3t/m3t_stats.json, regenerated main/rigid recording-level CSVs,
results/posthoc/m3t_decomposition_checks.{json,csv}. Writes paper/figures/TABLE_2.tex.
"""
import json
import os

import numpy as np
import pandas as pd

from paper_plot_style import path, FIG_DIR

st = json.load(open(path("results", "m3t", "m3t_stats.json")))
ph = json.load(open(path("results", "posthoc", "m3t_decomposition_checks.json")))
S = st["sensitivities_primary"]
T = ph["tests"]
CM = ph["cluster_means"]


def num(x, d=2, sign=True):
    return "$" + (f"{x:+.{d}f}" if sign else f"{x:.{d}f}") + "$"


def ci(c, d=2):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]".replace("-", "$-$")


def pfmt(p, ncl):
    kind = "exact" if ncl <= 16 else "MC"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$ ({kind})"
    return f"{p:.2g} ({kind})"


def tested(lab, r, note="", d=2, p=True):
    pp = r.get("p", r.get("p_signflip"))
    ps = pfmt(pp, r["n_clusters"]) if p else "---"
    return f"{lab} & {r['n_rec']}/{r['n_clusters']} & {num(r['mean'], d)} & {ci(r['ci95'], d)} & {ps} & {note}\\\\"


def point(lab, nrec, ncl, val, note, d=2, sign=True):
    return f"{lab} & {nrec}/{ncl} & {num(val, d, sign)} & --- & --- & {note}\\\\"


def tf(s):
    return s.astype(str).str.lower() == "true"


# rigid predictor lowers b - a in every recording (B3): stated in the admitted results summary (D2)
base = pd.read_csv(path("results", "m3t", "m3t_measures_main.csv"))
rigid = pd.read_csv(path("results", "m3t", "m3t_measures_rigid.csv"))
base = base[(base.task == 12) & tf(base.J_ok)].set_index("rid")
rigid = rigid.set_index("rid").reindex(base.index)
n_rig_lower = int(((rigid.J_b_hand - rigid.J_a_world) < (base.J_b_hand - base.J_a_world)).sum())
n_t1 = st["T1_primary_b_minus_a"]["n_rec"]
assert n_rig_lower == n_t1 == S["rigid"]["n_rec"]

# ---------------------------------------------------------------- upper block (registered)
t1, t5 = st["T1_primary_b_minus_a"], st["T5_descriptive_task12"]
assert t1["p_kind"] == "signflip_exact" and abs(t1["mean"] - T["d_ba"]["mean"]) < 1e-12
U = [r"\multicolumn{6}{l}{\textit{Registered endpoints (frozen plan; analysis-frozen follow-up on recordings previously analysed in M3)}}\\",
     tested("H-T1 (primary): $b-a$", t1, "supported")]
for key, lab in (("J_a_world", "\\quad world-rotation coefficient $a$"), ("J_b_hand", "\\quad wrist-proxy coefficient $b$")):
    v = t5[key]
    U.append(f"{lab} & {v['n_rec']}/{v['n_clusters']} & {num(v['cluster_mean'], 2, False)} (SD {v['cluster_sd']:.2f}) & --- & --- & descriptive\\\\")
U.append(tested("H-T2 (secondary): $b-b_{\\mathrm{shift}}$", st["T2_surrogate_b"], "consistent (uncorrected)"))
k = st["T2_surrogate_kappaH_descriptive"]
U.append(point("H-T2: $\\kappa_H-\\kappa_{H,\\mathrm{shift}}$", k["n_rec"], k["n_clusters"], k["cluster_mean"],
               "descriptive only", sign=True))
U.append(tested("H-T3 (secondary): $D_1=|1-\\kappa_W|-|1-\\kappa_H|$", st["T3_D1_closer_to_one"], "consistent (uncorrected)"))
t4 = st["T4_kappaW_vs_lambda"]
assert t4["consistent_pm0.15"] is False
U.append(f"H-T4 (secondary): $\\kappa_W-\\lambda$ within $\\pm$0.15 & {t4['n_rec']}/{t4['n_clusters']} & {num(t4['mean'])}"
         f" & 90\\% CI {ci(t4['ci90'])} & --- & criterion not met\\\\")
U.append(r"\addlinespace")
U.append(r"\multicolumn{6}{l}{\textit{Registered sensitivities of H-T1 ($b-a$)}}\\")
REG = [("rigid", "Rigid wrist predictor"), ("depth", "Vergence depth"),
       ("thr30", "Fast-phase threshold 30\\,deg\\,s$^{-1}$"), ("thr60", "Fast-phase threshold 60\\,deg\\,s$^{-1}$"),
       ("other_hand", "Other (non-primary) hand"), ("mid", "Left--right wrist midpoint"),
       ("J_rho_le_0.9", "Collinearity $|\\rho|\\leq 0.9$"), ("cov_ge_0.7", "Hand coverage $\\geq$ 0.7"),
       ("pid_date_clusters", "Date-only pseudo-participant clusters"), ("personalized_only", "Personalised calibration only"),
       ("mask_calibrated_1.8Hz", "Mask-calibrated (1.8\\,Hz)")]
assert len(REG) == 11 and set(k for k, _ in REG) == set(S)
for key, lab in REG:
    r = S[key]
    if r["n_clusters"] < 10:
        assert r.get("ci95") is None and r.get("p") is None
        U.append(point("\\quad " + lab, r["n_rec"], r["n_clusters"], r["mean"], "descriptive ($<$10 clusters)"))
    else:
        assert r["p_kind"] == "signflip_exact"
        note = f"lower than primary in {n_rig_lower}/{n_t1} rec" if key == "rigid" else ""
        U.append(tested("\\quad " + lab, r, note))

# ---------------------------------------------------------------- lower block (post hoc)
L = [r"\multicolumn{6}{l}{\textit{Post hoc, in-sample decomposition and inclusion checks (not registered)}}\\",
     tested("World-rotation coefficient $a>0$", T["a"], f"{T['a']['n_rec_positive']}/{T['a']['n_rec']} rec $>0$")]
# FWL identities, cluster means of recording-level terms
D = pd.read_csv(path("results", "posthoc", "m3t_decomposition_checks.csv"))
D = D[tf(D.J_ok)].assign(b_gamma=lambda x: x.b * x.gamma)
assert len(D) == 14 and D.pid.nunique() == 13
assert (D.kW_Ssp - D.a - D.b_gamma).abs().max() < 1e-12 and (D.kH_SspW - D.b - D.ovb_H).abs().max() < 1e-12
cmean = lambda c: float(D.groupby("pid")[c].mean().mean())
for c in ("a", "b", "kW_Ssp", "kH_SspW", "ovb_H", "gamma"):
    assert abs(cmean(c) - CM[c]) < 1e-12, c
bg = cmean("b_gamma")
L.append(f"\\quad $\\kappa_W=a+b\\gamma$ on $S_{{\\mathrm{{sp}},W}}$ & {len(D)}/{D.pid.nunique()} & {num(CM['kW_Ssp'], 3, False)} = "
         f"{num(CM['a'], 3, False)} + {num(bg, 3, False)} & --- & --- & descriptive; mean $\\gamma$ = {CM['gamma']:.3f}\\\\")
L.append(f"\\quad $\\kappa_H=b+$ omitted-variable term on $S_{{\\mathrm{{sp}},W}}$ & {len(D)}/{D.pid.nunique()} & {num(CM['kH_SspW'], 3, False)} = "
         f"{num(CM['b'], 3, False)} + {num(CM['ovb_H'], 3, False)} & --- & --- & descriptive; cluster mean of recording RMS ratios $x_W/x_H$ = {CM['ratio_WH']:.2f}\\\\")
L.append(tested("Standardised coefficients $\\beta_H-\\beta_W$", T["d_beta"],
                f"{CM['beta_H']:.2f} vs {CM['beta_W']:.2f}; similar, equivalence not established"))
L.append(tested("Unique $\\Delta R^2_H-\\Delta R^2_W$", T["d_dR2"],
                f"{CM['dR2_H']:.3f} vs {CM['dR2_W']:.3f}; similar, equivalence not established", d=3))
L.append(tested("Head-translation regressor added ($b-a$)", T["d_ht3"], ""))
L.append(tested("Rigid split $b_{\\mathrm{tr}}-a$", T["d_rig3"], f"rotation lever-arm term $c$ = {CM['rig3_c_rot']:.2f}"))
L.append(tested("$k=0$ linear bridge ($b-a$)", T["d_k0"], ""))
L.append(tested("Near-world target ($b-a$)", T["d_near"], ""))
inc = ph["inclusion"]
assert inc["all_finite"]["n_clusters"] > 16 and inc["all_finite_excluding_not_converged"]["n_clusters"] > 16
n_nc = inc["all_finite"]["n_rec"] - inc["all_finite_excluding_not_converged"]["n_rec"]
assert inc["all_finite"]["n_rec"] - inc["all_finite_excluding_not_converged"]["n_rec"] == n_nc
L.append(tested("All finite task-12 fits ($b-a$)", inc["all_finite"], f"includes {n_nc} not converged"))
L.append(tested("All finite, excluding not converged ($b-a$)", inc["all_finite_excluding_not_converged"], ""))
r2 = T["d_R2_hand_over_W_ht"]
L.append(point("$R^2$ gain from adding wrist proxy to world + head translation", r2["n_rec"], r2["n_clusters"],
               r2["mean"], f"descriptive ($\\geq 0$ by construction); {r2['n_rec_positive']}/{r2['n_rec']} rec $>0$", d=3))

tex = r"""%% generated by paper/figures/gen_table2.py -- do not edit by hand
\begin{table}[p]
\centering
\footnotesize
\caption{\textbf{Wrist-proxy versus world-rotation coefficients during walking while reading (M3-T, task 12).}
Upper block: registered endpoints and the 11 registered sensitivities of H-T1. Lower block (separated by the double rule): post hoc, in-sample analyses that were not registered and are descriptive decompositions or robustness checks of H-T1. $b$, coefficient on the wrist-translation proxy $x_H$; $a$, coefficient on the world-rotation predictor $x_W$; both are unstandardised gait-band (0.8--3\,Hz) slow-phase coefficients from the frozen joint fit and are not stabilisation fractions. Estimates are means of date$\times$age pseudo-participant cluster means; n is recordings/clusters. 95\% CIs are percentile cluster-bootstrap intervals (10{,}000 resamples); $p$ values are two-sided cluster sign-flip tests, exact for $\leq$16 clusters and Monte Carlo (MC, 20{,}000 flips) otherwise. Clusters are date$\times$age pseudo-participant clusters except in the date-only clustering sensitivity. Secondary endpoints are uncorrected. H-T4 is decided by its registered rule (90\% CI inside $\pm$0.15). Rows with fewer than 10 clusters, $\kappa_H-\kappa_{H,\mathrm{shift}}$ and the $R^2$ gain carry no CI or $p$. The wrist-proxy slope on the joint support $S_{\mathrm{sp},W}$ (0.878, 14/13) differs from the standalone $\kappa_H$ on $S_{\mathrm{sp},H}$ (""" + f"{st['T5_descriptive_task12']['H_kappa_sp']['cluster_mean']:.2f}, {st['T5_descriptive_task12']['H_kappa_sp']['n_rec']}/{st['T5_descriptive_task12']['H_kappa_sp']['n_clusters']}" + r"""); neither is a measure of tracking accuracy.}
\label{tab:m3t}
{\setlength{\tabcolsep}{3pt}%
\resizebox{\textwidth}{!}{\begin{tabular}{>{\raggedright\arraybackslash}p{5.6cm}llll>{\raggedright\arraybackslash}p{4.2cm}}
\toprule
Quantity & n & Estimate & 95\% CI & $p$ (sign-flip) & Note\\
\midrule
""" + "\n".join(U) + r"""
\midrule\midrule
""" + "\n".join(L) + r"""
\bottomrule
\end{tabular}}}
\end{table}
"""
out = os.path.join(FIG_DIR, "TABLE_2.tex")
open(out, "w").write(tex)
print("Saved:", out, "| rigid lower:", n_rig_lower, "/", n_t1)
