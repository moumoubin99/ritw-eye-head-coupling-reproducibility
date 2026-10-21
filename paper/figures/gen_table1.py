"""Table 1: (a) cohort, reading medium, calibration and walking behaviour; (b) M3 pre-registered endpoints.

Reads only the evidence inventory:
  results/m3_cohort_frozen.json, results/m3_stats.json, results/m3_measures_main.csv,
  results/m3t/m3t_stats.json, results/m3t/m3t_measures_main.csv,
  data/ritw_meta/reading_in_the_wild_seattle/ritw_annotations.csv (reading medium).
Writes paper/figures/TABLE_1.tex.
"""
import json
import os

import numpy as np
import pandas as pd

from paper_plot_style import path, FIG_DIR

coh = json.load(open(path("results", "m3_cohort_frozen.json")))
m3 = json.load(open(path("results", "m3_stats.json")))
m3t = json.load(open(path("results", "m3t", "m3t_stats.json")))
mm = pd.read_csv(path("results", "m3_measures_main.csv"))
tm = pd.read_csv(path("results", "m3t", "m3t_measures_main.csv"))
ann = pd.read_csv(path("data", "ritw_meta", "reading_in_the_wild_seattle", "ritw_annotations.csv"))


def tf(s):
    return s.astype(str).str.lower() == "true"


def num(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return "$" + s + "$"


def ci(c, d=2):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]".replace("-", "$-$")


def pfmt(p, kind):
    lab = {"ols_cluster_t": "cluster $t$", "signflip_exact": "exact sign-flip"}[kind]
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$ ({lab})"
    return f"{p:.2g} ({lab})"


# ---------------------------------------------------------------- cohort counts
cnt = {(c["task"], c["status"]): c["n"] for c in coh["counts"]}
n_all = {k: cnt[(k, "included")] + cnt[(k, "excluded_walk")] for k in (11, 12)}
n_walk = {k: cnt[(k, "included")] for k in (11, 12)}
g2 = m3["gate2"]
assert sum(g2["status_by_task"]["task11"].values()) == n_walk[11] == m3t["T6_task11_descriptive"]["n_loaded"]
assert sum(g2["status_by_task"]["task12"].values()) == n_walk[12] == m3t["cohort"]["task12_loaded"]
c = m3t["cohort"]
# the 14 M3-T recordings are a subset of the 16 M3-eligible task-12 recordings
J = tm[(tm.task == 12) & tf(tm.J_ok)]
m3ok12 = set(mm[(mm.task == 12) & tf(mm.gait_ok)].rid)
assert len(J) == c["task12_included"] and set(J.rid) <= m3ok12 and J.pid.nunique() == c["task12_clusters"]
assert len(m3ok12) == g2["n_R_ok"]

# ---------------------------------------------------------------- medium and calibration (task 12)
L = tm[(tm.task == 12) & (tm.load_status == "loaded")].merge(
    ann[["id", "medium", "calibration_success"]], left_on="rid", right_on="id", how="left")
L["J"] = tf(L.J_ok)
assert L.medium.notna().all() and len(L) == n_walk[12]
# gaze kind used by the pipeline agrees with the RitW calibration flag
assert ((L.gaze_kind == "personalized") == L.calibration_success.astype(bool)).all()
med = {m: (int(L[L.medium == m].J.sum()), int((L.medium == m).sum())) for m in ("Object", "Digital", "Print")}
cal = {k: (int(L[L.gaze_kind == k].J.sum()), int((L.gaze_kind == k).sum())) for k in ("personalized", "general")}
side = c["primary_side_task12"]

# ---------------------------------------------------------------- behaviour medians (M3-eligible)
E = mm[tf(mm.gait_ok)]
beh = [("Walking speed (m\\,s$^{-1}$)", "speed", 2),
       ("Step frequency (Hz)", "step_freq", 2),
       ("Gaze pitch (deg)", "gaze_pitch", 0),
       ("Fast-phase fraction of gait-band support", "gait_fast_frac", 2),
       ("Gait-band head-motion power $P$", "gait_P", 3)]


def med_iqr(task, col, d):
    v = E[E.task == task][col]
    q1, q2, q3 = v.quantile([.25, .5, .75])
    return f"{num(q2, d)} ({num(q1, d)}, {num(q3, d)})"


# ---------------------------------------------------------------- Table 1a
A = []
A.append(r"\multicolumn{3}{l}{\textit{Cohort (recordings; date$\times$age pseudo-participant clusters)}}\\")
A.append(f"RitW Seattle recordings & {n_all[11]} & {n_all[12]}\\\\")
A.append(f"Walking $\\geq$ 30\\,s, loaded & {n_walk[11]} & {n_walk[12]}\\\\")
A.append(f"M3 estimator-eligible (gait band) & {g2['n_N_ok']} rec / {g2['clusters_N_ok']} cl"
         f" & {g2['n_R_ok']} rec / {g2['clusters_R_ok']} cl\\\\")
t6 = m3t["T6_task11_descriptive"]
m3t_main = pd.read_csv(path("results", "m3t", "m3t_measures_main.csv"))
n_cov11 = int(((m3t_main.task == 11) & (m3t_main.cov_primary >= 0.5)).sum())
assert t6["cov_ge_0.5"]["n"] == 1 and n_cov11 == 2 and t6["n_loaded"] == (m3t_main.task == 11).sum()
h3, hp = m3["H_S3_diagnostic"], m3["H_P"]["effect"]
A.append(f"\\quad pooled across tasks & \\multicolumn{{2}}{{l}}{{{h3['n_rec']} rec / {h3['n_clusters']} unique cl "
         f"({hp['n_clusters_both']} cl in both tasks)}}\\\\")
A.append(f"M3-T joint-fit eligible (H-T1 set) & not testable$^{{\\dagger}}$"
         f" & {c['task12_included']} rec / {c['task12_clusters']} cl\\\\")
A.append(r"\addlinespace")
A.append(r"\multicolumn{3}{l}{\textit{Task 12 composition: M3-T eligible / loaded}}\\")
for m in ("Object", "Digital", "Print"):
    A.append(f"\\quad Reading medium: {m} & --- & {med[m][0]}/{med[m][1]}\\\\")
A.append(f"\\quad Gaze calibration: personalised & --- & {cal['personalized'][0]}/{cal['personalized'][1]}\\\\")
A.append(f"\\quad Gaze calibration: general & --- & {cal['general'][0]}/{cal['general'][1]}\\\\")
A.append(f"\\quad Primary (more visible) hand, right / left & --- & {side['right']} right, {side['left']} left (of {side['right'] + side['left']})\\\\")
A.append(r"\addlinespace")
A.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{Walking behaviour, M3-eligible recordings: median (IQR); "
         f"n = {g2['n_N_ok']} / {g2['n_R_ok']}}}}}\\\\")
for lab, col, d in beh:
    A.append(f"\\quad {lab} & {med_iqr(11, col, d)} & {med_iqr(12, col, d)}\\\\")

# ---------------------------------------------------------------- Table 1b
s5 = m3["H_S5_descriptive"]


def contrast_row(name, status, pred, key_or_dict, verdict):
    e = m3[key_or_dict] if isinstance(key_or_dict, str) else key_or_dict
    return (f"{name} & {status} & {pred} & {num(e['delta'], 2, True)} {ci(e['ci95'])} & {pfmt(e['p'], e['p_kind'])}"
            f" & {e['n_clusters_N']}/{e['n_clusters_R']} & {verdict}\\\\")


B = []
B.append(contrast_row("H-P: $\\Delta c$", "primary", "$<0$", m3["H_P"]["effect"],
                      "refuted (opposite direction)"))
B.append(contrast_row("H-S1: $\\Delta\\kappa_{\\mathrm{sp}}$", "secondary", "$\\leq 0$", "H_S1",
                      "direction as predicted; not equivalent ($\\pm$0.10)"))
h2 = m3["H_S2"]
B.append(f"H-S2: $\\kappa_{{\\mathrm{{sp}}}}$, walking not reading & secondary & CI low $>0.5$ & "
         f"{num(h2['mean'])} {ci(h2['ci95'])} & --- & {h2['n_clusters']}/--- & pass\\\\")
h3 = m3["H_S3_diagnostic"]
B.append(f"H-S3: $c$ on fast-phase fraction (+ condition) & diagnostic & coef.\\ $<0$ & "
         f"{num(h3['gait_fast_frac']['coef'], 2, True)} {ci(h3['gait_fast_frac']['ci95'])} & "
         f"{pfmt(h3['gait_fast_frac']['p'], 'ols_cluster_t')} & {h3['n_rec']} rec / {h3['n_clusters']} cl"
         f" & as expected (partly constructive)\\\\")
B.append(f"\\quad condition term & & & {num(h3['cond']['coef'], 2, True)} {ci(h3['cond']['ci95'])} & "
         f"{pfmt(h3['cond']['p'], 'ols_cluster_t')} & & \\\\")
B.append(contrast_row("H-S4: $\\Delta\\kappa_{\\mathrm{raw}}$", "secondary",
                      "$<0$, $|\\Delta\\kappa_{\\mathrm{raw}}|>|\\Delta\\kappa_{\\mathrm{sp}}|$", "H_S4",
                      "opposite on both"))
ab = m3["H_S4"]["abs_raw_minus_abs_sp"]
B.append(f"\\quad $|\\Delta\\kappa_{{\\mathrm{{raw}}}}|-|\\Delta\\kappa_{{\\mathrm{{sp}}}}|$ & & & "
         f"{num(ab['est'], 2, True)} {ci(ab['ci95'])} & --- & {ab['n_clusters']} cl & \\\\")
B.append(contrast_row("H-S6: $\\Delta c_{\\mathrm{common}}$", "secondary", "$<0$", "H_S6_c_common",
                      "opposite direction"))
B.append(r"\addlinespace")
B.append(r"\multicolumn{7}{l}{\textit{H-S5 (descriptive): cluster mean (SD), walking not reading vs walking while reading}}\\")
for lab, key in (("gait band $\\kappa_{\\mathrm{sp}}$", "gait_kappa_sp"),
                 ("gait band $\\kappa_{\\mathrm{raw}}$ (unmasked)", "gait_kappa_raw"),
                 ("gait band $c$", "gait_c")):
    a11, a12 = s5[key]["task11"], s5[key]["task12"]
    B.append(f"\\quad {lab} & descriptive & --- & {num(a11['mean'])} ({a11['sd']:.2f}) vs "
             f"{num(a12['mean'])} ({a12['sd']:.2f}) & --- & {a11['n_clusters']}/{a12['n_clusters']} & ---\\\\")
a11, a12 = s5["slow_kappa_sp"]["task11"], s5["slow_kappa_sp"]["task12"]
assert a11["n_clusters"] < 10 and a12["n_clusters"] < 10
B.append(f"\\quad slow band $\\kappa_{{\\mathrm{{sp}}}}$ & descriptive & --- & {num(a11['mean'])} vs {num(a12['mean'])}"
         f" & --- & {a11['n_clusters']}/{a12['n_clusters']} & point only ($<$10 clusters)\\\\")

# H-P / H-S1 mean cross-checks against the descriptive block
assert abs(m3["H_P"]["effect"]["mean_N"] - s5["gait_c"]["task11"]["mean"]) < 1e-12
assert abs(m3["H_S1"]["mean_R"] - s5["gait_kappa_sp"]["task12"]["mean"]) < 1e-12

tex = r"""%% generated by paper/figures/gen_table1.py -- do not edit by hand
\begin{table}[p]
\centering
\footnotesize
\caption{\textbf{Cohort, walking behaviour and the M3 pre-registered endpoints.}
\textbf{(a)} Cohort flow, task-12 composition and walking behaviour. Clusters are date$\times$age pseudo-participant clusters (cl); rec, recordings. $^{\dagger}$""" + f"{n_cov11} of {t6['n_loaded']} loaded task-11 recordings have primary-hand coverage $\\geq$ 0.5 and only {t6['cov_ge_0.5']['n']} also passes the estimator-eligibility checks" + r""", so no hand-referenced endpoint is testable in non-reading walking (descriptive only, Supplementary Note 1). Behaviour medians are recording-level medians over M3-eligible recordings.
\textbf{(b)} M3 endpoints in the gait band (0.8--3\,Hz). $\Delta$ = walking while reading (task 12) minus walking not reading (task 11). Estimates are differences of cluster means (all-clusters design) with percentile cluster-bootstrap 95\% CIs (5{,}000 resamples); $p$ from cluster-robust OLS on cluster-condition means with $t(G-1)$. H-S3 is a cluster-robust OLS regression of recording-level $c$ on fast-phase fraction and condition (41 recordings, 30 clusters), with model-based 95\% CIs. Secondary endpoints are uncorrected. Cluster counts are task 11 / task 12. $\kappa_{\mathrm{raw}}$ is the unmasked gait-band gain; $c=\kappa_{\mathrm{raw}}-\kappa_{\mathrm{sp}}$. Rows with fewer than 10 clusters are descriptive point estimates without CI or $p$.}
\label{tab:cohort_m3}
\textbf{a}\par\smallskip
\begin{tabular}{lll}
\toprule
 & Walking, not reading (task 11) & Walking while reading (task 12)\\
\midrule
""" + "\n".join(A) + r"""
\bottomrule
\end{tabular}

\medskip
\textbf{b}\par\smallskip
{\setlength{\tabcolsep}{3pt}%
\resizebox{\textwidth}{!}{\begin{tabular}{>{\raggedright\arraybackslash}p{3.0cm}l>{\raggedright\arraybackslash}p{2.2cm}ll>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}p{3.2cm}}
\toprule
Endpoint & Status & Prediction & Estimate [95\% CI] & $p$ (kind) & Clusters (11/12) & Verdict\\
\midrule
""" + "\n".join(B) + r"""
\bottomrule
\end{tabular}}}
\end{table}
"""
out = os.path.join(FIG_DIR, "TABLE_1.tex")
open(out, "w").write(tex)
print("Saved:", out)
print("medium", med, "cal", cal, "side", side)
for lab, col, d in beh:
    print(col, E.groupby("task")[col].median().round(3).to_dict())
