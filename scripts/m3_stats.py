"""M3 statistics (EXPERIMENT_PLAN_M3.md). Reads results/m3_measures_<cfg>.csv and the frozen cohort;
writes results/m3_stats.json. Clusters = date_age pid (sensitivity: pid_date). R = task 12, N = task 11.
Design = the stricter of gate 1 (cohort, trajectories only) and gate 2 (estimability flags of the
gaze-based estimator; gaze-dependent by construction, so its counts and failure reasons are reported).
paired: cluster-mean differences, sign-flip test (exact for <= 16 clusters; assumes differences are
symmetric about 0 under H0 and clusters independent), cluster bootstrap CI.
all_clusters: every cluster with >= 1 condition; delta of condition means of cluster means; bootstrap
over the union of clusters; p from OLS mean ~ cond on cluster-condition means with cluster-robust t.
unsupported: descriptives only (no tests)."""
import argparse, json, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default="results")
ap.add_argument("--cohort", default="results/m3_cohort_frozen.json")
ap.add_argument("--B", type=int, default=5000)
ap.add_argument("--n_perm", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/m3_stats.json")
a = ap.parse_args()
R, N = 12, 11
EQ = 0.10            # TOST equivalence bound for delta kappa_sp (90% bootstrap CI inside +-EQ)
START_TOL = 0.02     # k = 1 vs k = 0 start disagreement flag
MIN_CLU = 10         # minimum clusters for any inferential test
ORDER = ["unsupported", "all_clusters", "paired"]
G1_MAP = {"paired": "paired", "unpaired": "all_clusters", "unsupported": "unsupported"}
REQUIRED = ["rid", "task", "pid", "pid_date", "activity", "gaze_kind", "load_status", "gait_ok", "gait_status",
            "gait_fit_status", "gait_start_used", "gait_start1_status", "gait_start0_status", "gait_kappa_sp_start1", "gait_kappa_sp", "gait_kappa_sp_start0", "gait_kappa_sp_first",
            "gait_kappa_raw", "gait_c", "gait_c_common", "gait_kappa_sp_S", "gait_kappa_raw_S", "gait_fast_frac",
            "gait_f_ep_rate", "gait_P", "gait_n_Ssp_s", "gait_n_Sraw_s", "gait_kappa_sp_cx", "gait_kappa_sp_cy",
            "gait_c_cx", "gait_c_cy", "gait_kappa_sp_rot_given_trans", "gait_trans_coef", "slow_ok", "slow_status",
            "slow_kappa_sp", "slow_kappa_raw", "slow_c", "speed", "gaze_pitch", "step_freq", "invalid_frac_walk"]


def truthy(x):
    return x.astype(str).str.lower() == "true"


def load(cfg):
    p = os.path.join(a.indir, f"m3_measures_{cfg}.csv")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:   # headerless file: treat as an empty cohort
        df = pd.DataFrame()
    df = df.reindex(columns=list(dict.fromkeys(REQUIRED + list(df.columns))))
    for c in [c for c in df.columns if c.endswith(("_status", "_kind")) or c in ("pid", "pid_date", "activity")]:
        df[c] = df[c].astype(object)   # string columns stay strings even in an empty table
    df["ok"] = truthy(df.gait_ok)   # gait_ok requires a converged fixed point and the minimum durations
    return df


def gate2(df, clu="pid"):
    ok = df[df.ok]
    pr = set(ok[ok.task == R][clu]) & set(ok[ok.task == N][clu])
    nR = int((ok.task == R).sum())
    cR, cN = ok[ok.task == R][clu].nunique(), ok[ok.task == N][clu].nunique()
    design = "paired" if (len(pr) >= MIN_CLU and nR >= 15) else ("all_clusters" if min(cR, cN) >= MIN_CLU else "unsupported")
    reasons = {f"task{k}": v.gait_status.fillna("missing").value_counts().to_dict() for k, v in df.groupby("task")}
    return {"paired_clusters_ok": len(pr), "n_R_ok": nR, "n_N_ok": int((ok.task == N).sum()),
            "clusters_R_ok": int(cR), "clusters_N_ok": int(cN), "n_rec_ok": int(len(ok)), "n_rec": int(len(df)),
            "status_by_task": reasons, "design": design}


def cmeans(df, col, clu):
    d = df[df.ok & np.isfinite(pd.to_numeric(df[col], errors="coerce"))]
    return d.groupby([clu, "task"])[col].mean().unstack()


def signflip(d, rng):
    d = np.asarray(d, float)
    n, obs = len(d), d.mean()
    if n <= 16:
        S = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        return float(np.mean(np.abs(S @ d / n) >= abs(obs) - 1e-12)), "signflip_exact"
    S = rng.choice([-1, 1], size=(a.n_perm, n))
    return float((np.sum(np.abs(S @ d / n) >= abs(obs) - 1e-12) + 1) / (a.n_perm + 1)), f"signflip_random{a.n_perm}"


def pct(x, lvl):
    x = np.asarray(x)[np.isfinite(x)]
    q = (1 - lvl) / 2
    return [float(np.quantile(x, q)), float(np.quantile(x, 1 - q))]


def descr(df, col, clu="pid"):
    cm = cmeans(df, col, clu)
    out = {}
    for c in (R, N):
        x = cm[c].dropna() if c in cm else pd.Series(dtype=float)
        out[f"task{c}"] = {"n_clusters": int(len(x)), "mean": float(x.mean()) if len(x) else None,
                           "sd": float(x.std()) if len(x) > 1 else None}
    return {"design": "descriptive", **out}


def paired(df, col, clu="pid"):
    cm = cmeans(df, col, clu)
    if R not in cm or N not in cm:
        return {"design": "paired", "n_clusters": 0}
    d = (cm[R] - cm[N]).dropna()
    if len(d) < MIN_CLU:
        return {"design": "paired", "n_clusters": int(len(d)), "note": f"< {MIN_CLU} paired clusters"}
    rng = np.random.default_rng(a.seed)
    p, kind = signflip(d.values, rng)
    bm = d.values[rng.integers(0, len(d), (a.B, len(d)))].mean(1)
    return {"design": "paired", "n_clusters": int(len(d)), "delta": float(d.mean()), "p": p, "p_kind": kind,
            "ci95": pct(bm, .95), "ci90": pct(bm, .90), "mean_R": float(cm.loc[d.index, R].mean()),
            "mean_N": float(cm.loc[d.index, N].mean())}


def boot_union(cm, rng, f):
    """Cluster bootstrap over the union of clusters; f(cm_resampled) -> statistic. Clusters keep both
    conditions they have."""
    ids = cm.index.values
    out = np.empty(a.B)
    for b in range(a.B):
        out[b] = f(cm.loc[rng.choice(ids, len(ids))])
    return out


def allclusters(df, col, clu="pid"):
    cm = cmeans(df, col, clu)
    if R not in cm or N not in cm:
        return {"design": "all_clusters", "n_clusters_R": 0, "n_clusters_N": 0}
    cR, cN = int(cm[R].notna().sum()), int(cm[N].notna().sum())
    if min(cR, cN) < MIN_CLU:
        return {"design": "all_clusters", "n_clusters_R": cR, "n_clusters_N": cN, "note": f"< {MIN_CLU} clusters"}
    dl = lambda m: m[R].mean() - m[N].mean()
    rng = np.random.default_rng(a.seed)
    bm = boot_union(cm[[R, N]], rng, dl)
    long = cm[[R, N]].stack().dropna().rename("y").reset_index()
    long["cond"] = (long.task == R).astype(int)
    m = smf.ols("y ~ cond", long).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(long[clu])[0]}, use_t=True)
    return {"design": "all_clusters", "n_clusters_R": cR, "n_clusters_N": cN,
            "n_clusters_both": int((cm[R].notna() & cm[N].notna()).sum()), "delta": float(dl(cm)),
            "p": float(m.pvalues["cond"]), "p_kind": "ols_cluster_t", "ci95": pct(bm, .95), "ci90": pct(bm, .90),
            "mean_R": float(cm[R].mean()), "mean_N": float(cm[N].mean())}


def effect(df, col, design, clu="pid"):
    if df is None:
        return None
    if design == "paired":
        return paired(df, col, clu)
    if design == "all_clusters":
        return allclusters(df, col, clu)
    return descr(df, col, clu)


def hp_verdict(e):
    if not e or "p" not in e:
        return "not_estimable"
    if e["p"] < 0.05 and e["ci95"][1] < 0:
        return "supported"
    if e["ci95"][0] > 0:
        return "opposite_direction"
    return "inconclusive"


def ols_cluster(df, formula, design, clu="pid"):
    if df is None or design == "unsupported":
        return None
    d = df[df.ok].copy()
    d["cond"] = (d.task == R).astype(int)
    d["logP"] = np.log(pd.to_numeric(d.gait_P, errors="coerce"))
    cols = [c for c in ["gait_c", "gait_fast_frac", "speed", "logP", "gaze_pitch", "step_freq", "invalid_frac_walk"] if c in formula]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    if d[clu].nunique() < MIN_CLU or d.cond.nunique() < 2:
        return {"note": f"< {MIN_CLU} clusters or one condition", "n_clusters": int(d[clu].nunique())}
    m = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[clu])[0]}, use_t=True)
    return {k: {"coef": float(m.params[k]), "se": float(m.bse[k]), "p": float(m.pvalues[k]),
                "ci95": [float(x) for x in m.conf_int().loc[k]]} for k in m.params.index} | {
        "n_rec": int(m.nobs), "n_clusters": int(d[clu].nunique()), "se": "cluster-robust, t(G-1)"}


def run(df, design, clu="pid"):
    return {"delta_c": effect(df, "gait_c", design, clu), "delta_kappa_sp": effect(df, "gait_kappa_sp", design, clu)}


def abs_contrast(df, design, clu="pid"):
    """H-S4: |delta kappa_raw| - |delta kappa_sp|, joint cluster bootstrap (both measures per resample)."""
    if df is None or design == "unsupported":
        return None
    cr, cs = cmeans(df, "gait_kappa_raw", clu), cmeans(df, "gait_kappa_sp", clu)
    if R not in cr or N not in cr or R not in cs or N not in cs:
        return None
    cm = pd.concat({"raw": cr[[R, N]], "sp": cs[[R, N]]}, axis=1)
    rng = np.random.default_rng(a.seed)
    if design == "paired":
        cm = cm.dropna()
        f = lambda m: abs((m["raw"][R] - m["raw"][N]).mean()) - abs((m["sp"][R] - m["sp"][N]).mean())
    else:
        f = lambda m: abs(m["raw"][R].mean() - m["raw"][N].mean()) - abs(m["sp"][R].mean() - m["sp"][N].mean())
    if len(cm) < MIN_CLU:
        return {"note": f"< {MIN_CLU} clusters"}
    return {"est": float(f(cm)), "ci95": pct(boot_union(cm, rng, f), .95), "n_clusters": int(len(cm))}


main = load("main")
with open(a.cohort) as fh:
    cohort = json.load(fh)
g1 = G1_MAP[cohort["gate"]["primary_design"]]
g2 = gate2(main)
design = ORDER[min(ORDER.index(g1), ORDER.index(g2["design"]))]
loaded = main[main.load_status == "loaded"]
# registered start rule: k = 1 fit if converged, else k = 0 fit. Each start over-masks in a different
# regime (synthetic start-regime check), so one failed start is expected and is not by itself evidence of
# a second fixed point; when both converge, a difference > START_TOL flags start sensitivity.
both_conv = (loaded.gait_start1_status == "converged") & (loaded.gait_start0_status == "converged")
st_sens = loaded.ok & both_conv & (loaded.gait_kappa_sp_start1 - loaded.gait_kappa_sp_start0).abs().gt(START_TOL)
xt = pd.crosstab(loaded.gait_status.fillna("missing").astype(str),
                 loaded.gait_start1_status.fillna("missing").astype(str) + "/" + loaded.gait_start0_status.fillna("missing").astype(str))
out = {"gate1": cohort["gate"], "gate2": g2, "design": design, "clusters": "date_age pseudo-participants "
       "(recording date x age entry); the same person recorded on different dates forms separate clusters "
       "even when the age entry is unchanged, so clusters are assumed independent across dates",
       "status_x_start1_start0_status": {r: xt.loc[r].to_dict() for r in xt.index},
       "start_used_among_ok": loaded[loaded.ok].gait_start_used.value_counts().to_dict(),
       "start_used_among_ok_by_task": {f"task{k}": v.gait_start_used.value_counts().to_dict()
                                       for k, v in loaded[loaded.ok].groupby("task")},
       "status_x_start1_start0_status_by_task": {
           f"task{k}": pd.crosstab(v.gait_status.fillna("missing").astype(str),
                                   v.gait_start1_status.fillna("missing").astype(str) + "/" +
                                   v.gait_start0_status.fillna("missing").astype(str)).to_dict("index")
           for k, v in loaded.groupby("task")},
       "n_fit_not_converged": int((loaded.gait_fit_status == "not_converged").sum()),
       "n_start_sensitive_among_ok": int(st_sens.sum()),
       "both_starts_converged_among_ok": int((loaded.ok & both_conv).sum())}

# ---- primary: estimator-separation contrast c = kappa_raw - kappa_sp
hp = effect(main, "gait_c", design)
out["H_P"] = {"effect": hp, "verdict": hp_verdict(hp) if design != "unsupported" else "unsupported_descriptive_only"}

# ---- secondaries
s1 = effect(main, "gait_kappa_sp", design)
if s1 and "ci90" in s1:
    s1["tost_equivalent_pm0.10_bootstrap_ci_inversion"] = bool(s1["ci90"][0] > -EQ and s1["ci90"][1] < EQ)
out["H_S1"] = s1
cmN = cmeans(main, "gait_kappa_sp", "pid")
xN = cmN[N].dropna().values if N in cmN else np.array([])
if len(xN) >= 2:
    rng = np.random.default_rng(a.seed)
    ci = pct(xN[rng.integers(0, len(xN), (a.B, len(xN)))].mean(1), .95)
    out["H_S2"] = {"n_clusters": int(len(xN)), "mean": float(xN.mean()), "ci95": ci}
    if design != "unsupported" and len(xN) >= MIN_CLU:
        out["H_S2"]["pass_lo_gt_0.5"] = bool(ci[0] > 0.5)
out["H_S3_diagnostic"] = ols_cluster(main, "gait_c ~ gait_fast_frac + cond", design)
s4 = effect(main, "gait_kappa_raw", design)
if s4 is not None and design != "unsupported":
    s4["abs_raw_minus_abs_sp"] = abs_contrast(main, design)
out["H_S4"] = s4
desc = {}
for b in ("gait", "slow"):
    d = main.copy()
    d["ok"] = truthy(d[f"{b}_ok"])
    for m in ("kappa_sp", "kappa_raw", "c"):
        desc[f"{b}_{m}"] = descr(d, f"{b}_{m}")
    desc[f"{b}_status_by_task"] = {f"task{k}": v[f"{b}_status"].fillna("missing").value_counts().to_dict()
                                   for k, v in d.groupby("task")}
out["H_S5_descriptive"] = desc
out["H_S6_c_common"] = effect(main, "gait_c_common", design)

# ---- sensitivities (pre-registered list, numbered as in the plan); same design as the primary
sens = {}
for cfg in ("thr30", "thr60", "dil1", "dil4"):
    sens[f"1_{cfg}"] = run(load(cfg), design)
sens["2_personalized_only"] = run(main[main.gaze_kind == "personalized"].copy(), design)
g2d = gate2(main, "pid_date")
dd = ORDER[min(ORDER.index(design), ORDER.index(g2d["design"]))]
sens["3_date_clusters"] = {"design": dd, **run(main, dd, "pid_date")}
sens["4_covariates"] = ols_cluster(main, "gait_c ~ cond + speed + logP + gaze_pitch + step_freq + invalid_frac_walk", design)
for cfg in ("trans0.6", "trans5"):
    d = load(cfg)
    if d is not None:
        d["gait_c_trans"] = d.gait_kappa_raw - d.gait_kappa_sp_rot_given_trans
        sens[f"5_{cfg}_posthoc_regression"] = {"delta_c": effect(d, "gait_c_trans", design),
                                               "delta_kappa_sp": effect(d, "gait_kappa_sp_rot_given_trans", design),
                                               "gait_trans_coef_median": float(d[d.ok].gait_trans_coef.median())}
if design == "paired":
    sens["6_all_clusters"] = run(main, "all_clusters")
sens["7_cpf_components"] = {ax: {"delta_c": effect(main, f"gait_c_{ax}", design),
                                 "delta_kappa_sp": effect(main, f"gait_kappa_sp_{ax}", design)} for ax in ("cx", "cy")}
ex = main[~((main.task == N) & main.activity.astype(str).str.startswith("Physical Exercise"))].copy()
sens["8_exclude_exercise"] = run(ex, design)
sens["9_min20"] = run(load("min20"), design)
sens["10a_exclude_start_disagreement"] = run(main[~main.index.isin(st_sens[st_sens].index)].copy(), design)
sens["10b_both_starts_converged_only"] = run(main[main.index.isin(both_conv[both_conv].index)].copy(), design)
sens["10c_start1_only"] = run(main[main.gait_start_used == 1].copy(), design)
lin = main.copy()
lin["gait_c_first"] = lin.gait_kappa_raw - lin.gait_kappa_sp_first
sens["kappa_sp_first"] = {"delta_c": effect(lin, "gait_c_first", design), "delta_kappa_sp": effect(lin, "gait_kappa_sp_first", design)}
out["sensitivity"] = sens

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w") as fh:
    json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps({k: out.get(k) for k in ("gate2", "design", "H_P", "H_S1", "H_S2", "H_S6_c_common")}, indent=1, default=str))
