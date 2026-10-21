"""M3-T inference (refine-logs/EXPERIMENT_PLAN_M3T.md). Task 12 one-sample cluster-level tests; task 11
descriptive. Clusters = date_age pseudo-participants (column pid), as in the frozen M3 primary analysis."""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--indir", default="results/m3t")
ap.add_argument("--out", default="results/m3t/m3t_stats.json")
ap.add_argument("--B", type=int, default=10000)
ap.add_argument("--n_perm", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
COV_MIN, COV_HI = 0.5, 0.7     # primary-hand valid fraction of walking samples (gaze-free inclusion)
MIN_CLU = 10
EQ_LAMBDA = 0.15               # consistency band for kappa_W - lambda (90% CI inside +-EQ_LAMBDA)
R, N = 12, 11


def truthy(x):
    return x.astype(str).str.lower() == "true"


def load(cfg):
    p = os.path.join(a.indir, f"m3t_measures_{cfg}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    for k in ("H_ok", "W_ok", "J_ok"):
        df[k] = truthy(df[k]) if k in df else False
    df["d_ba"] = df.J_b_hand - df.J_a_world
    return df


def incl(df, cov_col="cov_primary", cov=COV_MIN, need=("J_ok",)):
    """Inclusion: loaded, hand coverage of walking samples >= cov, and the ok flags the endpoint uses."""
    m = (df.load_status == "loaded") & (df[cov_col] >= cov)
    for k in need:
        m &= df[k]
    return df[m].copy()


def signflip(d, rng):
    d = np.asarray(d, float); n, obs = len(d), d.mean()
    if n <= 16:
        S = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        return float(np.mean(np.abs(S @ d / n) >= abs(obs) - 1e-12)), "signflip_exact"
    S = rng.choice([-1, 1], size=(a.n_perm, n))
    return float((np.sum(np.abs(S @ d / n) >= abs(obs) - 1e-12) + 1) / (a.n_perm + 1)), f"signflip_random{a.n_perm}"


def one_sample(df, col, clu="pid"):
    d = df[np.isfinite(df[col])]
    cm = d.groupby(clu)[col].mean()
    res = {"n_rec": int(len(d)), "n_clusters": int(len(cm)), "mean": float(cm.mean()) if len(cm) else None}
    if len(cm) < MIN_CLU:
        return {**res, "note": f"< {MIN_CLU} clusters: descriptive only"}
    rng = np.random.default_rng(a.seed)
    bm = cm.values[rng.integers(0, len(cm), (a.B, len(cm)))].mean(1)
    p, kind = signflip(cm.values, rng)
    return {**res, "ci95": [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))],
            "ci90": [float(np.quantile(bm, .05)), float(np.quantile(bm, .95))], "p": p, "p_kind": kind}


def verdict(e):
    if "ci95" not in e:
        return "descriptive_only"
    if e["p"] < 0.05 and e["ci95"][0] > 0:
        return "supported"
    return "opposite" if e["ci95"][1] < 0 else "inconclusive"


main = load("main")
T12 = incl(main[main.task == R])                      # primary endpoint population (J_ok)
T12H = incl(main[main.task == R], need=("J_ok", "H_ok"))
T12W = incl(main[main.task == R], need=("J_ok", "W_ok"))
T12HW = incl(main[main.task == R], need=("J_ok", "H_ok", "W_ok"))
out = {"cohort": {"task12_loaded": int(((main.task == R) & (main.load_status == "loaded")).sum()),
                  "task12_cov_ge_min": int(((main.task == R) & (main.cov_primary >= COV_MIN)).sum()),
                  "task12_included": int(len(T12)), "task12_clusters": int(T12.pid.nunique()),
                  "task12_cov_ok_but_J_not_ok": int(((main.task == R) & (main.cov_primary >= COV_MIN)
                                                     & ~main.J_ok.fillna(False).astype(bool)).sum()),
                  "primary_side_task12": T12.primary_side.value_counts().to_dict(),
                  "H_status_task12": main[main.task == R].H_status.astype(str).value_counts().to_dict(),
                  "W_status_task12": main[main.task == R].W_status.astype(str).value_counts().to_dict()}}
gate = T12.pid.nunique() >= MIN_CLU
out["gate_ok"] = bool(gate)

# primary: b - a (joint two-regressor fit on the world-desaccaded slow-phase response)
e = one_sample(T12, "d_ba")
out["T1_primary_b_minus_a"] = {**e, "verdict": verdict(e) if gate else "descriptive_only"}
# secondaries
T12["d_shift_b"] = T12.J_b_hand - T12.Jshift_b_hand    # NaN (unavailable) when no shift was accepted
T12H["d_shift_kH"] = T12H.H_kappa_sp - T12H.Hshift_kappa_sp
T12HW["D1"] = (1 - T12HW.W_kappa_sp).abs() - (1 - T12HW.H_kappa_sp).abs()
T12W["dW_lambda"] = T12W.W_kappa_sp - T12W["lambda"]
out["T2_surrogate_b"] = one_sample(T12, "d_shift_b")
x = T12H[np.isfinite(T12H.d_shift_kH)]
out["T2_surrogate_kappaH_descriptive"] = {"n_rec": int(len(x)), "n_clusters": int(x.pid.nunique()),
                                          "cluster_mean": float(x.groupby("pid").d_shift_kH.mean().mean()) if len(x) else None,
                                          "note": "descriptive only (synthetic fragmented check failed, plan section H-T2)"}
out["surrogate_counts_task12"] = {"Jshift_n_ok": T12.Jshift_n_ok.describe().to_dict(),
                                  "Jshift_zero_accepted": int((T12.Jshift_n_ok == 0).sum()),
                                  "Hshift_n_ok": T12H.Hshift_n_ok.describe().to_dict()}
out["T3_D1_closer_to_one"] = one_sample(T12HW, "D1")
e = one_sample(T12W, "dW_lambda")
if "ci90" in e:
    e["consistent_pm0.15"] = bool(e["ci90"][0] > -EQ_LAMBDA and e["ci90"][1] < EQ_LAMBDA)
fin = T12W[np.isfinite(T12W.W_kappa_sp) & np.isfinite(T12W["lambda"])]
if len(fin) >= 5:   # descriptive: recording-level, no p-value (does not respect the clustering)
    e["spearman_rec_level_descriptive"] = {"rho": float(spearmanr(fin.W_kappa_sp, fin["lambda"])[0]), "n": int(len(fin))}
out["T4_kappaW_vs_lambda"] = e
desc = {}
POP = {"T12 (J_ok)": (T12, ("J_a_world", "J_b_hand", "d_ba", "J_rho", "Jshift_b_hand", "Jshift_a_world", "lambda",
                             "rho_HW", "cov_primary", "J_check")),
       "T12H (J_ok & H_ok)": (T12H, ("H_kappa_sp", "H_kappa_raw", "H_fast_frac", "H_P", "Hshift_kappa_sp")),
       "T12W (J_ok & W_ok)": (T12W, ("W_kappa_sp", "W_kappa_raw", "W_fast_frac", "W_P"))}
for pop, (df_, cols) in POP.items():
    for col in cols:
        x = df_.groupby("pid")[col].mean()
        desc[col] = {"population": pop, "cluster_mean": float(x.mean()), "cluster_sd": float(x.std()),
                     "rec_median": float(df_[col].median()), "n_rec": int(df_[col].notna().sum()),
                     "n_clusters": int(x.notna().sum())}
out["T5_descriptive_task12"] = desc
t11 = main[(main.task == N) & (main.load_status == "loaded")]
d11 = {"n_loaded": int(len(t11)), "cov_primary_median": float(t11.cov_primary.median())}
for lab, sub in (("cov_ge_0.5", incl(t11, need=("J_ok", "H_ok", "W_ok"))),
                 ("all_ok_any_cov", t11[t11.H_ok & t11.J_ok & t11.W_ok])):
    d11[lab] = {"n": int(len(sub)), **{c: float(sub[c].median()) if len(sub) else None
                                      for c in ("H_kappa_sp", "W_kappa_sp", "J_a_world", "J_b_hand", "lambda", "cov_primary")}}
out["T6_task11_descriptive"] = d11

# sensitivities of the primary endpoint (same inclusion rule unless stated)
sens = {}
for cfg in ("rigid", "depth", "thr30", "thr60", "mid"):
    d = load(cfg)
    if d is not None:
        sub = d[d.task == R]
        # midpoint: coverage of the analysed (both-hands) signal; others: primary-hand coverage
        sens[cfg] = one_sample(incl(sub, cov_col="hand_valid_walk" if cfg == "mid" else "cov_primary"), "d_ba")
d = load("other")
if d is not None:
    sens["other_hand"] = one_sample(incl(d[d.task == R], cov_col="cov_other"), "d_ba")
sens["J_rho_le_0.9"] = one_sample(T12[T12.J_rho.abs() <= 0.9], "d_ba")
# recording-specific missing-data calibration (handref.mask_calibration, 1.8 Hz): primary contrast minus the
# bias this recording's own walking / gaze-invalid / hand-missing pattern induces in an exact 50:50 mixture
if "cal_ba_1.8" in T12:
    Tc = T12[T12["cal_ok_1.8"].fillna(False).astype(bool)].copy()
    Tc["d_ba_cal"] = Tc.d_ba - Tc["cal_ba_1.8"]   # subtracts the synthetic-signal offset, not a real-bias correction
    sens["mask_calibrated_1.8Hz"] = one_sample(Tc, "d_ba_cal")
    out["mask_calibration_task12"] = {f: {"n_ok": int(T12[f"cal_ok_{f}"].fillna(False).astype(bool).sum()),
                                          "absmax": float(T12[f"cal_ba_{f}"].abs().max())}
                                      for f in ("1.0", "1.8", "3.0") if f"cal_ba_{f}" in T12}
sens["cov_ge_0.7"] = one_sample(incl(main[main.task == R], cov=COV_HI), "d_ba")
sens["pid_date_clusters"] = one_sample(T12, "d_ba", clu="pid_date")
sens["personalized_only"] = one_sample(T12[T12.gaze_kind == "personalized"], "d_ba")
out["sensitivities_primary"] = sens
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as fh:
    json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
