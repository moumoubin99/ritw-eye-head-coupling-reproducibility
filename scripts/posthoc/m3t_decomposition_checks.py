"""POST HOC (after the M3-T freeze, results/m3t_freeze.json; auto-review-loop round 1, AGY reviewer).
Not a registered endpoint; no frozen file is modified. Re-derives, with the frozen internals and the same
arguments as handref.joint_fit, the accepted world-desaccaded slow-phase response y = e_sp and the design
columns on Ssp for every task-12 recording with primary-hand coverage >= 0.5, then reports descriptive
decompositions of the registered H-T1 contrast:
  (1) a > 0 test; FWL identity kappa_W(Ssp) = a + b*gamma, gamma = <xH,xW>/|xW|^2; single-regressor hand
      slope on the same support and its omitted-variable term
  (2) predictor scale: RMS of xW and xH, standardised coefficients beta = coef*|x|/|y|, uncentred
      R^2 = 1 - SSE/sum(y^2) for W alone, H alone, joint; unique Delta R^2 of each predictor
  (3) rigid split: xH (translation) and x_rot = bp(qH_rigid - qH_trans) as separate regressors, and the
      median wrist-to-gaze-point lever arm |d g - p| on Ssp
  (4) head-translation control: x_ht = bp(-(I - g g^T) v_cpf / |p|) (fixed world point at wrist distance)
      added as a third regressor; and the 2-regressor fit of y on [xW + x_ht, xH] ("near-world" target)
  (5) model-neutral bridge: e_sp rebuilt with a k = 0 linear bridge across the same fast-phase spans
  (6) inclusion check from the frozen CSV: b - a over all task-12 recordings with finite (a, b)
All in-sample. Output: results/posthoc/m3t_decomposition_checks.{json,csv}"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, savgol_filter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ritw_kappa import io as rio
from ritw_kappa import slowphase as sp
from ritw_kappa import handref as hr

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/seattle")
ap.add_argument("--cohort", default="results/m3_cohort_frozen.json")
ap.add_argument("--main_csv", default="results/m3t/m3t_measures_main.csv")
ap.add_argument("--n_jobs", type=int, default=10)
ap.add_argument("--B", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/posthoc/m3t_decomposition_checks")
a = ap.parse_args()
THR, DIL = 40.0, 2

main = pd.read_csv(a.main_csv)
main["J_ok"] = main.J_ok.astype(str).str.lower() == "true"
cand = main[(main.task == 12) & (main.load_status == "loaded") & (main.cov_primary >= 0.5)]
rec, _ = rio.load_annotations()
meta = {r["rid"]: r for r in json.load(open(a.cohort))["recordings"]}


def ls(X, y):
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c, 1 - np.sum((y - X @ c) ** 2) / np.sum(y ** 2)


def one(rid):
    r = rec[rec.id == rid].iloc[0]
    D = sp.load_recording(a.root, rid, (float(r.whisper_start), float(r.whisper_end)))
    fr = hr.read_frames(rio.find_recording_dir(a.root, rid))
    side, G = hr.primary_side(fr, D["t"], D["walk"])
    G = G[side]
    t, g = D["t"], D["g"]
    inv = D["invalid"] | G["invalid"]
    qH = hr.gpred_hand(g, G, "trans")
    qR = hr.gpred_hand(g, G, "rigid")
    rr = sp.band_gains(t, g, D["gdot"], D["w_cpf"], D["walk"], inv, THR, DIL, bands=("gait",),
                       v_cpf=-qH, depth=np.ones(len(t)))["gait"]
    if rr.get("fit_status") == "no_fit":
        return {"rid": int(rid), "status": "no_fit"}
    B = sp.BANDS["gait"]
    sos = butter(2, [B["lo"], B["hi"]], "bandpass", fs=sp.FS, output="sos")
    gdet = savgol_filter(g, 5, 2, deriv=1, delta=1 / sp.FS, axis=0)
    gpred = -np.cross(D["w_cpf"], g)
    k0 = 0.0 if rr["start_used"] == 0 else 1.0
    fit = sp._fit_sp(gdet, D["gdot"], gpred, inv, D["walk"], sos, B, THR, DIL, 0.5, k0, 50, 1e-3)
    st = fit["st"]
    S, segs = st["Ssp"], st["segs"]
    bp = lambda x: sp._bp_segments(sos, x, segs)[S].ravel()
    y, xW, xH = st["e_sp"][S].ravel(), st["h"][S].ravel(), bp(qH)
    xR = bp(qR - qH)
    d = np.linalg.norm(G["p"], axis=1)
    vperp = D["v_cpf"] - np.einsum("ni,ni->n", D["v_cpf"], g)[:, None] * g
    xT = bp(np.nan_to_num(-vperp / d[:, None]))
    bridge, _ = sp.short_runs(st["F"], 0.5)
    y0 = sp._bp_segments(sos, sp.impute(D["gdot"], gpred, 0.0, bridge), segs)[S].ravel()
    nrm = lambda x: float(np.sqrt(np.sum(x ** 2)))
    cj, R2j = ls(np.c_[xW, xH], y)
    cw, R2w = ls(xW[:, None], y)
    ch, R2h = ls(xH[:, None], y)
    c3r, R23r = ls(np.c_[xW, xH, xR], y)
    c3t, R23t = ls(np.c_[xW, xH, xT], y)
    cwt, R2wt = ls(np.c_[xW, xT], y)
    cn, R2n = ls(np.c_[xW + xT, xH], y)
    c0, _ = ls(np.c_[xW, xH], y0)
    lev = np.linalg.norm(d[:, None] * g - G["p"], axis=1)[st["Ssp"]]
    gam = float(xH @ xW / (xW @ xW))
    return {"rid": int(rid), "status": "ok", "a": cj[0], "b": cj[1], "gamma": gam,
            "kW_Ssp": float(cw[0]), "kH_SspW": float(ch[0]), "ovb_H": float(cj[0] * (xW @ xH) / (xH @ xH)),
            "rms_W": nrm(xW) / np.sqrt(S.sum()), "rms_H": nrm(xH) / np.sqrt(S.sum()),
            "rms_rot": nrm(xR) / np.sqrt(S.sum()), "ratio_WH": nrm(xW) / nrm(xH),
            "beta_W": cj[0] * nrm(xW) / nrm(y), "beta_H": cj[1] * nrm(xH) / nrm(y),
            "R2_W": R2w, "R2_H": R2h, "R2_joint": R2j, "dR2_W": R2j - R2h, "dR2_H": R2j - R2w,
            "rig3_a": c3r[0], "rig3_b_tr": c3r[1], "rig3_c_rot": c3r[2], "lever_med_m": float(np.median(lev)),
            "ht3_a": c3t[0], "ht3_b": c3t[1], "ht3_c_ht": c3t[2], "R2_W_ht": R2wt, "R2_ht3": R23t,
            "near_a": cn[0], "near_b": cn[1], "k0_a": c0[0], "k0_b": c0[1]}


rows = Parallel(n_jobs=a.n_jobs)(delayed(one)(int(r)) for r in cand.rid)
df = pd.DataFrame(rows).merge(cand[["rid", "pid", "J_ok", "J_a_world", "J_b_hand", "W_kappa_sp", "H_kappa_sp"]], on="rid")
df["check"] = np.abs(df.a - df.J_a_world).combine(np.abs(df.b - df.J_b_hand), max)
for k, (u, v) in {"d_ba": ("b", "a"), "d_beta": ("beta_H", "beta_W"), "d_dR2": ("dR2_H", "dR2_W"),
                  "d_rig3": ("rig3_b_tr", "rig3_a"), "d_ht3": ("ht3_b", "ht3_a"), "d_near": ("near_b", "near_a"),
                  "d_k0": ("k0_b", "k0_a")}.items():
    df[k] = df[u] - df[v]
df["d_R2_hand_over_W_ht"] = df.R2_ht3 - df.R2_W_ht
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out + ".csv", index=False)


def clu(d, col):
    cm = d.groupby("pid")[col].mean()
    v = cm.values
    rng = np.random.default_rng(a.seed)
    bm = v[rng.integers(0, len(v), (a.B, len(v)))].mean(1)
    n = len(v)
    if n <= 16:
        Sg = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        p = float(np.mean(np.abs(Sg @ v / n) >= abs(v.mean()) - 1e-12))
    else:
        Sg = rng.choice([-1, 1], size=(20000, n))
        p = float((np.sum(np.abs(Sg @ v / n) >= abs(v.mean()) - 1e-12) + 1) / 20001)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(bm, .025)), float(np.quantile(bm, .975))],
            "p_signflip": p, "n_rec": int(len(d)), "n_clusters": n, "n_rec_positive": int((d[col] > 0).sum())}


J = df[df.J_ok]
cols = ["a", "b", "gamma", "kW_Ssp", "kH_SspW", "ovb_H", "rms_W", "rms_H", "rms_rot", "ratio_WH", "beta_W", "beta_H",
        "R2_W", "R2_H", "R2_joint", "dR2_W", "dR2_H", "rig3_a", "rig3_b_tr", "rig3_c_rot", "lever_med_m", "ht3_a",
        "ht3_b", "ht3_c_ht", "R2_W_ht", "R2_ht3", "near_a", "near_b", "k0_a", "k0_b"]
out = {"label": "POST HOC (after freeze); descriptive decompositions of the registered H-T1; in-sample",
       "population": "task 12, J_ok (H-T1 population), cluster = pid (date x age)",
       "max_abs_check_vs_frozen_csv": float(J.check.max()),
       "cluster_means": {c: float(J.groupby("pid")[c].mean().mean()) for c in cols},
       "tests": {c: clu(J, c) for c in ("a", "d_ba", "d_beta", "d_dR2", "d_rig3", "d_ht3", "d_near", "d_k0",
                                         "d_R2_hand_over_W_ht")}}
fin = main[(main.task == 12) & (main.load_status == "loaded") & np.isfinite(main.J_a_world) & np.isfinite(main.J_b_hand)].copy()
fin["d_ba"] = fin.J_b_hand - fin.J_a_world
conv = fin[fin.W_status.astype(str) != "not_converged"] if "W_status" in fin else fin
ex = fin[~fin.J_ok & (fin.cov_primary >= 0.5)]
out["inclusion"] = {"excluded_cov_ok_finite": ex[["rid", "pid", "W_status", "cov_primary", "walk_s", "W_n_Ssp_s", "d_ba"]]
                    .to_dict("records"),
                    "all_finite": clu(fin, "d_ba"), "all_finite_excluding_not_converged": clu(conv, "d_ba"),
                    "all_finite_cov_ge_0.5": clu(fin[fin.cov_primary >= 0.5], "d_ba")}
json.dump(out, open(a.out + ".json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
