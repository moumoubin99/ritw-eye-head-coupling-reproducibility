"""V1 synthetic recovery grid (data-free). Writes results/v1_grid.csv and results/v1_summary.json."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from ritw_kappa import v1
from ritw_kappa.signal import group_delay_step

ap = argparse.ArgumentParser()
ap.add_argument("--kappas", type=float, nargs="+", default=[0.3, 0.6, 0.9, 1.0])
ap.add_argument("--Ps", type=float, nargs="+", default=[0.0005, 0.001, 0.003, 0.01, 0.03, 0.1])
ap.add_argument("--ds", type=float, nargs="+", default=[0.35, 0.6, 1.5, 5.0])
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--n_jobs", type=int, default=2)
ap.add_argument("--p_min", type=float, default=v1.LAM_PMIN, help="training-fold P_min (placeholder until R001)")
ap.add_argument("--bias_tol", type=float, default=0.1)
ap.add_argument("--outdir", default="results")
a = ap.parse_args()

t0 = time.time()
scales = {P: v1.scale_for_P(P) for P in a.Ps}
jobs = [(k, P, d, s) for k in a.kappas for P in a.Ps for d in a.ds for s in range(a.seeds)]
rows = Parallel(n_jobs=a.n_jobs, verbose=0)(delayed(v1.cell)(k, P, d, 1000 + s, scale=scales[P], p_min=a.p_min) for k, P, d, s in jobs)
df = pd.DataFrame([r for rr in rows for r in rr])
os.makedirs(a.outdir, exist_ok=True)
df.to_csv(os.path.join(a.outdir, "v1_grid.csv"), index=False)

g = df[df.kind == "grid"].groupby(["P_target", "d", "sigma", "W", "kappa_true"]).agg(
    bias=("bias", "mean"), rmse=("rmse", "mean"), sd=("kappa_hat", "std"), P_meas=("P_meas", "median")).reset_index()
acc = g.groupby(["P_target", "d", "sigma", "W"]).agg(max_abs_bias=("bias", lambda x: np.max(np.abs(x))),
                                                       max_sd=("sd", "max"), max_rmse=("rmse", "max"), P_meas=("P_meas", "median")).reset_index()
acc["acceptable"] = acc.max_abs_bias < a.bias_tol
acc.to_csv(os.path.join(a.outdir, "v1_acceptable_cells.csv"), index=False)
reg = (df.P_target >= a.p_min) & (df.d >= 1.5)   # operating regime: above P_min, non-near targets
mm = df[(df.kind == "misalign") & reg]
b0 = mm[mm.mis == 0].set_index(["kappa_true", "P_target", "d", "seed"]).kappa_hat
mm = mm.assign(dk=mm.kappa_hat.values - b0.reindex(pd.MultiIndex.from_frame(mm[["kappa_true", "P_target", "d", "seed"]])).values)
mis = mm.groupby("mis").dk.agg(["mean", "std"]).reset_index()
com = df[df.kind == "common_rot"]
summ = {
    "n_cells": int(len(jobs)), "sec": round(time.time() - t0, 1), "bias_tol": a.bias_tol,
    "P_min_used": a.p_min, "P_min_is_placeholder": a.p_min == v1.LAM_PMIN,
    "bias_definition": "window-level mean of (kappa_hat - kappa_true) over all trailing windows after 3 s settling",
    "acceptable_frac_P_ge_Pmin_d_ge_1.5": float(acc[(acc.P_target >= a.p_min) & (acc.d >= 1.5)].acceptable.mean()),
    "acceptable_frac_by_P": acc.groupby("P_target").acceptable.mean().to_dict(),
    "acceptable_frac_by_d": acc.groupby("d").acceptable.mean().to_dict(),
    "acceptable_frac_by_sigma": acc.groupby("sigma").acceptable.mean().to_dict(),
    "acceptable_frac_by_W": acc.groupby("W").acceptable.mean().to_dict(),
    "misalignment_dkappa_vs_0deg_in_regime": {f"{m:g}deg": float(r["mean"]) for m, r in mis.set_index("mis").iterrows()},
    "bias_in_regime_by_kappa_true": df[(df.kind == "grid") & reg].groupby("kappa_true").bias.mean().to_dict(),
    "bias_in_regime_by_d": df[(df.kind == "grid") & reg].groupby("d").bias.mean().to_dict(),
    "rmse_in_regime_by_W": df[(df.kind == "grid") & reg].groupby("W").rmse.mean().to_dict(),
    "common_rotation_max_abs_diff": float(np.max(np.abs(com.kappa_hat - com.kappa_hat_base))),
    "group_delay_s": {str(W): group_delay_step(W) for W in (1.0, 2.0, 4.0)},
}
json.dump(summ, open(os.path.join(a.outdir, "v1_summary.json"), "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
