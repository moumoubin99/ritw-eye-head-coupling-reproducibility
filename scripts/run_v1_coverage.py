"""R007: V1 acceptable-region coverage of REAL event windows. Every E1 event window (pos and neg, all
qualified, at t_s - H) is re-simulated with the recording's own head rotation, eye translation and gaze
geometry for kappa_true in {0.3, 0.6, 0.9, 1.0}. A window is covered iff
  its ACTUAL E1 feature row is excited: P finite, P >= fold P_min and m = 1 (fold = the training clusters
  that exclude the event's cluster; a window with m = 0 is never covered), and
  max_kappa |bias| < 0.1, where bias is computed under the recording's observed validity mask, using the
  time-varying MEASURED depth series when MPS depth covers >= 50% of the analysed W window (any sample
  still missing depth is filled with each fallback depth and the worst case taken), otherwise the WORST of
  the fallback depths {0.35, 0.6, 1.5, 5} m (missing depth is never resolved optimistically).
  and the geometry is inside the V1 misalignment bound (10 deg). Geometry basis:
  - if the data-driven residual-rotation diagnostic is VALIDATED under the V1 nuisances
    (run_v1_rotation.py: all cases resolved and worst-case error tol_deg < 5), a recording counts only when
    its estimated angle + tol_deg <= 10 deg (unresolved -> not covered);
  - otherwise (the current state: near-depth gait-locked translation makes the rotation weakly identifiable)
    the bound rests on the documented default extrinsic and V1 coverage is reported as CONDITIONAL on it
    (geometry_basis = "documented_extrinsic_assumed"); the per-recording angles are descriptive only.
Pass: coverage >= 0.80 at the primary noise level (sigma 0.3 deg); sigma 1.0 deg reported as sensitivity.
W, H and grouping come from the frozen cohort for real data (an explicit different --W/--H is refused).
--synthetic N runs the same code on the synthetic cohort (pipeline check only)."""
import argparse, json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa.features import KappaPool
from ritw_kappa import experiment as ex, v1
from ritw_kappa.signal import residual_rotation

FALLBACK_D = (0.35, 0.6, 1.5, 5.0)
KAPPAS = (0.3, 0.6, 0.9, 1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", type=int, default=0)
ap.add_argument("--cohort", default="results/cohort_frozen.json")
ap.add_argument("--W", type=float, default=None)
ap.add_argument("--H", type=float, default=None)
ap.add_argument("--rotation", default="results/v1_rotation_recovery.json")
ap.add_argument("--mis_bound", type=float, default=10.0)
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 1.0])
ap.add_argument("--tol", type=float, default=0.1)
ap.add_argument("--n_jobs", type=int, default=2)
ap.add_argument("--grouping", default=None)
ap.add_argument("--tag", default="main")
ap.add_argument("--outdir", default="results")
a = ap.parse_args()

if a.synthetic:
    from ritw_kappa.cohort import synthetic_rec
    recs = [synthetic_rec(f"p{i}", i) for i in range(a.synthetic)]
    W, H, src, grouping = a.W or 2.0, a.H or 1.0, "synthetic", None
else:
    from ritw_kappa.cohort import load_cohort
    c = json.load(open(a.cohort))
    W, H, src = c["W"], c["H"], c["decision"]
    if (a.W is not None and a.W != W) or (a.H is not None and a.H != H):
        raise SystemExit(f"--W/--H differ from the frozen cohort (W={W}, H={H}); V1 coverage uses the frozen values")
    grouping = a.grouping or c.get("grouping", "date_age")
    recs, _, _ = load_cohort(c["root"], c["rec_ids"], 2, strict=True, grouping=grouping)
rot = json.load(open(a.rotation)) if os.path.exists(a.rotation) else None
rot_tol = float(rot["tol_deg"]) if rot and rot.get("validated") else None
geom = {}
for r in recs:
    rr = residual_rotation(r.s)
    ang = rr["angle_deg"] if rr else None
    conv = bool(rr and rr.get("converged"))
    geom[r.rec_id] = {"angle_deg": ang, "converged": conv,
                      "ok": bool(rot_tol is None or (conv and ang + rot_tol <= a.mis_bound))}
geometry_basis = "residual_rotation_validated" if rot_tol is not None else "documented_extrinsic_assumed"
pool = KappaPool(recs, W)
pids = sorted({r.pid for r in recs})
fk = {p: pool.fit({q: 1 for q in pids if q != p}) for p in pids}
pmin = {p: fk[p].P_min for p in pids}
df = ex.e1_rows(recs, W, H)
df = df[df.cls.isin(["pos", "neg"])]
df = pd.concat([fk[p].transform(g) for p, g in df.groupby("pid")]).sort_index()
byrec = {r.rec_id: r for r in recs}


def one_rec(rid, rows):
    r = byrec[rid]
    out = []
    for _, e in rows.iterrows():
        te = e.t_s - H
        excited = bool(np.isfinite(e.P) and e.P >= pmin[r.pid] and e.m > 0)
        for sg in a.sigmas:
            res = v1.driven_window_error(r, te, W, pmin[r.pid], KAPPAS, FALLBACK_D, sigma_deg=sg,
                                         seed=int(e.name), measured_depth=True)
            row = {"rec_id": rid, "pid": r.pid, "cls": e.cls, "t_s": e.t_s, "matched": bool(e.matched),
                   "sigma": sg, "P_min": pmin[r.pid], "P_row": float(e.P), "m_row": float(e.m),
                   "excited": excited, "geometry_ok": geom[rid]["ok"]}
            if res is None:
                row.update(valid=False, covered=False, covered_no_geom=False, depth_measured=False)
            else:
                dm = res["measured_depth_used"]
                dset = res["measured_keys"] if dm else FALLBACK_D
                worst = max(abs(res[(k, d)][0]) for k in KAPPAS for d in dset)
                rmse = max(res[(k, d)][1] for k in KAPPAS for d in dset)
                row.update(valid=True, depth_measured=dm, max_abs_bias=worst, max_rmse=rmse,
                           covered=bool(excited and worst < a.tol and geom[rid]["ok"]),
                           covered_no_geom=bool(excited and worst < a.tol),
                           **{f"bias_d{d:g}": max(abs(res[(k, d)][0]) for k in KAPPAS) for d in FALLBACK_D})
            out.append(row)
    return out


parts = Parallel(n_jobs=a.n_jobs)(delayed(one_rec)(rid, g) for rid, g in df.groupby("rec_id"))
R = pd.DataFrame([x for p in parts for x in p])
pid_of = {r.rec_id: r.pid for r in recs}
summ = {"source": src, "W": W, "H": H, "tol": a.tol, "n_windows": int(len(R) // len(a.sigmas)),
        "depth_measured_frac": float(R.depth_measured.mean()) if len(R) else np.nan,
        "settings": {"rec_ids": sorted(int(r.rec_id) for r in recs) if not a.synthetic else None,
                     "pid_map": ({str(int(k)): v for k, v in sorted(pid_of.items())} if not a.synthetic else None),
                     "W": W, "H": H, "grouping": grouping},
        "geometry": {"basis": geometry_basis, "rotation_check": a.rotation,
                     "rotation_validated": rot_tol is not None, "tol_deg": rot_tol,
                     "mis_bound_deg": a.mis_bound, "per_recording": {str(k): v for k, v in geom.items()},
                     "frac_recordings_ok": float(np.mean([v["ok"] for v in geom.values()])) if geom else np.nan}}
for sg, g in R.groupby("sigma"):
    summ[f"sigma{sg:g}"] = {
        "coverage": float(g.covered.mean()), "coverage_pos": float(g[g.cls == "pos"].covered.mean()),
        "coverage_neg": float(g[g.cls == "neg"].covered.mean()), "valid_frac": float(g.valid.mean()),
        "coverage_matched": float(g[g.matched].covered.mean()),
        "coverage_no_geometry_gate": float(g.covered_no_geom.mean()),
        "excited_frac": float(g.excited.mean()),
        "median_max_abs_bias": float(g.max_abs_bias.median()) if "max_abs_bias" in g else np.nan,
        "frac_bias_ok_by_fallback_depth": {f"{d:g}": float((g[f"bias_d{d:g}"] < a.tol).mean())
                                           for d in FALLBACK_D if f"bias_d{d:g}" in g},
    }
summ["V1_pass"] = bool(summ.get(f"sigma{a.sigmas[0]:g}", {}).get("coverage", 0) >= 0.80)
os.makedirs(a.outdir, exist_ok=True)
R.to_csv(os.path.join(a.outdir, f"v1_coverage_{a.tag}.csv"), index=False)
json.dump(summ, open(os.path.join(a.outdir, f"v1_coverage_{a.tag}.json"), "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
