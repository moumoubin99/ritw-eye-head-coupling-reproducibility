"""V1 extension: validates the data-driven residual-rotation diagnostic (signal.residual_rotation) under
the V1 nuisances (known misalignment x eye translation depth x gaze noise x constant / switching / drifting
kappa). The worst-case angle error over the grid (`tol_deg`) is what run_v1_coverage.py adds to each real
recording's estimated angle before comparing it with the 10 deg V1 misalignment bound."""
import argparse, json, os, sys, itertools
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa import v1

ap = argparse.ArgumentParser()
ap.add_argument("--mis", type=float, nargs="+", default=[0, 3, 6, 10, 15])
ap.add_argument("--depths", type=float, nargs="+", default=[0.35, 0.6, 1.5, 5.0])
ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 1.0])
ap.add_argument("--kappas", nargs="+", default=["const", "switch", "drift"])
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--n_jobs", type=int, default=8)
ap.add_argument("--out", default="results/v1_rotation_recovery.json")
a = ap.parse_args()
grid = list(itertools.product(a.mis, a.depths, a.sigmas, a.kappas, range(a.seeds)))
rows = Parallel(n_jobs=a.n_jobs)(delayed(v1.rotation_recovery)(m, d, s, k, 100 + i)
                                 for i, (m, d, s, k, _) in enumerate(grid))
R = pd.DataFrame(rows)
ok = R[R.resolved]
summ = {"n": len(R), "n_resolved": int(len(ok)),
        "angle_err_max": float(ok.angle_err.max()), "angle_err_p95": float(ok.angle_err.quantile(0.95)),
        "rot_err_max": float(ok.rot_err_deg.max()),
        "by_kappa": ok.groupby("kappa").angle_err.max().to_dict(),
        "by_depth": {f"{k:g}": v for k, v in ok.groupby("d").angle_err.max().to_dict().items()},
        "by_sigma": {f"{k:g}": v for k, v in ok.groupby("sigma").angle_err.max().to_dict().items()}}
summ["tol_deg"] = summ["angle_err_max"]
summ["validated"] = bool(len(ok) == len(R) and summ["tol_deg"] < 5.0)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
R.to_csv(a.out.replace(".json", ".csv"), index=False)
json.dump(summ, open(a.out, "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
