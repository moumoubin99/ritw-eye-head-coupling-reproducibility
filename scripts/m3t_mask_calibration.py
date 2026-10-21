"""M3-T pre-freeze check (no recorded gaze / head / hand kinematics used): recording-specific equal-mixture
calibration of the primary contrast b - a under each task-12 recording's own walking mask, gaze-invalid mask
and primary-hand missing-data pattern (handref.mask_calibration; truth 0).
Output: results/m3t/m3t_mask_calibration.csv and _summary.json"""
import json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa import io as rio
from ritw_kappa import slowphase as sp
from ritw_kappa import handref as hr

ROOT = "data/raw/seattle"
c = json.load(open("results/m3_cohort_frozen.json"))
meta = {r["rid"]: r for r in c["recordings"]}
rec, _ = rio.load_annotations()


def one(rid):
    r = rec[rec.id == rid].iloc[0]
    D = sp.load_recording(ROOT, rid, (float(r.whisper_start), float(r.whisper_end)))
    out = {"rid": rid, "task": meta[rid]["task"], "pid": meta[rid]["pid"]}
    if isinstance(D, str):
        return {**out, "load": D}
    fr = hr.read_frames(rio.find_recording_dir(ROOT, rid))
    side, G = hr.primary_side(fr, D["t"], D["walk"])
    out.update(load="ok", side=side, cov_primary=float((~G[side]["invalid"])[D["walk"]].mean()))
    for f in hr.CAL_FREQS_REC:
        cr = hr.mask_calibration(D, fr, side, f)
        out[f"cal_ba_{f}"], out[f"cal_ok_{f}"] = cr["b_minus_a"], cr["ok"]
    return out


if __name__ == "__main__":
    ids = [r for r in c["rec_ids"] if meta[r]["task"] == 12]
    df = pd.DataFrame(Parallel(n_jobs=10)(delayed(one)(r) for r in ids))
    df.to_csv("results/m3t/m3t_mask_calibration.csv", index=False)
    d = df[(df.load == "ok") & (df.cov_primary >= 0.5)]
    summ = {}
    for f in hr.CAL_FREQS_REC:
        x = d[d[f"cal_ok_{f}"].astype(bool)][f"cal_ba_{f}"]
        cm = d[d[f"cal_ok_{f}"].astype(bool)].groupby("pid")[f"cal_ba_{f}"].mean()
        summ[str(f)] = {"n_ok": int(len(x)), "n": int(len(d)), "rec_median": float(x.median()), "rec_min": float(x.min()),
                        "rec_max": float(x.max()), "rec_absmax": float(x.abs().max()), "cluster_mean": float(cm.mean())}
    json.dump(summ, open("results/m3t/m3t_mask_calibration_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
