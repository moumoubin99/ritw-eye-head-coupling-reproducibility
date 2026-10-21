"""M3-T recording-level measures (refine-logs/EXPERIMENT_PLAN_M3T.md). Same frozen M3 cohort and window
(whisper_start..whisper_end), same frozen estimator (slowphase.band_gains); the hand-referenced prediction
enters through handref.w_equiv. Output: results/m3t/m3t_measures_<cfg>.csv"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa import io as rio
from ritw_kappa import slowphase as sp
from ritw_kappa import handref as hr

# name: (hand source, mode, depth, thr, n_shift, world)
CONFIGS = {
    "main": ("primary", "trans", "norm", 40.0, hr.N_SHIFT, True),
    "rigid": ("primary", "rigid", "norm", 40.0, 0, False),
    "depth": ("primary", "trans", "vergence", 40.0, 0, False),
    "other": ("other", "trans", "norm", 40.0, 0, True),
    "mid": ("mid", "trans", "norm", 40.0, 0, True),
    "thr30": ("primary", "trans", "norm", 30.0, 0, True),
    "thr60": ("primary", "trans", "norm", 60.0, 0, True),
}
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/seattle")
ap.add_argument("--cohort", default="results/m3_cohort_frozen.json")
ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
ap.add_argument("--n_jobs", type=int, default=10)
ap.add_argument("--outdir", default="results/m3t")
ap.add_argument("--limit", type=int, default=0, help="smoke test: first N cohort recordings")
a = ap.parse_args()
rec, _ = rio.load_annotations()
with open(a.cohort) as fh:
    c = json.load(fh)
meta = {r["rid"]: r for r in c["recordings"]}


def one(rid):
    m = meta[rid]
    base = {"rid": int(rid), "task": m["task"], "pid": m["pid"], "pid_date": m["pid_date"], "activity": m["activity"]}
    r = rec[rec.id == rid].iloc[0]
    D = sp.load_recording(a.root, rid, (float(r.whisper_start), float(r.whisper_end)))
    if isinstance(D, str):
        return [(n, {**base, "load_status": D}) for n in a.configs]
    d = rio.find_recording_dir(a.root, rid)
    if not os.path.exists(os.path.join(d, "mps", "hand_tracking", "hand_tracking_frames.jsonl")):
        return [(n, {**base, "load_status": "no_hand"}) for n in a.configs]
    fr = hr.read_frames(d)
    side, G = hr.primary_side(fr, D["t"], D["walk"])
    other = "right" if side == "left" else "left"
    cov = {s: float((~G[s]["invalid"])[D["walk"]].mean()) if D["walk"].any() else np.nan for s in G}
    base.update(load_status="loaded", gaze_kind=D["gaze_kind"], primary_side=side, cov_primary=cov[side],
                cov_other=cov[other], walk_s=float(D["walk"].sum() / sp.FS))
    out = []
    for name in a.configs:
        src, mode, dep, thr, nsh, world = CONFIGS[name]
        Gx = G[side] if src == "primary" else (G[other] if src == "other" else hr.midpoint(G["left"], G["right"]))
        depth = hr.depth_vergence(D, Gx) if dep == "vergence" else None
        res = hr.measure_hand(D, Gx, thr, 2, mode, depth, n_shift=nsh, world=world, joint=True)
        if name == "main":   # recording-specific missing-data calibration (no recorded kinematics used)
            for f in hr.CAL_FREQS_REC:
                cr = hr.mask_calibration(D, fr, side, f, thr, 2)
                res[f"cal_ba_{f}"], res[f"cal_ok_{f}"] = cr["b_minus_a"], cr["ok"]
        out.append((name, {**base, "hand_source": src, **res}))
    return out


rids = c["rec_ids"][:a.limit] if a.limit else c["rec_ids"]
parts = Parallel(n_jobs=a.n_jobs)(delayed(one)(r) for r in rids)
os.makedirs(a.outdir, exist_ok=True)
for name in a.configs:
    df = pd.DataFrame([f for p in parts for (n, f) in p if n == name])
    df.to_csv(os.path.join(a.outdir, f"m3t_measures_{name}.csv"), index=False)
    print(name, len(df), "recordings", flush=True)
