"""M3 recording-level measures for every pre-registered measurement configuration.
Each recording is loaded once; all configurations run on it. Output: results/m3_measures_<cfg>.csv
--smoke_task16 runs the same code on task16 non-reading walking (pipeline smoke test only)."""
import argparse, json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa import io as rio
from ritw_kappa import slowphase as sp

CONFIGS = {  # name: (thr_deg, dil, depth_fill, min_s override)
    "main": (40.0, 2, None, None),
    "thr30": (30.0, 2, None, None), "thr60": (60.0, 2, None, None),
    "dil1": (40.0, 1, None, None), "dil4": (40.0, 4, None, None),
    "trans0.6": (40.0, 2, 0.6, None), "trans5": (40.0, 2, 5.0, None),
    "min20": (40.0, 2, None, {"gait": 20.0}),
}

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/seattle")
ap.add_argument("--cohort", default="results/m3_cohort_frozen.json")
ap.add_argument("--smoke_task16", action="store_true")
ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
ap.add_argument("--n_jobs", type=int, default=10)
ap.add_argument("--outdir", default="results")
a = ap.parse_args()
rec, t16 = rio.load_annotations()

if a.smoke_task16:
    # task16: whole recording, annotated reading intervals removed from the walking mask
    rids = sorted(t16.id.unique())
    tag = "smoke_task16_"
else:
    with open(a.cohort) as fh:
        c = json.load(fh)
    rids = c["rec_ids"]
    tag = ""
    meta = {r["rid"]: r for r in c["recordings"]}


# declared schema of the identifier / status columns: written even when the cohort is empty
SCHEMA = ["rid", "task", "pid", "pid_date", "activity", "load_status", "gaze_kind", "gait_ok", "gait_status",
          "gait_fit_status", "slow_ok", "slow_status", "slow_fit_status"]


def failed(rid, why):
    row = {"rid": int(rid), "load_status": why, "gait_ok": False, "slow_ok": False,
           "gait_status": why, "slow_status": why}
    return [(name, dict(row)) for name in a.configs]


def one(rid):
    if a.smoke_task16:
        D = sp.load_recording(a.root, rid)
        if isinstance(D, str):
            return failed(rid, D)
        rd = np.zeros(len(D["t"]), bool)
        for s, f in rio.reading_intervals(rid, rec, t16):
            rd |= (D["t"] - D["t0"] >= s) & (D["t"] - D["t0"] <= f)
        D["walk"] = D["walk"] & ~rd
    else:
        r = rec[rec.id == rid].iloc[0]
        D = sp.load_recording(a.root, rid, (float(r.whisper_start), float(r.whisper_end)))
        if isinstance(D, str):
            return failed(rid, D)
    out = []
    for name in a.configs:
        thr, dil, dep, mins = CONFIGS[name]
        m = sp.measure(D, thr, dil, dep, mins)
        flat = {"load_status": "loaded", **{k: v for k, v in m.items() if not isinstance(v, dict)}}
        for b in ("gait", "slow"):
            for k, v in m[b].items():
                flat[f"{b}_{k}"] = v
        out.append((name, flat))
    return out


parts = Parallel(n_jobs=a.n_jobs)(delayed(one)(r) for r in rids)
os.makedirs(a.outdir, exist_ok=True)
for name in a.configs:
    rows = [f for p in parts for (n, f) in p if n == name]
    df = pd.DataFrame(rows)
    df = df.reindex(columns=list(dict.fromkeys(SCHEMA + list(df.columns))))
    if not a.smoke_task16 and len(df):
        df["task"] = df.rid.map(lambda r: meta[r]["task"])
        df["pid"] = df.rid.map(lambda r: meta[r]["pid"])
        df["pid_date"] = df.rid.map(lambda r: meta[r]["pid_date"])
        df["activity"] = df.rid.map(lambda r: meta[r]["activity"])
    df.to_csv(os.path.join(a.outdir, f"m3_measures_{tag}{name}.csv"), index=False)
    print(name, len(df), "recordings", flush=True)
