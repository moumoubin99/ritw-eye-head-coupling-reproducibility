"""M3-T pre-analysis check (no gaze outcome): hand-tracking frame convention, time base and coverage
during walking inside the frozen M3 window. Uses only the trajectory-derived walking mask and grid.
Coverage = fraction of walking grid samples on which handref.side_on_grid marks the hand valid, i.e. the
same measurement-validity rule the analysis uses (confidence >= CONF_THR, native runs split at timestamp
gaps, run end samples excluded)."""
import json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa import io as rio
from ritw_kappa import slowphase as sp
from ritw_kappa import handref as hr

ROOT = "data/raw/seattle"
Rd, tc = rio.DEFAULT_T_CPF_DEVICE[:3, :3], rio.DEFAULT_T_CPF_DEVICE[:3, 3]
c = json.load(open("results/m3_cohort_frozen.json"))
meta = {r["rid"]: r for r in c["recordings"]}
rec, _ = rio.load_annotations()


def hands(rid):
    f = os.path.join(ROOT, str(rid), "mps/hand_tracking/hand_tracking_frames.jsonl")
    ts, conf, pos = [], {"left": [], "right": []}, {"left": [], "right": []}
    with open(f) as fh:
        for l in fh:
            o = json.loads(l)
            ts.append(o["tracking_timestamp_us"])
            for s in ("left", "right"):
                h = o["hand_poses"][s]
                conf[s].append(h["existence_confidence"])
                pos[s].append(h["T_wrist_device"]["translation"])
    t = np.asarray(ts) * 1e-6
    return t, {s: np.asarray(conf[s], float) for s in conf}, {s: np.asarray(pos[s], float) / 1000.0 @ Rd.T + tc for s in pos}


def one(rid):
    r = rec[rec.id == rid].iloc[0]
    D = sp.load_recording(ROOT, rid, (float(r.whisper_start), float(r.whisper_end)))
    if isinstance(D, str):
        return {"rid": rid, "load": D}
    th, conf, pos = hands(rid)
    w = D["walk"]
    out = {"rid": rid, "task": meta[rid]["task"], "pid": meta[rid]["pid"], "pid_date": meta[rid]["pid_date"], "load": "ok",
           "hand_rate": float(1 / np.median(np.diff(th))), "hand_t_lo_minus_grid": float(th[0] - D["t"][0]),
           "hand_t_hi_minus_grid": float(th[-1] - D["t"][-1]), "walk_s": float(w.sum() / sp.FS)}
    fr = hr.read_frames(os.path.join(ROOT, str(rid)))
    G = {s: hr.side_on_grid(fr, s, D["t"]) for s in ("left", "right")}
    for s in ("left", "right"):
        ok = ~G[s]["invalid"]
        out[f"{s}_cov0.5"] = float(ok[w].mean()) if w.any() else np.nan
        p = G[s]["p"][ok & w]
        out[f"{s}_xyz_med"] = np.round(np.median(p, 0), 3).tolist() if len(p) else None
        out[f"{s}_dist_med"] = float(np.median(np.linalg.norm(p, axis=1))) if len(p) else np.nan
    either = ~G["left"]["invalid"] | ~G["right"]["invalid"]
    out["either_cov0.5"] = float(either[w].mean()) if w.any() else np.nan
    out["both_cov0.5"] = float((~G["left"]["invalid"] & ~G["right"]["invalid"])[w].mean()) if w.any() else np.nan
    return out


rows = Parallel(n_jobs=10)(delayed(one)(r) for r in c["rec_ids"])
df = pd.DataFrame(rows)
df.to_csv("results/m3t/hand_coverage.csv", index=False)
ok = df[df.load == "ok"].copy()
ok["best_cov"] = ok[["left_cov0.5", "right_cov0.5"]].max(1)
summ = {}
for t, g in ok.groupby("task"):
    summ[f"task{t}"] = {"n": int(len(g)), "hand_rate_med": float(g.hand_rate.median()),
                        "t_lo_offset_med_s": float(g.hand_t_lo_minus_grid.median()),
                        "t_hi_offset_med_s": float(g.hand_t_hi_minus_grid.median()),
                        **{k: [float(np.nanquantile(g[k], q)) for q in (0.1, 0.25, 0.5, 0.75)]
                           for k in ("left_cov0.5", "right_cov0.5", "best_cov", "either_cov0.5", "both_cov0.5", "left_dist_med", "right_dist_med")},
                        "n_best_cov_ge_0.5": int((g.best_cov >= 0.5).sum()), "n_best_cov_ge_0.7": int((g.best_cov >= 0.7).sum()),
                        "clusters_pid_date_best_cov_ge_0.5": int(g[g.best_cov >= 0.5].pid_date.nunique()),
                        "clusters_pid_best_cov_ge_0.5": int(g[g.best_cov >= 0.5].pid.nunique()),
                        "n_both_cov_ge_0.5": int((g["both_cov0.5"] >= 0.5).sum())}
json.dump(summ, open("results/m3t/hand_coverage_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print(ok.groupby("task")[["left_xyz_med", "right_xyz_med"]].head(3).to_string())
