"""M3 cohort + feasibility gate 1 (EXPERIMENT_PLAN_M3.md). Uses ONLY metadata and trajectories; the gaze
CSV is opened for its timestamp column alone (the task window is defined relative to the first gaze
sample). No gaze value is read. Output: results/m3_cohort_frozen.json"""
import argparse, ast, json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa import io as rio
from ritw_kappa.slowphase import BANDS, FS, runs, walking_mask

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/raw/seattle")
ap.add_argument("--min_walk_s", type=float, default=30.0)
ap.add_argument("--n_jobs", type=int, default=10)
ap.add_argument("--out", default="results/m3_cohort_frozen.json")
a = ap.parse_args()

rec, _ = rio.load_annotations(grouping="date_age")
rec_d, _ = rio.load_annotations(grouping="date")
pid_date = dict(zip(rec_d.id, rec_d.pid))
sub = rec[rec.task.isin([11, 12])].copy()


def tags(s):
    act, unint = None, None
    for q, v in ast.literal_eval(s):
        if "unintentionally" in q:
            unint = v[0] if v else None
        elif q == "":
            act = v[0] if v else None
    return act, unint


def one(row):
    d = rio.find_recording_dir(a.root, row.id)
    out = {"rid": int(row.id), "task": int(row.task), "pid": row.pid, "pid_date": pid_date[row.id]}
    out["activity"], out["unintentional_read"] = tags(row.tags) if isinstance(row.tags, str) else (None, None)
    if d is None:
        return {**out, "status": "no_dir"}
    gp, gk = rio.gaze_path(d)
    tp, tk = rio.trajectory_path(d)
    out.update(gaze_kind=gk, traj_kind=tk)
    if gp is None or tp is None:
        return {**out, "status": "no_gaze_or_traj"}
    tg = pd.read_csv(gp, usecols=["tracking_timestamp_us"]).tracking_timestamp_us.to_numpy() * 1e-6
    tg.sort()
    T = rio.load_trajectory(tp, tk)
    tt = T["t_us"] * 1e-6
    lo = max(tg[0], tt[0], tg[0] + float(row.whisper_start))
    hi = min(tg[-1], tt[-1], tg[0] + float(row.whisper_end))
    if hi - lo < 30:
        return {**out, "status": "short_window", "window_s": float(hi - lo)}
    t = np.arange(lo, hi, 1 / FS)
    v, up = T["v"], T["up"]
    vh = np.linalg.norm(v - np.einsum("ni,ni->n", v, up)[:, None] * up, axis=1)
    walk = walking_mask(np.interp(t, tt, vh))
    # usable walking for the gait band: walking bouts >= min_seg_s, minus the trims at both ends
    G = BANDS["gait"]
    usable = sum(b - a - 2 * G["trim_s"] * FS for a, b in runs(walk) if b - a >= G["min_seg_s"] * FS) / FS
    out.update(window_s=float(hi - lo), walk_s=float(walk.sum() / FS), walk_usable_s=float(usable),
               traj_gap_max_s=float(np.max(np.diff(tt[(tt >= lo) & (tt <= hi)]), initial=0)))
    ok = out["walk_usable_s"] >= a.min_walk_s and not (row.task == 11 and out["unintentional_read"] == "Yes")
    return {**out, "status": "included" if ok else "excluded_walk"}


R = Parallel(n_jobs=a.n_jobs)(delayed(one)(r) for r in sub.itertuples())
df = pd.DataFrame(R)
inc = df[df.status == "included"]
paired = sorted(set(inc[inc.task == 12].pid) & set(inc[inc.task == 11].pid))
paired_d = sorted(set(inc[inc.task == 12].pid_date) & set(inc[inc.task == 11].pid_date))
gate = {"paired_clusters": len(paired), "n_R_included": int((inc.task == 12).sum()),
        "pass_paired": bool(len(paired) >= 10 and (inc.task == 12).sum() >= 15)}
clusters_R = inc[inc.task == 12].pid.nunique()
clusters_N = inc[inc.task == 11].pid.nunique()
gate["primary_design"] = "paired" if gate["pass_paired"] else (
    "unpaired" if min(clusters_R, clusters_N) >= 10 else "unsupported")
summ = {"root": a.root, "min_walk_s": a.min_walk_s, "grouping": "date_age",
        "counts": df.groupby(["task", "status"]).size().rename("n").reset_index().to_dict("records"),
        "walk_s_median_by_task": inc.groupby("task").walk_s.median().to_dict(),
        "walk_usable_s_median_by_task": inc.groupby("task").walk_usable_s.median().to_dict(),
        "clusters_R": int(clusters_R), "clusters_N": int(clusters_N),
        "paired_clusters_date": len(paired_d), "gate": gate, "paired_pids": paired,
        "rec_ids": inc.rid.tolist(), "recordings": df.to_dict("records")}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(summ, open(a.out, "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print(json.dumps({k: summ[k] for k in ("counts", "walk_s_median_by_task", "clusters_R", "clusters_N",
                                       "paired_clusters_date", "gate")}, indent=1, default=str))
