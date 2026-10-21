"""Loaders for RitW annotations and Aria MPS eye-gaze / SLAM trajectory CSVs.

Conventions (from facebookresearch/reading_in_the_wild):
* task16 `start k`/`finish k`/`calib`/`end` and ritw `whisper_start/end` are seconds
  relative to the first eye-gaze sample (`tracking_timestamp_us.iloc[0]`).
* personalized gaze is preferred, general gaze is the fallback.
* `cpf_transform` (projection_utils.get_default_camera_config) maps device-frame points
  into CPF: it is used as `T_cpf @ head_pose_diff_device @ inv(T_cpf)` in
  GazeRays.head_correction, so R_CPF_device = cpf_transform[:3, :3].
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ANN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ritw_repo", "reading_in_the_wild_seattle")

# Default T_CPF_device from the official repo (projection_utils.get_default_camera_config).
DEFAULT_T_CPF_DEVICE = np.array([
    [-0.03186013725398906, 0.7933534010715845, 0.6079270619591683, 0.0702907945],
    [-0.9986295320622568, 1.7481548431064198e-09, -0.05233600761538715, 0.002372],
    [-0.04152095070292633, -0.608761349798219, 0.79226613561642, 0.00882276712],
    [0.0, 0.0, 0.0, 1.0],
])

def load_annotations(ann_dir: str = ANN_DIR, grouping: str = "date_age") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (recordings, task16) with a derived participant cluster id `pid`.

    RitW releases no participant id. Clusters are built only by MERGING, never by splitting:
      date_age (primary): one cluster per (date, age bracket) -- a person keeps one age bracket
                          within a recording day, so their same-day recordings stay together;
                          two people sharing date+bracket are merged (conservative).
      date (sensitivity): one cluster per recording date (strictly coarser).
    Neither rule can join one person's recordings made on different dates; this is reported
    as a limitation, and the analysis is described as participant-cluster (not participant) LOPO.
    """
    rec = pd.read_csv(os.path.join(ann_dir, "ritw_annotations.csv"))
    keys = ["date", "age"] if grouping == "date_age" else ["date"]
    rec = rec.sort_values(keys + ["start_time"]).reset_index(drop=True)
    rec["pid"] = "P" + (rec.groupby(keys, sort=True).ngroup() + 1).astype(str).str.zfill(3)
    t16 = pd.read_csv(os.path.join(ann_dir, "task16_annotations.csv"))
    t16 = t16.rename(columns={"notes": "scenario"})
    t16 = t16.merge(rec[["id", "pid", "date", "split"]], on="id", how="left")
    return rec, t16


def reading_intervals(rec_id: int, rec: pd.DataFrame, t16: pd.DataFrame) -> list[tuple[float, float]]:
    """Annotated reading intervals (s, relative to first gaze sample)."""
    row16 = t16[t16.id == rec_id]
    if len(row16):
        r = row16.iloc[0]
        out = []
        for k in range(1, 11):
            s, f = r.get(f"start {k}"), r.get(f"finish {k}")
            if pd.notna(s) and pd.notna(f):
                out.append((float(s), float(f)))
        return out
    r = rec[rec.id == rec_id]
    if len(r) and bool(r.iloc[0]["reading"]) and pd.notna(r.iloc[0]["whisper_start"]):
        return [(float(r.iloc[0]["whisper_start"]), float(r.iloc[0]["whisper_end"]))]
    return []


def find_recording_dir(root: str, rec_id) -> str | None:
    cand = os.path.join(root, str(rec_id))
    if os.path.isdir(cand):
        return cand
    hits = glob.glob(os.path.join(root, "**", str(rec_id)), recursive=True)
    hits = [h for h in hits if os.path.isdir(h)]
    return hits[0] if hits else None


def gaze_path(rec_dir: str) -> tuple[str | None, str]:
    for kind in ("personalized", "general"):
        p = os.path.join(rec_dir, "mps", "eye_gaze", f"{kind}_eye_gaze.csv")
        if os.path.exists(p):
            return p, kind
    return None, "none"


def trajectory_path(rec_dir: str) -> tuple[str | None, str]:
    for kind in ("open_loop", "closed_loop"):
        p = os.path.join(rec_dir, "mps", "slam", f"{kind}_trajectory.csv")
        if os.path.exists(p):
            return p, kind
    return None, "none"


def gaze_dirs_cpf(df: pd.DataFrame) -> np.ndarray:
    """Unit gaze direction in CPF, vectorised version of the official conversion.

    Binocular (left/right yaw): intersect the two rays (IPD 63 mm), y from pitch;
    otherwise monocular x=tan(yaw), y=tan(pitch), z=1.
    """
    if "left_yaw_rads_cpf" in df.columns:
        ipd = 63.0
        tl = np.tan(df.left_yaw_rads_cpf.to_numpy(float))
        tr = np.tan(df.right_yaw_rads_cpf.to_numpy(float))
        d = np.maximum(tr - tl, 1e-6)
        z = ipd / d
        x = 0.5 * ipd * (tl + tr) / d
        y = z * np.tan(df.pitch_rads_cpf.to_numpy(float))
    else:
        x = np.tan(df.yaw_rads_cpf.to_numpy(float))
        y = np.tan(df.pitch_rads_cpf.to_numpy(float))
        z = np.ones_like(x)
    v = np.stack([x, y, z], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def load_gaze(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="c")
    df = df.sort_values("tracking_timestamp_us").drop_duplicates("tracking_timestamp_us").reset_index(drop=True)
    g = gaze_dirs_cpf(df)
    out = pd.DataFrame({"t_us": df.tracking_timestamp_us.to_numpy(np.int64),
                        "gx": g[:, 0], "gy": g[:, 1], "gz": g[:, 2]})
    out["depth"] = df["depth_m"].to_numpy(float) if "depth_m" in df.columns else np.nan
    if "left_yaw_rads_cpf" in df.columns:
        out["vergence"] = (df.right_yaw_rads_cpf - df.left_yaw_rads_cpf).to_numpy(float)
        widths = [df[f"{c}_high_rads_cpf"] - df[f"{c}_low_rads_cpf"] for c in ("left_yaw", "right_yaw", "pitch")
                  if f"{c}_high_rads_cpf" in df.columns]
    else:
        out["vergence"] = np.nan
        widths = [df[f"{c}_high_rads_cpf"] - df[f"{c}_low_rads_cpf"] for c in ("yaw", "pitch")
                  if f"{c}_high_rads_cpf" in df.columns]
    out["q_width"] = np.mean(np.stack([w.to_numpy(float) for w in widths], 1), 1) if widths else np.nan
    out["pitch_g"] = df["pitch_rads_cpf"].to_numpy(float)
    out["yaw_g"] = np.arctan2(out.gx, out.gz)
    return out


def _quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.empty((len(qx), 3, 3))
    R[:, 0, 0] = 1 - 2 * (qy**2 + qz**2); R[:, 0, 1] = 2 * (qx*qy - qz*qw); R[:, 0, 2] = 2 * (qx*qz + qy*qw)
    R[:, 1, 0] = 2 * (qx*qy + qz*qw); R[:, 1, 1] = 1 - 2 * (qx**2 + qz**2); R[:, 1, 2] = 2 * (qy*qz - qx*qw)
    R[:, 2, 0] = 2 * (qx*qz - qy*qw); R[:, 2, 1] = 2 * (qy*qz + qx*qw); R[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)
    return R


def load_trajectory(path: str, kind: str) -> pd.DataFrame:
    """Return t_us, R_world_device (N,3,3), v_world (N,3), omega_device (N,3), up_world (N,3)."""
    frame = "odometry" if kind == "open_loop" else "world"
    df = pd.read_csv(path, engine="c")
    df = df.sort_values("tracking_timestamp_us").drop_duplicates("tracking_timestamp_us").reset_index(drop=True)
    R = _quat_to_R(*(df[f"q{a}_{frame}_device"].to_numpy(float) for a in "xyzw"))
    if f"device_linear_velocity_x_{frame}" in df.columns:
        v = np.stack([df[f"device_linear_velocity_{a}_{frame}"].to_numpy(float) for a in "xyz"], 1)
    else:  # closed loop reports velocity in the device frame
        vd = np.stack([df[f"device_linear_velocity_{a}_device"].to_numpy(float) for a in "xyz"], 1)
        v = np.einsum("nij,nj->ni", R, vd)
    w = np.stack([df[f"angular_velocity_{a}_device"].to_numpy(float) for a in "xyz"], 1)
    gcols = [f"gravity_{a}_{frame}" for a in "xyz"]
    if all(c in df.columns for c in gcols):
        grav = df[gcols].to_numpy(float)
        up = -grav / np.linalg.norm(grav, axis=1, keepdims=True)
    else:
        up = np.tile([0.0, 0.0, 1.0], (len(df), 1))
    return {"t_us": df.tracking_timestamp_us.to_numpy(np.int64), "R": R, "v": v, "w": w, "up": up, "kind": kind}
