"""Frame check through the real loaders: write MPS-format gaze + open-loop CSVs for a head rotating
in place while the eyes hold a far world-fixed target; load via io + build_session.
Expected: kappa ~= 1 with the default T_CPF_device, clearly off with identity T; turn rate ~ imposed yaw rate."""
import argparse, json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from ritw_kappa import io as rio
from ritw_kappa.signal import build_session, window_stats, residual_rotation
from ritw_kappa.features import FoldKappa

ap = argparse.ArgumentParser()
ap.add_argument("--dur", type=float, default=60.0)
ap.add_argument("--target_m", type=float, default=50.0)
ap.add_argument("--yaw_rate", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/check_frames.json")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

T = rio.DEFAULT_T_CPF_DEVICE
R_cd, t_cd = T[:3, :3], T[:3, 3]
fs_tr = 1000
tt = np.arange(int(a.dur * fs_tr)) / fs_tr
f = 1.9
ph = rng.uniform(0, 2 * np.pi, 3)
w_cpf = np.stack([0.5 * np.sin(2 * np.pi * f * tt + ph[0]), 0.3 * np.sin(np.pi * f * tt + ph[1]),
                  0.2 * np.sin(np.pi * f * tt + ph[2])], 1)
w_dev = w_cpf @ R_cd                                    # CPF -> device (R_cd^T applied row-wise)
# world: z up. Start with CPF forward (z_cpf) along world +x, level; add constant yaw about world up.
R0_wc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # columns: CPF x,y,z in world
R_wd = np.empty((len(tt), 3, 3)); R_wd[0] = R0_wc @ R_cd       # world <- device = world<-CPF * CPF<-device
up = np.array([0, 0, 1.0])
w_dev_tot = np.empty_like(w_dev)
for i in range(len(tt)):
    wy = R_wd[i].T @ (a.yaw_rate * up)                  # world yaw expressed in device frame
    w_dev_tot[i] = w_dev[i] + wy
    if i + 1 < len(tt):
        R_wd[i + 1] = R_wd[i] @ Rot.from_rotvec(w_dev_tot[i] / fs_tr).as_matrix()
q = Rot.from_matrix(R_wd).as_quat()                     # x y z w
traj = pd.DataFrame({"tracking_timestamp_us": (tt * 1e6).astype(np.int64) + 1_000_000,
                     "tx_odometry_device": 0.0, "ty_odometry_device": 0.0, "tz_odometry_device": 0.0,
                     "qx_odometry_device": q[:, 0], "qy_odometry_device": q[:, 1],
                     "qz_odometry_device": q[:, 2], "qw_odometry_device": q[:, 3],
                     "device_linear_velocity_x_odometry": 0.0, "device_linear_velocity_y_odometry": 0.0,
                     "device_linear_velocity_z_odometry": 0.0,
                     "angular_velocity_x_device": w_dev_tot[:, 0], "angular_velocity_y_device": w_dev_tot[:, 1],
                     "angular_velocity_z_device": w_dev_tot[:, 2],
                     "gravity_x_odometry": 0.0, "gravity_y_odometry": 0.0, "gravity_z_odometry": -9.81})
# eyes fixate a far world point that keeps up with the slow yaw only via saccade-free tracking: target
# is world-fixed for gait rotation, so we place it along the yaw-only heading (no gait) -> pursuit at yaw rate
tg = np.arange(int(a.dur * 60)) / 60.0
k = np.round(tg * fs_tr).astype(int)
heading = a.yaw_rate * tg
p_w = a.target_m * np.stack([np.cos(heading), np.sin(heading), np.zeros_like(tg)], 1)
p_dev = np.einsum("nji,nj->ni", R_wd[k], p_w)           # device origin at world origin
p_cpf = p_dev @ R_cd.T + t_cd
X, Y, Z = p_cpf.T
ipd = 0.063
gaze = pd.DataFrame({"tracking_timestamp_us": (tg * 1e6).astype(np.int64) + 1_000_000,
                     "left_yaw_rads_cpf": np.arctan((X - ipd / 2) / Z), "right_yaw_rads_cpf": np.arctan((X + ipd / 2) / Z),
                     "pitch_rads_cpf": np.arctan(Y / Z), "depth_m": np.linalg.norm(p_cpf, axis=1)})
for c in ("left_yaw", "right_yaw", "pitch"):
    gaze[f"{c}_low_rads_cpf"] = gaze[f"{c}_rads_cpf"] - 0.01
    gaze[f"{c}_high_rads_cpf"] = gaze[f"{c}_rads_cpf"] + 0.01

res = {}
with tempfile.TemporaryDirectory() as d:
    gp, tp = os.path.join(d, "gaze.csv"), os.path.join(d, "open_loop_trajectory.csv")
    gaze.to_csv(gp, index=False); traj.to_csv(tp, index=False)
    G, TR = rio.load_gaze(gp), rio.load_trajectory(tp, "open_loop")
    Tmis = T.copy(); Tmis[:3, :3] = Rot.from_rotvec(np.deg2rad(8.0) * np.array([0.6, 0.0, 0.8])).as_matrix() @ T[:3, :3]
    for name, Tm in (("default_T", T), ("identity_T", np.eye(4)), ("misaligned8deg_T", Tmis)):
        s = build_session(G, TR, Tm)
        ws = window_stats(s, 2.0)
        sl = slice(300, None)
        kap = ws["S"][sl] / ws["P"][sl]
        res[name] = {"kappa_median": float(np.nanmedian(kap)), "kappa_iqr": [float(x) for x in np.nanpercentile(kap, [25, 75])],
                     "residual_rotation": residual_rotation(s)}
        if name == "default_T":
            res["turn_rate_median"] = float(np.nanmedian(s["turn"][sl]))
            res["imposed_yaw_rate"] = a.yaw_rate
            res["depth_median"] = float(np.nanmedian(s["depth"]))
            res["gaze_g_vs_true_max_angle_deg"] = float(np.degrees(np.nanmax(np.arccos(np.clip(
                np.sum(s["g"][: len(p_cpf)] * (p_cpf / np.linalg.norm(p_cpf, axis=1, keepdims=True))[: len(s['g'])], 1), -1, 1)))))
res["pass"] = bool(abs(res["default_T"]["kappa_median"] - 1) < 0.05 and abs(res["identity_T"]["kappa_median"] - 1) > 0.2
                   and abs(res["turn_rate_median"] - a.yaw_rate) < 0.02
                   and res["default_T"]["residual_rotation"]["angle_deg"] < 1.0
                   and abs(res["misaligned8deg_T"]["residual_rotation"]["angle_deg"] - 8.0) < 1.0)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
