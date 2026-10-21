"""M3-T (refine-logs/EXPERIMENT_PLAN_M3T.md): hand-referenced (target-relative) slow-phase prediction.

All vectors in CPF (X left, Y up, Z fwd), metres. The MPS hand-tracking field T_wrist_device stores the
wrist position in the device frame (millimetres) and the wrist orientation as an xyzw quaternion of the
wrist->device rotation (pre-analysis check, results/m3t/hand_coverage_summary.json and plan section 2).
The pose is expressed in the head-fixed device frame, so its time derivative is already the motion of
the hand relative to the head, i.e. what the eye-in-head must follow to keep a hand-held target foveated.

  p       wrist position relative to the CPF origin
  v       d p / dt (relative to the head, CPF axes)
  w_h     wrist angular velocity relative to the head (CPF axes; rigid sensitivity only)
  target  P = d * g, d = |p| (primary) or vergence depth (sensitivity)
  vP      v                       (trans: target moves with the wrist point; primary)
          v + w_h x (P - p)       (rigid: target rigidly attached to the wrist frame; sensitivity)
  gpred_H (I - g g^T) vP / d      (kappa_H = 1 <=> gaze fully follows the hand-held target)

The frozen estimator slowphase.band_gains builds gpred = -w x g from its angular-velocity argument. For any
vector q perpendicular to g, w = q x g gives -w x g = g x (q x g) = q, so band_gains(..., w_equiv(gpred_H, g))
runs the frozen M3 estimator unchanged with gpred = gpred_H."""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation as Rot, RotationSpline

from . import io as rio
from . import slowphase as sp

CONF_THR = 0.5      # existence_confidence >= CONF_THR marks a valid hand frame
GAP_FACTOR = 1.5    # a native timestamp step > GAP_FACTOR x median interval splits a run (missing frames)
MIN_RUN = 4         # minimum native frames in a run (cubic spline)
DEPTH_CLIP = (0.2, 1.5)
Rd = rio.DEFAULT_T_CPF_DEVICE[:3, :3]
TC = rio.DEFAULT_T_CPF_DEVICE[:3, 3]


def read_frames(rec_dir: str) -> dict:
    """Native hand frames: t (s), and per side conf, p_dev (m), q (xyzw)."""
    f = os.path.join(rec_dir, "mps", "hand_tracking", "hand_tracking_frames.jsonl")
    ts, out = [], {s: {"conf": [], "p": [], "q": []} for s in ("left", "right")}
    with open(f) as fh:
        for line in fh:
            o = json.loads(line)
            ts.append(o["tracking_timestamp_us"])
            for s in ("left", "right"):
                h = o["hand_poses"][s]
                out[s]["conf"].append(h["existence_confidence"])
                out[s]["p"].append(h["T_wrist_device"]["translation"])
                out[s]["q"].append(h["T_wrist_device"]["quaternion"])
    t = np.asarray(ts, float) * 1e-6
    o = np.argsort(t, kind="stable")
    res = {"t": t[o]}
    for s in out:
        res[s] = {"conf": np.asarray(out[s]["conf"], float)[o], "p_dev": np.asarray(out[s]["p"], float)[o] / 1000.0,
                  "q": np.asarray(out[s]["q"], float)[o]}
    return res


def native_runs(th: np.ndarray, ok: np.ndarray) -> list[tuple[int, int]]:
    """Runs of valid native frames, split at confidence failures and at timestamp steps > GAP_FACTOR x the
    median native interval (missing frames), keeping runs of >= MIN_RUN frames."""
    med = float(np.median(np.diff(th)))
    brk = np.r_[False, np.diff(th) > GAP_FACTOR * med]
    out = []
    for a, b in sp.runs(ok):
        cuts = [a] + [i for i in range(a + 1, b) if brk[i]] + [b]
        out += [(c0, c1) for c0, c1 in zip(cuts[:-1], cuts[1:]) if c1 - c0 >= MIN_RUN]
    return out


def side_on_grid(fr: dict, side: str, t: np.ndarray) -> dict:
    """Wrist position / velocity / angular velocity of one hand on the analysis grid t (CPF).
    Within each run of valid native frames (native_runs) the position is a cubic spline of the native frames
    and the orientation a rotation spline, both evaluated on the grid; velocities are then taken on the grid
    with the same central difference the loader applies to gaze (np.gradient on t), so the hand and gaze
    derivatives share one differentiation operator. Grid samples outside a run, and the first / last grid
    sample of each run (one-sided difference), are invalid."""
    th = fr["t"]
    H = fr[side]
    ok = H["conf"] >= CONF_THR
    p = H["p_dev"] @ Rd.T + TC
    n = len(t)
    P = np.full((n, 3), np.nan)
    V = np.zeros((n, 3))
    W = np.zeros((n, 3))
    valid = np.zeros(n, bool)
    for a, b in native_runs(th, ok):
        i0, i1 = np.searchsorted(t, th[a], "left"), np.searchsorted(t, th[b - 1], "right")
        if i1 - i0 < 3:
            continue
        tg = t[i0:i1]
        P[i0:i1] = CubicSpline(th[a:b], p[a:b], axis=0)(tg)
        V[i0:i1] = np.gradient(P[i0:i1], tg, axis=0)
        Rg = RotationSpline(th[a:b], Rot.from_quat(H["q"][a:b]))(tg)            # wrist -> device
        rv = (Rg[2:] * Rg[:-2].inv()).as_rotvec() / (tg[2:] - tg[:-2])[:, None]  # spatial, device frame
        W[i0 + 1:i1 - 1] = rv @ Rd.T
        valid[i0 + 1:i1 - 1] = True
    if valid.any():
        # continuity fill: invalid samples get values linearly interpolated between the bracketing valid
        # samples (position, velocity and angular velocity), so that the band-pass filters inside the
        # estimator do not see artificial zero motion in short gaps. Filled samples stay invalid (masked
        # from fitting and coverage); gaps longer than the estimator's max_gap_s still split its segments.
        idx = np.flatnonzero(valid)
        fill = lambda X: np.stack([np.interp(np.arange(n), idx, X[idx, k]) for k in range(3)], 1)
        P, V, W = fill(P), fill(V), fill(W)
    return {"p": P, "v": V, "w": W, "invalid": ~valid}


def primary_side(fr: dict, t: np.ndarray, walk: np.ndarray) -> tuple[str, dict]:
    """Primary hand = the hand with more valid walking grid samples (no gaze used)."""
    G = {s: side_on_grid(fr, s, t) for s in ("left", "right")}
    cov = {s: float((~G[s]["invalid"] & walk).sum()) for s in G}
    s = "left" if cov["left"] > cov["right"] else "right"
    return s, G


def midpoint(GL: dict, GR: dict) -> dict:
    return {"p": (GL["p"] + GR["p"]) / 2, "v": (GL["v"] + GR["v"]) / 2, "w": np.zeros_like(GL["v"]),
            "invalid": GL["invalid"] | GR["invalid"]}


def shift(G: dict, k: int) -> dict:
    """Circular time shift of the hand signals by k grid samples (surrogate control)."""
    return {key: np.roll(val, k, axis=0) for key, val in G.items()}


def gpred_hand(g: np.ndarray, G: dict, mode: str = "trans", depth: np.ndarray | None = None) -> np.ndarray:
    d = np.linalg.norm(G["p"], axis=1) if depth is None else depth
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    vP = G["v"]
    if mode == "rigid":
        vP = vP + np.cross(G["w"], d[:, None] * g - G["p"])
    q = (vP - np.einsum("ni,ni->n", vP, g)[:, None] * g) / d[:, None]
    return np.nan_to_num(q)


def w_equiv(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Angular-velocity argument for slowphase.band_gains such that -w x g = q (q perpendicular to g)."""
    return np.cross(q, g)


def depth_vergence(D: dict, G: dict) -> np.ndarray:
    """Vergence depth clipped to DEPTH_CLIP; |p| where missing."""
    d = np.linalg.norm(G["p"], axis=1)
    if D["depth"] is None:
        return d
    ok = np.isfinite(D["depth"])
    d = d.copy()
    d[ok] = np.clip(D["depth"][ok], *DEPTH_CLIP)
    return d


def hand_lambda(D: dict, G: dict, band: str = "gait") -> dict:
    """Gaze-free prediction lambda: gain of the hand-direction rate (I - u u^T) v / |p| on the world prediction
    -w_head x u (u = p / |p|), on walking segments split by hand gaps > 0.5 s, same band-pass and trimming as M3.
    Under target-relative stabilisation the world-referenced slow-phase gain should approach lambda."""
    B = sp.BANDS[band]
    sos = butter(2, [B["lo"], B["hi"]], "bandpass", fs=sp.FS, output="sos")
    u = G["p"] / np.linalg.norm(G["p"], axis=1, keepdims=True)
    u = np.nan_to_num(u)
    y = (G["v"] - np.einsum("ni,ni->n", G["v"], u)[:, None] * u) / np.linalg.norm(G["p"], axis=1)[:, None]
    y = np.nan_to_num(y)
    x = -np.cross(D["w_cpf"], u)
    _, gap = sp.short_runs(G["invalid"], 0.5)
    segs = [(a, b) for a, b in sp.runs(D["walk"] & ~gap) if b - a >= B["min_seg_s"] * sp.FS]
    tr = int(B["trim_s"] * sp.FS)
    S = np.zeros(len(y), bool)
    for a, b in segs:
        S[a + tr:b - tr] = True
    S &= ~G["invalid"]
    if S.sum() < B["min_s"] * sp.FS:
        return {"lambda": np.nan, "lambda_n_s": float(S.sum() / sp.FS), "rho_HW": np.nan}
    yb, xb = sp._bp_segments(sos, y, segs), sp._bp_segments(sos, x, segs)
    # rho_HW: uncentred gait-band correlation of the two gaze-free proxies (descriptive only; the joint-fit
    # collinearity diagnostic is joint_fit's rho on the actual design columns)
    rho = np.sum(yb[S] * xb[S]) / np.sqrt(np.sum(yb[S] ** 2) * np.sum(xb[S] ** 2))
    return {"lambda": float(np.sum(yb[S] * xb[S]) / np.sum(xb[S] ** 2)), "lambda_n_s": float(S.sum() / sp.FS),
            "rho_HW": float(rho)}


N_SHIFT, SHIFT_MARGIN_S = 24, 20.0   # circular-shift surrogates: N_SHIFT shifts evenly spaced in [margin, L - margin]
KEYS = ("ok", "status", "fit_status", "kappa_sp", "kappa_raw", "c", "fast_frac", "P", "n_Ssp_s", "start_used",
        "start1_status", "start0_status", "kappa_sp_cx", "kappa_sp_cy")


def _pick(r: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": r.get(k, np.nan) for k in KEYS}


def shift_samples(n: int, n_shift: int = N_SHIFT) -> list[int]:
    m = int(SHIFT_MARGIN_S * sp.FS)
    return [int(round(k)) for k in np.linspace(m, n - m, n_shift)] if n > 2 * m + n_shift else []


def joint_fit(D: dict, qH: np.ndarray, inv: np.ndarray, thr_deg=40.0, dil=2) -> dict:
    """Two-regressor fit (a = world rotation, b = hand) on the world-desaccaded slow-phase response, through the
    frozen estimator (v_cpf = -qH, depth = 1 gives its translation regressor = qH). The accepted world fit is
    re-derived with the frozen internals (same arguments, same start rule) to obtain the actual design columns
    on Ssp: rho = uncentred correlation of the two filtered columns, rank of the design. check = max abs
    difference between the re-derived and the returned (a, b). ok = world fit ok, finite a and b, rank 2."""
    t, g = D["t"], D["g"]
    r = sp.band_gains(t, g, D["gdot"], D["w_cpf"], D["walk"], inv, thr_deg, dil, bands=("gait",),
                      v_cpf=-qH, depth=np.ones(len(t)))["gait"]
    a, b = r.get("kappa_sp_rot_given_trans", np.nan), r.get("trans_coef", np.nan)
    res = {"a": a, "b": b, "rho": np.nan, "rank": 0, "check": np.nan}
    if r.get("fit_status") != "no_fit":
        B = sp.BANDS["gait"]
        sos = butter(2, [B["lo"], B["hi"]], "bandpass", fs=sp.FS, output="sos")
        gdet = savgol_filter(g, 5, 2, deriv=1, delta=1 / sp.FS, axis=0)
        gpred = -np.cross(D["w_cpf"], g)
        k0 = 0.0 if r["start_used"] == 0 else 1.0
        fit = sp._fit_sp(gdet, D["gdot"], gpred, inv, D["walk"], sos, B, thr_deg, dil, 0.5, k0, 50, 1e-3)
        st = fit["st"]
        h, Ssp = st["h"], st["Ssp"]
        htr = sp._bp_segments(sos, qH, st["segs"])
        X = np.stack([h[Ssp].ravel(), htr[Ssp].ravel()], 1)
        if len(X):
            res["rank"] = int(np.linalg.matrix_rank(X))
            res["rho"] = float(X[:, 0] @ X[:, 1] / np.sqrt((X[:, 0] @ X[:, 0]) * (X[:, 1] @ X[:, 1])))
            coef = np.linalg.lstsq(X, st["e_sp"][Ssp].ravel(), rcond=None)[0]
            res["check"] = float(np.nanmax(np.abs(coef - [a, b])))
    res["ok"] = bool(r.get("ok", False) and np.isfinite(a) and np.isfinite(b) and res["rank"] == 2)
    return res


def measure_hand(D: dict, G: dict, thr_deg=40.0, dil=2, mode="trans", depth=None, n_shift=N_SHIFT,
                 world=True, joint=True) -> dict:
    """World (W) and hand-referenced (H) slow-phase gains on the same validity mask (gaze | hand invalid),
    the joint two-regressor fit on the W-desaccaded response (a = world, b = hand), circular-shift
    surrogates of the hand signals, and the gaze-free lambda. Gait band only."""
    inv = D["invalid"] | G["invalid"]
    out = {"hand_valid_walk": float((~G["invalid"])[D["walk"]].mean()) if D["walk"].any() else np.nan}
    qH = gpred_hand(D["g"], G, mode, depth)
    rH = sp.band_gains(D["t"], D["g"], D["gdot"], w_equiv(qH, D["g"]), D["walk"], inv, thr_deg, dil, bands=("gait",))
    out.update(_pick(rH["gait"], "H"))
    if world:
        rW = sp.band_gains(D["t"], D["g"], D["gdot"], D["w_cpf"], D["walk"], inv, thr_deg, dil, bands=("gait",))
        out.update(_pick(rW["gait"], "W"))
    if joint:
        J = joint_fit(D, qH, inv, thr_deg, dil)
        out.update(J_ok=J["ok"], J_a_world=J["a"], J_b_hand=J["b"], J_rho=J["rho"], J_rank=J["rank"],
                   J_check=J["check"])
    # surrogate: the same estimator with the hand signals circularly shifted (alignment destroyed, hand
    # kinematics and their spectrum kept); summarised as the mean over ok shifts
    kH, bJ, aJ = [], [], []
    for k in (shift_samples(len(D["t"]), n_shift) if n_shift else []):
        Gs = shift(G, k)
        inv_s = D["invalid"] | Gs["invalid"]
        qs = gpred_hand(D["g"], Gs, mode, depth)
        rs = sp.band_gains(D["t"], D["g"], D["gdot"], w_equiv(qs, D["g"]), D["walk"], inv_s, thr_deg, dil, bands=("gait",))
        if rs["gait"].get("ok", False):
            kH.append(rs["gait"]["kappa_sp"])
        if joint:
            Js = joint_fit(D, qs, inv_s, thr_deg, dil)
            if Js["ok"]:
                bJ.append(Js["b"]); aJ.append(Js["a"])
    if n_shift:
        out.update(Hshift_n_ok=len(kH), Hshift_kappa_sp=float(np.mean(kH)) if kH else np.nan,
                   Jshift_n_ok=len(bJ), Jshift_b_hand=float(np.mean(bJ)) if bJ else np.nan,
                   Jshift_a_world=float(np.mean(aJ)) if aJ else np.nan)
    out.update(hand_lambda(D, G))
    return out


CAL_FREQS_REC = (1.0, 1.8, 3.0)


def mask_calibration(D: dict, fr: dict, side: str, frequency: float, thr_deg=40.0, dil=2) -> dict:
    """Recording-specific equal-mixture calibration (recorded gaze, head and hand kinematics do not enter the
    calibration response or predictors).
    Exact continuous-time 50:50 mixture of the hand-translation and world-rotation predictions (the Codex
    round-1 probe, scripts/m3t_synthetic.calibration), placed on this recording's grid with its walking mask,
    its gaze-invalid mask and its native hand timestamps / existence confidences of the analysed side, then
    passed through side_on_grid and the joint fit exactly as the data are. b - a is the total estimation offset for
    this specified synthetic signal under the recording's masks and sampling (truth 0); it includes the clean
    calibration offset and is not a bias correction or error bound for the real signal."""
    t = D["t"]
    n = len(t)
    zero = np.zeros(n)
    om = 2 * np.pi * frequency
    A = 0.01
    gx, gy = A * np.sin(om * t), A * np.cos(om * t)
    g = np.stack([gx, gy, np.sqrt(1 - gx ** 2 - gy ** 2)], 1)
    gdot_true = np.stack([A * om * np.cos(om * t), -A * om * np.sin(om * t), zero], 1)
    pos = lambda tt: np.stack([0.4 * 2 * A * np.sin(om * tt), 0 * tt, 0.4 + 0 * tt], 1)
    vel = np.stack([0.4 * 2 * A * om * np.cos(om * t), zero, zero], 1)
    p = pos(t)
    qh = (vel - np.sum(vel * g, 1)[:, None] * g) / np.linalg.norm(p, axis=1)[:, None]
    w_cpf = np.cross(2 * gdot_true - qh, g)
    th = fr["t"]
    frc = {"t": th, side: {"conf": fr[side]["conf"], "p_dev": (pos(th) - TC) @ Rd,
                           "q": np.tile([0.0, 0.0, 0.0, 1.0], (len(th), 1))}}
    G = side_on_grid(frc, side, t)
    Dc = {"t": t, "g": g, "gdot": np.gradient(g, t, axis=0), "w_cpf": w_cpf, "invalid": D["invalid"],
          "walk": D["walk"], "depth": None}
    J = joint_fit(Dc, gpred_hand(g, G, "trans"), D["invalid"] | G["invalid"], thr_deg, dil)
    return {"b_minus_a": J["b"] - J["a"] if J["ok"] else np.nan, "ok": J["ok"]}
