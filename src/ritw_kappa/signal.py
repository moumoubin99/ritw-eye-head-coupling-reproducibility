"""Signal path: common 60 Hz clock, causal gait-band filtering, P, kappa.

    g(t)      unit gaze in CPF (eye-in-head direction)
    w(t)      head angular velocity, rotated device -> CPF
    gdot_pred = -w x g          (perfect rotational compensation of a far target)
    band-pass: causal 2nd-order Butterworth (scipy N=2) 0.8-3 Hz on gdot and gdot_pred
    P(t)      = mean_W |gdot_pred_bp|^2
    kappa(t)  = mean_W <gdot_bp, gdot_pred_bp> / (P + lam),  lam = 0.05 * P_min
All window statistics are trailing (samples in (t-W, t]), so a value at t uses only data <= t.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from . import FS
from .io import DEFAULT_T_CPF_DEVICE

BAND = (0.8, 3.0)
BUTTER_N = 2
SETTLE_S = 1.0  # band-pass output treated as invalid for 1 s after a gaze gap


def bandpass_sos(fs: float = FS, band=BAND, n: int = BUTTER_N):
    return sps.butter(n, band, btype="bandpass", fs=fs, output="sos")


def causal_bp(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Causal band-pass along axis 0. NaNs are zero-filled (validity handled separately)."""
    x = np.nan_to_num(np.asarray(x, float))
    return sps.sosfilt(bandpass_sos(fs), x, axis=0)


def trailing_mean(x: np.ndarray, n: int, valid: np.ndarray | None = None, min_frac: float = 0.8):
    """Mean of x over the last n samples (inclusive), ignoring invalid samples.

    Returns (mean, frac_valid). mean is NaN where fewer than min_frac*n valid samples.
    """
    x = np.asarray(x, float)
    ok = np.isfinite(x) if valid is None else (valid & np.isfinite(x))
    xs = np.where(ok, x, 0.0)
    c = np.concatenate([[0.0], np.cumsum(xs)])
    k = np.concatenate([[0.0], np.cumsum(ok.astype(float))])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - n, 0)
    s = c[idx] - c[lo]
    cnt = k[idx] - k[lo]
    frac = cnt / n
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / cnt
    m[(frac < min_frac)] = np.nan
    return m, frac


GAZE_FRESH_S = 1.5 / FS     # a gaze sample is usable for <= 25 ms after its timestamp
TRAJ_STALE_S = 0.05         # trajectory hold limit when a grid bin has no sample


def _causal_box(t_src_s, x_src, t_grid_s, width_s, stale_s=TRAJ_STALE_S):
    """Causal box-car mean of the finite irregular samples in (t-width, t]. If the bin is empty,
    hold the last finite sample for at most `stale_s`; otherwise NaN."""
    x_src = np.asarray(x_src, float)
    fin = np.isfinite(x_src).all(axis=tuple(range(1, x_src.ndim))) if x_src.ndim > 1 else np.isfinite(x_src)
    ts, xs = t_src_s[fin], x_src[fin]
    cs = np.concatenate([np.zeros((1,) + xs.shape[1:]), np.cumsum(xs, 0)])
    hi = np.searchsorted(ts, t_grid_s, side="right")
    lo = np.searchsorted(ts, t_grid_s - width_s, side="right")
    cnt = hi - lo
    shape = (-1,) + (1,) * (xs.ndim - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (cs[hi] - cs[lo]) / cnt.reshape(shape)
    last = np.clip(hi - 1, 0, max(len(ts) - 1, 0))
    hold = (cnt == 0) & (hi > 0)
    if len(ts):
        hold &= (t_grid_s - ts[last]) <= stale_s
        out[hold] = xs[last[hold]]
    out[(cnt == 0) & ~hold] = np.nan
    return out


def _causal_index(t_src_s, t_grid_s, fresh_s):
    """Index of the last source sample at or before each grid time, and whether it is fresh."""
    j = np.searchsorted(t_src_s, t_grid_s, side="right") - 1
    ok = j >= 0
    jc = np.clip(j, 0, len(t_src_s) - 1)
    ok &= (t_grid_s - t_src_s[jc]) <= fresh_s
    return jc, ok


def _settled(bad: np.ndarray, settle: int) -> np.ndarray:
    """True where no bad sample occurred in the last `settle` samples (inclusive)."""
    c = np.concatenate([[0], np.cumsum(bad)])
    idx = np.arange(1, len(bad) + 1)
    return (c[idx] - c[np.maximum(idx - settle, 0)]) == 0


def build_session(gaze, traj, T_cpf_device: np.ndarray = DEFAULT_T_CPF_DEVICE, fs: float = FS) -> dict:
    """Put gaze + trajectory on a 60 Hz grid anchored at the first gaze sample (t=0).

    Strictly causal: a grid value at t uses the last gaze sample at or before t (zero-order hold,
    fresh for <= GAZE_FRESH_S) and the trajectory samples in (t - 1/fs, t]."""
    t0 = gaze["t_us"].iloc[0]
    tg = (gaze["t_us"].to_numpy(np.int64) - t0) / 1e6
    n = int(np.floor(tg[-1] * fs)) + 1
    t = np.arange(n) / fs
    jn, hit = _causal_index(tg, t, GAZE_FRESH_S)
    gdir_ok = np.isfinite(gaze[["gx", "gy", "gz"]].to_numpy(float)).all(1)
    gaze_ok = hit & gdir_ok[jn]

    def gcol(c):
        v = gaze[c].to_numpy(float)[jn]
        v[~gaze_ok] = np.nan
        return v
    g = np.stack([gcol("gx"), gcol("gy"), gcol("gz")], 1)

    tt = (traj["t_us"] - t0) / 1e6
    R_cd = T_cpf_device[:3, :3]
    r_dev = -R_cd.T @ T_cpf_device[:3, 3]           # CPF origin in device frame
    w_dev = _causal_box(tt, traj["w"], t, 1.0 / fs)
    v_w = _causal_box(tt, traj["v"], t, 1.0 / fs)
    k, k_ok = _causal_index(tt, t, TRAJ_STALE_S)
    R_wd = traj["R"][k]
    up = traj["up"][k]
    traj_ok = k_ok & np.isfinite(w_dev).all(1) & np.isfinite(v_w).all(1) & np.isfinite(R_wd).all((1, 2))
    w_dev[~traj_ok] = np.nan
    v_w[~traj_ok] = np.nan

    w_cpf = w_dev @ R_cd.T
    v_dev = np.einsum("nji,nj->ni", R_wd, v_w)
    v_eye_cpf = (v_dev + np.cross(w_dev, r_dev)) @ R_cd.T
    f_w = np.einsum("nij,j->ni", R_wd, R_cd.T @ np.array([0.0, 0.0, 1.0]))
    pitch_h = np.arcsin(np.clip(np.sum(f_w * up, 1), -1, 1))
    turn = np.sum(np.einsum("nij,nj->ni", R_wd, w_dev) * up, 1)
    v_vert = np.sum(v_w * up, 1)
    v_h = np.linalg.norm(v_w - v_vert[:, None] * up, axis=1)
    pitch_h[~traj_ok] = np.nan

    gdot = np.full_like(g, np.nan)
    gdot[1:] = (g[1:] - g[:-1]) * fs
    gpred = -np.cross(w_cpf, g)
    settle = int(round(SETTLE_S * fs))
    bad = ~(np.isfinite(gdot).all(1) & np.isfinite(gpred).all(1))
    valid = _settled(bad, settle)
    traj_valid = _settled(~traj_ok, settle)

    s = {
        "t": t, "fs": fs, "g": g, "gdot": gdot, "gpred": gpred, "w": w_cpf, "v_eye": v_eye_cpf,
        "v_h": v_h, "turn": turn, "pitch_h": pitch_h, "valid": valid, "traj_ok": traj_ok,
        "traj_valid": traj_valid,
        "depth": gcol("depth"), "vergence": gcol("vergence"), "q_width": gcol("q_width"),
        "pitch_g": gcol("pitch_g"), "yaw_g": gcol("yaw_g"), "gaze_missing": ~gaze_ok,
    }
    add_bandpassed(s)
    return s


def add_bandpassed(s: dict) -> dict:
    gb = causal_bp(s["gdot"], s["fs"])
    pb = causal_bp(s["gpred"], s["fs"])
    wb = causal_bp(s["w"], s["fs"])
    s["dot"] = np.sum(gb * pb, 1)
    s["pp"] = np.sum(pb * pb, 1)
    s["ee"] = np.sum(gb * gb, 1)
    s["hh"] = np.sum(wb * wb, 1)
    s["gdot_bp"], s["gpred_bp"] = gb, pb
    return s


def window_stats(s: dict, W: float) -> dict:
    """Trailing-window means: S=<gdot,gpred>, P, gaze band power E, head band power Hb."""
    n = int(round(W * s["fs"]))
    v = s["valid"]
    S, frac = trailing_mean(s["dot"], n, v)
    P, _ = trailing_mean(s["pp"], n, v)
    E, _ = trailing_mean(s["ee"], n, v)
    Hb, _ = trailing_mean(s["hh"], n, s.get("traj_valid", np.isfinite(s["hh"])))
    return {"S": S, "P": P, "E": E, "Hb": Hb, "frac_valid": frac}


def kappa_from_stats(S, P, P_min: float, kappa0: float, lam_frac: float = 0.05):
    """kappa = S/(P+lam); mask m = 1[P >= P_min]; kappa filled with kappa0 where m=0 or undefined."""
    lam = lam_frac * P_min
    with np.errstate(invalid="ignore"):
        k = S / (P + lam)
        m = (P >= P_min)
    m = m & np.isfinite(k)
    kf = np.where(m, k, kappa0)
    return kf, m.astype(float), k


def group_delay_step(W: float, fs: float = FS, f0: float = 1.8, dur: float = 20.0):
    """Delay of the kappa estimate after a step change of true gain 1 -> 0 at t=dur/2.

    A pure gait-band rotation drives gdot_pred; gdot = kappa_true(t) * gdot_pred.
    Returns the time from the step until kappa_hat crosses 0.5.
    """
    t = np.arange(int(dur * fs)) / fs
    p = np.stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * 0.7 * f0 * t), 0 * t], 1) * 0.2
    ktrue = np.where(t < dur / 2, 1.0, 0.0)
    gb = causal_bp(ktrue[:, None] * p, fs)
    pb = causal_bp(p, fs)
    n = int(round(W * fs))
    S, _ = trailing_mean(np.sum(gb * pb, 1), n)
    P, _ = trailing_mean(np.sum(pb * pb, 1), n)
    k = S / P
    after = np.where((t >= dur / 2) & (k < 0.5))[0]
    return float(t[after[0]] - dur / 2) if len(after) else np.nan


def residual_rotation(s: dict, mask: np.ndarray | None = None, min_samples: int = 600, n_iter: int = 40,
                      translation: bool = True, block_s: float = 8.0, max_step: float = 0.1):
    """Data-driven check of the device->CPF rotation. If the true head rate in CPF is R w (R a residual
    rotation), then gdot ~ -kappa(t) (R w) x g - (I - g g^T)(R v)/d(t). Model, per block j of `block_s`:
        gdot_bp ~ a_j [X0 + sum_k b_k X_k] + c_j T + l_j L,
    X0 = bp(g x w'), X_k = bp(g x (e_k x w')), T = bp(-(I - g g^T) v'), L = bp(g x dw'/dt), w' = R_cur w,
    v' = R_cur v (eye translation velocity `v_eye`). a_j (coupling gain), c_j (inverse depth) and l_j (a
    small timing lag) are block-local, so changing coupling, depth and sensor timing are not absorbed into
    the rotation; b (shared) is the rotation increment. Alternating least squares: (1) per-block fit of
    (a_j, c_j, l_j) at b = 0, (2) shared fit of b on the residual with regressors a(t) X_k; R_cur is updated
    by a damped step (|b| <= max_step rad). The translation term matters: gait-locked eye translation at
    near depth is otherwise absorbed as a spurious rotation (tens of degrees at 0.6 m; run_v1_rotation.py).
    A full 3x3 fit is not used: the component of R w along g does not move the gaze and is unidentifiable.
    Returns {angle_deg, rotvec, gain (median a_j), inv_depth (median c_j), n, n_blocks, converged};
    None if too few samples. Unconverged or implausible (median gain <= 0.2) fits have converged=False and
    must be treated as unresolved."""
    from scipy.spatial.transform import Rotation as _Rot
    g, w, fs = s["g"], s["w"], s["fs"]
    ok = s["valid"] & np.isfinite(g).all(1) & np.isfinite(w).all(1) & np.isfinite(s["gdot_bp"]).all(1)
    use_v = translation and "v_eye" in s
    if use_v:
        ok &= np.isfinite(s["v_eye"]).all(1)
    if mask is not None:
        ok &= mask
    if ok.sum() < min_samples:
        return None
    gn, wn = np.nan_to_num(g), np.nan_to_num(w)
    vn = np.nan_to_num(s["v_eye"]) if use_v else None
    y = np.nan_to_num(s["gdot_bp"])
    nb = int(round(block_s * fs))
    blk = np.arange(len(g)) // nb
    blocks = [np.flatnonzero(ok & (blk == j)) for j in np.unique(blk[ok])]
    blocks = [ix for ix in blocks if len(ix) >= nb // 2]
    if sum(len(ix) for ix in blocks) < min_samples:
        return None
    E = np.eye(3)
    r = np.zeros(3)
    converged = False
    for _ in range(n_iter):
        Rc = _Rot.from_rotvec(r).as_matrix()
        wr = wn @ Rc.T
        X0 = causal_bp(np.cross(gn, wr), fs)
        Xk = [causal_bp(np.cross(gn, np.cross(E[k], wr)), fs) for k in range(3)]
        L = causal_bp(np.cross(gn, np.gradient(wr, axis=0) * fs), fs)
        cols = [X0, L]
        if use_v:
            vr = vn @ Rc.T
            cols.append(causal_bp(-(vr - np.sum(gn * vr, 1, keepdims=True) * gn), fs))
        fit = np.zeros(len(g))
        a_t = np.zeros(len(g))
        gains, invd = [], []
        resid = np.zeros_like(y)
        for ix in blocks:                                   # (1) block-local gain, lag, inverse depth
            A = np.stack([c[ix].reshape(-1) for c in cols], 1)
            coef = np.linalg.lstsq(A, y[ix].reshape(-1), rcond=None)[0]
            resid[ix] = y[ix] - (A @ coef).reshape(-1, 3)
            a_t[ix] = coef[0]
            gains.append(coef[0]); invd.append(coef[2] if use_v else np.nan)
        allix = np.concatenate(blocks)                      # (2) shared rotation increment
        B = np.stack([(a_t[allix, None] * Xk[k][allix]).reshape(-1) for k in range(3)], 1)
        db = np.linalg.lstsq(B, resid[allix].reshape(-1), rcond=None)[0]
        nrm = np.linalg.norm(db)
        if nrm > max_step:
            db *= max_step / nrm
        r = (_Rot.from_rotvec(db) * _Rot.from_rotvec(r)).as_rotvec()
        if nrm < 1e-4:
            converged = True
            break
    gain = float(np.median(gains))
    return {"angle_deg": float(np.degrees(np.linalg.norm(r))), "rotvec": [float(x) for x in r],
            "gain": gain, "inv_depth": float(np.nanmedian(invd)) if use_v else None,
            "n": int(sum(len(ix) for ix in blocks)), "n_blocks": len(blocks),
            "converged": bool(converged and gain > 0.2)}


