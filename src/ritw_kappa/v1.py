"""V1 data-free recovery: does kappa_hat recover a known rotational gain under gait-like head motion,
translational (depth-dependent) eye motion, gaze noise, window length and extrinsic misalignment?

Ground truth (600 Hz):  gdot = -kappa_true * (w x g) - (I - g g^T) v_eye / d + recentring
Measured (60 Hz):       g + white angular noise sigma; w_meas = causal 1/60 s box mean of R_mis w
"""
from __future__ import annotations

import numpy as np

from .signal import causal_bp, trailing_mean

FS_TRUE, FS = 600, 60
LAM_PMIN = 0.003  # provisional P_min placeholder (real value comes from R001 steady walking)


def _rot(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    th = np.deg2rad(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def gait_omega(t, rng, f_step=1.9):
    """CPF axes (x right, y down, z forward): pitch about x at step freq, yaw (y) / roll (z) at stride."""
    f = f_step * (1 + 0.03 * rng.standard_normal())
    ph = rng.uniform(0, 2 * np.pi, 3)
    jitter = np.cumsum(rng.standard_normal(len(t))) * 0.002  # slow phase wander
    w = np.stack([1.0 * np.sin(2 * np.pi * f * t + ph[0] + jitter),
                  0.6 * np.sin(np.pi * f * t + ph[1] + jitter),
                  0.4 * np.sin(np.pi * f * t + ph[2] + jitter)], 1)
    return w + 0.05 * rng.standard_normal(w.shape)


def simulate(kappa_true, scale, d, rng, dur=60.0, return_v=False):
    """kappa_true: scalar, or a callable t -> kappa (time-varying coupling)."""
    n = int(dur * FS_TRUE)
    t = np.arange(n) / FS_TRUE
    kt = kappa_true(t) if callable(kappa_true) else np.full(n, float(kappa_true))
    w = scale * gait_omega(t, rng)
    fstep = 1.9
    v = np.stack([0.03 * 2 * np.pi * fstep / 2 * np.cos(np.pi * fstep * t),     # lateral sway
                  0.02 * 2 * np.pi * fstep * np.cos(2 * np.pi * fstep * t),     # vertical bob
                  np.full(n, 1.2)], 1)                                          # forward
    g0 = np.array([0.0, 0.25, 1.0]); g0 /= np.linalg.norm(g0)
    g = np.empty((n, 3)); g[0] = g0
    dt = 1.0 / FS_TRUE
    for i in range(1, n):
        gi, wi, vi = g[i - 1], w[i - 1], v[i - 1]
        gd = -kt[i - 1] * np.cross(wi, gi) - (vi - np.dot(vi, gi) * gi) / d + 0.3 * (g0 - gi)
        x = gi + gd * dt
        g[i] = x / np.linalg.norm(x)
    return (t, g, w, v) if return_v else (t, g, w)


def measure(t, g, w, rng, sigma_deg=0.3, R_mis=np.eye(3), R_common=np.eye(3)):
    k = FS_TRUE // FS
    gm = g[k - 1::k] @ R_common.T
    wm = (w.reshape(-1, k, 3).mean(1) @ R_mis.T) @ R_common.T  # causal 1/60 s box mean
    gm = gm + np.deg2rad(sigma_deg) * rng.standard_normal(gm.shape)
    gm /= np.linalg.norm(gm, axis=1, keepdims=True)
    gdot = np.zeros_like(gm); gdot[1:] = (gm[1:] - gm[:-1]) * FS
    gpred = -np.cross(wm, gm)
    return causal_bp(gdot, FS), causal_bp(gpred, FS)


def kappa_hat(gdot_bp, gpred_bp, W, settle_s=3.0, p_min=LAM_PMIN):
    n = int(round(W * FS))
    S, _ = trailing_mean(np.sum(gdot_bp * gpred_bp, 1), n)
    P, _ = trailing_mean(np.sum(gpred_bp**2, 1), n)
    sl = slice(int(settle_s * FS), None)
    k = S[sl] / (P[sl] + 0.05 * p_min)
    return float(np.nanmedian(k)), float(np.nanmedian(P[sl])), k


def scale_for_P(target_P, rng_seed=0):
    """omega scale giving median P ~= target (P is quadratic in the scale)."""
    rng = np.random.default_rng(rng_seed)
    t, g, w = simulate(1.0, 1.0, 1e6, rng, dur=20.0)
    gb, pb = measure(t, g, w, rng, sigma_deg=0.0)
    _, P1, _ = kappa_hat(gb, pb, 2.0)
    return np.sqrt(target_P / P1)


def wstats(k, kappa_true):
    """Window-level bias and RMSE over all trailing windows after settling."""
    e = k[np.isfinite(k)] - kappa_true
    return (float(np.mean(e)), float(np.sqrt(np.mean(e ** 2)))) if len(e) else (np.nan, np.nan)


def cell(kappa_true, P_target, d, seed, sigmas=(0.1, 0.3, 1.0), Ws=(1.0, 2.0, 4.0),
         mis_deg=(0, 2, 5, 10, 30), scale=None, p_min=LAM_PMIN):
    rng = np.random.default_rng(seed)
    scale = scale if scale is not None else scale_for_P(P_target)
    t, g, w = simulate(kappa_true, scale, d, rng)
    rows = []
    for sg in sigmas:
        for W in Ws:
            gb, pb = measure(t, g, w, np.random.default_rng(seed + 7), sigma_deg=sg)
            kh, Pm, ks = kappa_hat(gb, pb, W, p_min=p_min)
            bw, rw = wstats(ks, kappa_true)
            rows.append(dict(kind="grid", kappa_true=kappa_true, P_target=P_target, P_meas=Pm, d=d, seed=seed,
                             sigma=sg, W=W, mis=0.0, kappa_hat=kh, bias=bw, rmse=rw, bias_median=kh - kappa_true))
    for m in mis_deg:
        R = _rot([1, 1, 0.3], m)
        gb, pb = measure(t, g, w, np.random.default_rng(seed + 7), sigma_deg=0.3, R_mis=R)
        kh, Pm, _ = kappa_hat(gb, pb, 2.0, p_min=p_min)
        rows.append(dict(kind="misalign", kappa_true=kappa_true, P_target=P_target, P_meas=Pm, d=d, seed=seed,
                         sigma=0.3, W=2.0, mis=float(m), kappa_hat=kh, bias=kh - kappa_true))
    # common-rotation invariance (noise-free so the comparison is exact): rotating gaze and omega
    # by the same frame change must leave kappa unchanged
    gb, pb = measure(t, g, w, rng, sigma_deg=0.0)
    k0, _, _ = kappa_hat(gb, pb, 2.0)
    gb, pb = measure(t, g, w, rng, sigma_deg=0.0, R_common=_rot([0.2, 1, 0.5], 40))
    kh, Pm, _ = kappa_hat(gb, pb, 2.0)
    rows.append(dict(kind="common_rot", kappa_true=kappa_true, P_target=P_target, P_meas=Pm, d=d, seed=seed,
                     sigma=0.0, W=2.0, mis=0.0, kappa_hat=kh, bias=kh - kappa_true, kappa_hat_base=k0))
    return rows


def rotation_recovery(mis_deg, d, sigma_deg, kappa_sched, seed, dur=120.0, P_target=0.01):
    """Validates signal.residual_rotation under the V1 nuisances: a KNOWN device->CPF misalignment
    (w_meas = R_mis w), eye translation at depth d, gaze noise sigma and constant or time-varying kappa.
    Returns the estimated angle, its error vs mis_deg and the full rotation error (deg)."""
    from scipy.spatial.transform import Rotation as _Rot
    from .signal import residual_rotation
    rng = np.random.default_rng(seed)
    kap = {"const": 0.9,
           "switch": lambda t: np.where((t // 10) % 2 == 0, 0.9, 0.4),           # break / recover every 10 s
           "drift": lambda t: 0.9 - 0.5 * (t / t[-1])}[kappa_sched]
    t, g, w, v = simulate(kap, scale_for_P(P_target), d, rng, dur=dur, return_v=True)
    axis = rng.standard_normal(3)
    R = _rot(axis, mis_deg)
    k = FS_TRUE // FS
    gm = g[k - 1::k] + np.deg2rad(sigma_deg) * rng.standard_normal((len(t) // k, 3))
    gm /= np.linalg.norm(gm, axis=1, keepdims=True)
    wm = w.reshape(-1, k, 3).mean(1) @ R.T
    gdot = np.zeros_like(gm); gdot[1:] = (gm[1:] - gm[:-1]) * FS
    valid = np.ones(len(gm), bool); valid[:int(3 * FS)] = False
    vm = v.reshape(-1, k, 3).mean(1) @ R.T                        # same wrong extrinsic on translation
    s = {"g": gm, "w": wm, "v_eye": vm, "fs": FS, "valid": valid, "gdot_bp": causal_bp(gdot, FS)}
    rr = residual_rotation(s)
    if rr is None or not rr["converged"]:
        return {"mis_deg": mis_deg, "d": d, "sigma": sigma_deg, "kappa": kappa_sched, "seed": seed, "resolved": False}
    R_est = _Rot.from_rotvec(rr["rotvec"]).as_matrix()               # estimate of the correction R^T
    err = np.degrees(_Rot.from_matrix(R_est @ R).magnitude())
    return {"mis_deg": mis_deg, "d": d, "sigma": sigma_deg, "kappa": kappa_sched, "seed": seed, "resolved": True,
            "angle_est": rr["angle_deg"], "angle_err": abs(rr["angle_deg"] - mis_deg), "rot_err_deg": float(err),
            "gain": rr["gain"]}


# ---------------------------------------------------------------- real-driver recovery (R007)

def simulate_driven(w60, v60, g60, kappas, depths, sub=FS_TRUE // FS, recentre=0.3):
    """Integrate gaze at 600 Hz under REAL 60 Hz drivers (head angular velocity w60 and eye
    velocity v60 in CPF, linearly interpolated) for every (kappa_true, depth) pair at once:
        gdot = -kappa (w x g) - (I - g g^T) v / d(t) + recentre * (g_real - g)
    g_real (measured gaze, forward-filled) is the slow re-centring target, so geometry follows the
    recording. Each depth is a constant (m) or a 60 Hz series (measured, time-varying).
    Returns g at 60 Hz, shape (len(kappas)*len(depths), n, 3)."""
    n = len(w60)
    t60 = np.arange(n) / FS
    tt = np.arange((n - 1) * sub + 1) / FS_TRUE
    itp = lambda x: np.stack([np.interp(tt, t60, x[:, j]) for j in range(3)], 1)
    w, v, gr = itp(w60), itp(v60), itp(g60)
    gr /= np.linalg.norm(gr, axis=1, keepdims=True)
    Dser = np.stack([np.interp(tt, t60, np.broadcast_to(np.asarray(d, float), (n,))) for d in depths])
    K = np.repeat(np.asarray(kappas, float), len(depths))[:, None]
    Didx = np.tile(np.arange(len(depths)), len(kappas))
    g = np.tile(gr[0], (len(K), 1))
    out = np.empty((len(K), n, 3)); out[:, 0] = g
    dt = 1.0 / FS_TRUE
    for i in range(1, len(tt)):
        wi, vi = w[i - 1], v[i - 1]
        vp = vi[None] - np.sum(g * vi, 1, keepdims=True) * g
        gd = -K * np.cross(wi[None], g) - vp / Dser[Didx, i - 1][:, None] + recentre * (gr[i - 1][None] - g)
        g = g + gd * dt
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        if i % sub == 0:
            out[:, i // sub] = g
    return out


def measured_depth_cases(rec, i0, i1, n_win, fallback=(0.35, 0.6, 1.5, 5.0), min_cov=0.5):
    """Measured MPS depth cases over samples [i0, i1), clipped to [0.2, 20] m. Coverage is judged on
    the ANALYSED window (the last `n_win` samples): below `min_cov` -> [] (caller falls back to the
    constant depths). If any sample in [i0, i1) lacks depth, it is filled with EACH fallback depth
    in turn (keys "measured|<d>") so the caller takes the worst case; missing depth is never
    interpolated optimistically. With complete depth: a single case "measured"."""
    d = rec.s["depth"][i0:i1].astype(float)
    if len(d) < n_win or np.isfinite(d[-n_win:]).mean() < min_cov:
        return []
    miss = ~np.isfinite(d)
    if not miss.any():
        return [("measured", np.clip(d, 0.2, 20.0))]
    out = []
    for f in fallback:
        x = d.copy(); x[miss] = f
        out.append((f"measured|{f:g}", np.clip(x, 0.2, 20.0)))
    return out


def driven_window_error(rec, t_end, W, p_min, kappas=(0.3, 0.6, 0.9, 1.0), depths=(0.35, 0.6, 1.5, 5.0),
                        sigma_deg=0.3, n_noise=5, pre_s=8.0, seed=0, measured_depth=True):
    """kappa_hat at the window ending at t_end, re-simulated with the recording's real drivers.
    Window statistics use the recording's OBSERVED validity mask (`valid`, same trailing-mean rule
    as the analysed feature), so a window that is missing in the analysis is missing here too.
    With measured_depth, the time-varying measured depth cases (measured_depth_cases) are added.
    Returns dict case -> (bias over noise seeds, rmse) keyed (kappa, depth), or None."""
    s = rec.s
    i1 = int(np.floor(t_end * FS + 1e-9)) + 1
    i0 = max(0, i1 - int((pre_s + W) * FS))
    if i1 - i0 < int((W + 2) * FS):
        return None
    w60, v60 = s["w"][i0:i1], s["v_eye"][i0:i1]
    ok = np.isfinite(w60).all(1) & np.isfinite(v60).all(1)
    if ok[-int(W * FS):].mean() < 0.8:
        return None
    w60, v60 = np.nan_to_num(w60), np.nan_to_num(v60)
    import pandas as pd
    g60 = pd.DataFrame(s["g"][i0:i1]).ffill().bfill().to_numpy()
    if not np.isfinite(g60).all():
        return None
    n = int(round(W * FS))
    mc = measured_depth_cases(rec, i0, i1, n, depths) if measured_depth else []
    cases = list(depths) + [x for _, x in mc]
    keys = list(depths) + [k for k, _ in mc]
    G = simulate_driven(w60, v60, g60, kappas, cases)
    va = s["valid"][i0:i1]
    res = {}
    for b, (k, d) in enumerate([(k, d) for k in kappas for d in keys]):
        e = []
        for r in range(n_noise):
            rng = np.random.default_rng(seed * 1000 + r)
            gm = G[b] + np.deg2rad(sigma_deg) * rng.standard_normal(G[b].shape)
            gm /= np.linalg.norm(gm, axis=1, keepdims=True)
            gdot = np.zeros_like(gm); gdot[1:] = (gm[1:] - gm[:-1]) * FS
            gb, pb = causal_bp(gdot, FS), causal_bp(-np.cross(w60, gm), FS)
            S = trailing_mean(np.sum(gb * pb, 1), n, va)[0][-1]
            P = trailing_mean(np.sum(pb * pb, 1), n, va)[0][-1]
            e.append(S / (P + 0.05 * p_min) - k)
        e = np.asarray(e)
        res[(k, d)] = (float(np.mean(e)), float(np.sqrt(np.mean(e ** 2))))
    res["measured_keys"] = [k for k, _ in mc]
    res["measured_depth_used"] = bool(mc)
    return res
