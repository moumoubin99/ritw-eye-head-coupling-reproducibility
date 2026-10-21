"""M3 (refine-logs/EXPERIMENT_PLAN_M3.md): slow-phase vs raw eye-head compensation gain.

Offline measurement, so zero-phase filtering is allowed. All vectors are in CPF (X left, Y up, Z fwd).
  g       unit gaze direction (binocular intersection, io.gaze_dirs_cpf)
  gdot    d g / dt on a 60 Hz grid (central difference)
  gpred   -omega_cpf x g            (kappa = 1 <=> full compensation of head rotation)
  F       fast-phase mask: |gdet - k*gpred| > thr (deg/s; gdet = 5-sample Savitzky-Golay velocity) or
          gaze invalid, dilated by `dil` samples; k is the current slow-phase gain (fixed-point iteration)
  gdot_sp gdot with the residual gdot - k*gpred bridged linearly across each F run <= max_gap_s;
          longer F runs (and runs touching the window edge) split the signal into segments
  kappa_sp  = sum_{S\\F} <gdot_sp_b, gpred_b> / sum_{S\\F} |gpred_b|^2   (both band-passed on the same
              contiguous segments; k, F and gdot_sp iterated to a fixed point from k = 1, k = 0 fallback)
  kappa_raw = sum_Sraw <gdot_b, gpred_b> / sum_Sraw |gpred_b|^2   (conventional score: no fast-phase
              handling; walking-only segments split only by long invalid-gaze spans)
  c = kappa_raw - kappa_sp          (separation between the conventional and the slow-phase estimator)
  c_common = kappa_raw_S - kappa_sp_S   (both on the slow-phase segments and the same support S: the
              effect of the signal replacement alone)
Zero-phase 2nd-order Butterworth band-pass per segment. Segments are walking-only (runs of walk & ~gap
>= min_seg_s); S = samples at least trim_s from the segment edges. Component gains (cx, cy) use the
CPF x and y components of the same vectors; they are projections, not isolated head-rotation axes."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter1d
from scipy.signal import butter, savgol_filter, sosfiltfilt, welch

from . import io as rio

FS = 60.0
BANDS = {"gait": dict(lo=0.8, hi=3.0, min_seg_s=10.0, trim_s=1.0, min_s=30.0),
         "slow": dict(lo=0.15, hi=0.5, min_seg_s=20.0, trim_s=4.0, min_s=60.0)}
WALK_V, RUN_V, WALK_SMOOTH_S, MIN_BOUT_S = 0.5, 2.5, 1.0, 4.0
MIN_FIT_S = 2.0     # numerical identifiability floor for an intermediate fit (not the reporting minimum)
MASK_TOL = 1e-3     # max fraction of S samples whose mask may change at the accepted fixed point
NEAR_S = 0.05       # a grid sample is invalid when no retained gaze observation lies within NEAR_S
MIN_GAZE_OBS = 10   # recordings with <= MIN_GAZE_OBS finite gaze directions fail loading


def runs(m: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of the True runs of a boolean array."""
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def walking_mask(vh: np.ndarray) -> np.ndarray:
    vs = uniform_filter1d(vh, int(WALK_SMOOTH_S * FS), mode="nearest")
    m = (vs > WALK_V) & (vs <= RUN_V)
    out = np.zeros_like(m)
    for a, b in runs(m):
        if b - a >= MIN_BOUT_S * FS:
            out[a:b] = True
    return out


def fast_phase(gdot: np.ndarray, invalid: np.ndarray, thr_deg: float, dil: int) -> np.ndarray:
    F = (np.linalg.norm(gdot, axis=1) > np.radians(thr_deg)) | invalid
    return binary_dilation(F, iterations=dil) if dil > 0 else F


def obs_invalid(t: np.ndarray, tobs: np.ndarray, max_s: float = NEAR_S) -> np.ndarray:
    """Grid samples with no retained observation (timestamps tobs, sorted) within max_s."""
    j = np.clip(np.searchsorted(tobs, t), 1, len(tobs) - 1)
    return np.minimum(np.abs(t - tobs[j - 1]), np.abs(tobs[j] - t)) > max_s


def short_runs(F: np.ndarray, max_gap_s: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Split F into bridgeable runs (<= max_gap_s, bracketed by valid samples) and a gap mask."""
    n, out = len(F), []
    gap = np.zeros(n, bool)
    for a, b in runs(F):
        if b - a > max_gap_s * FS or a == 0 or b == n:
            gap[a:b] = True
        else:
            out.append((a, b))
    return out, gap


def impute(gdot, gpred, k_hat, bridge):
    """Model-based desaccading: inside each bridgeable F run, the residual gdot - k_hat*gpred is
    replaced by a straight line between the bracketing samples; outside F, gdot is unchanged.
    k_hat = 0 is plain linear interpolation of gdot."""
    out = gdot.copy()
    r = gdot - k_hat * gpred
    for a, b in bridge:
        w = np.linspace(0, 1, b - a + 2)[1:-1, None]
        out[a:b] = k_hat * gpred[a:b] + (1 - w) * r[a - 1] + w * r[b]
    return out


def _bp_segments(sos, x, segs):
    y = np.zeros_like(x)
    for a, b in segs:
        y[a:b] = sosfiltfilt(sos, x[a:b], axis=0)
    return y


def _state(gdet, gdot, gpred, invalid, walk, sos, B, thr_deg, dil, max_gap_s, k_hat):
    """Mask, segments and filtered signals implied by slow-phase gain k_hat. Segments are walking-only:
    contiguous runs of walk & ~gap of at least min_seg_s, trimmed by trim_s at both ends."""
    n = len(gdot)
    tr = int(B["trim_s"] * FS)
    F = fast_phase(gdet - k_hat * gpred, invalid, thr_deg, dil)
    bridge, gap = short_runs(F, max_gap_s)
    segs = [(a, b) for a, b in runs(walk & ~gap) if b - a >= B["min_seg_s"] * FS]
    S = np.zeros(n, bool)
    for a, b in segs:
        S[a + tr:b - tr] = True
    Ssp = S & ~F
    h = _bp_segments(sos, gpred, segs)
    e_sp = _bp_segments(sos, impute(gdot, gpred, k_hat, bridge), segs)
    den = np.sum(h[Ssp] ** 2)
    k = float(np.sum(e_sp[Ssp] * h[Ssp]) / den) if Ssp.sum() >= MIN_FIT_S * FS and den > 0 else None
    return dict(k=k, F=F, gap=gap, segs=segs, S=S, Ssp=Ssp, h=h, e_sp=e_sp)


def _fit_sp(gdet, gdot, gpred, invalid, walk, sos, B, thr_deg, dil, max_gap_s, k0, n_iter, tol):
    """Fixed point of: F = dilate(|gdet - k*gpred| > thr or invalid) -> impute -> band-pass -> k.
    Iterates while the fit is numerically identifiable (|S minus F| >= MIN_FIT_S); the band's minimum
    duration is applied by the caller to the final state only. On convergence the state is rebuilt at
    the gain k* that met |k_update - k_hat| < tol; it is accepted only if the update from that state,
    k_update, is also within tol of k* and the masks F on S and S itself change on at most MASK_TOL of
    the S samples. The returned gain k* is the model value the returned state was built with, so the
    result is an approximate fixed point with residual |k_update - k*| < tol (the discrete mask can flip
    a few threshold samples in a 2-cycle). status: converged | not_converged | no_fit."""
    k_hat, k_first = k0, None
    st = None
    for it in range(n_iter):
        st = _state(gdet, gdot, gpred, invalid, walk, sos, B, thr_deg, dil, max_gap_s, k_hat)
        if st["k"] is None:
            return dict(status="no_fit", n_iter=it + 1, k=None, k_first=k_first, k_update=np.nan, st=st,
                        mask_change=np.nan)
        if k_first is None:
            k_first = st["k"]
        if abs(st["k"] - k_hat) < tol:
            fin = _state(gdet, gdot, gpred, invalid, walk, sos, B, thr_deg, dil, max_gap_s, st["k"])
            if fin["k"] is not None and abs(fin["k"] - st["k"]) < tol:
                ns = max(int(st["S"].sum()), 1)
                dmask = (np.sum((fin["F"] & fin["S"]) != (st["F"] & st["S"])) + np.sum(fin["S"] != st["S"])) / ns
                if dmask <= MASK_TOL:
                    return dict(status="converged", n_iter=it + 2, k=st["k"], k_first=k_first,
                                k_update=fin["k"], st=fin, mask_change=float(dmask))
        k_hat = st["k"]
    return dict(status="not_converged", n_iter=n_iter, k=st["k"], k_first=k_first, k_update=np.nan, st=st,
                mask_change=np.nan)


def _gain(e, h, m, ax=None):
    if ax is None:
        return float(np.sum(e[m] * h[m]) / np.sum(h[m] ** 2))
    return float(np.sum(e[m, ax] * h[m, ax]) / np.sum(h[m, ax] ** 2))


def band_gains(t, g, gdot, w_cpf, walk, invalid, thr_deg=40.0, dil=2, max_gap_s=0.5,
               bands=("gait", "slow"), v_cpf=None, depth=None, n_iter=50, tol=1e-3, min_s=None):
    """Core estimator; inputs on a common FS grid. Returns {band: dict of gains, counts and status}.
    Fast phases are departures from the current slow-phase model, |gdot - k*gpred| > thr (5-point
    Savitzky-Golay velocity); detection, imputation and k are iterated to a fixed point. The fit is
    run from k = 1 (full compensation) and from k = 0. Each start over-masks when |k_true - k_start|*|gpred|
    exceeds thr (k = 0: high true gain with strong head motion; k = 1: low true gain with strong head
    motion), so the registered rule is: use the k = 1 fit if it converged, else the k = 0 fit if it
    converged, else the k = 1 result (start_used records which). When both converge, their difference
    is reported as a fixed-point check.
    min_s = {band: seconds} overrides the band's minimum final |S minus F| (sensitivity).
    status: ok | no_fit | not_converged | insufficient_final | insufficient_raw. Only status ok sets ok."""
    gdet = savgol_filter(g, 5, 2, deriv=1, delta=1 / FS, axis=0)
    gpred = -np.cross(w_cpf, g)
    gtr = None
    if v_cpf is not None and depth is not None:   # translation term for a fixed world target at depth d
        vperp = v_cpf - np.einsum("ni,ni->n", v_cpf, g)[:, None] * g
        gtr = -vperp / depth[:, None]
    n = len(t)
    _, inv_gap = short_runs(invalid, max_gap_s)
    res = {"walk_s": float(walk.sum() / FS)}
    for bn in bands:
        B = {**BANDS[bn], **({"min_s": min_s[bn]} if min_s and bn in min_s else {})}
        sos = butter(2, [B["lo"], B["hi"]], "bandpass", fs=FS, output="sos")
        tr = int(B["trim_s"] * FS)
        # conventional raw coupling: no fast-phase handling; walking-only segments split only by long
        # invalid-gaze spans; evaluated on valid samples
        segs_r = [(a, b) for a, b in runs(walk & ~inv_gap) if b - a >= B["min_seg_s"] * FS]
        S_r = np.zeros(n, bool)
        for a, b in segs_r:
            S_r[a + tr:b - tr] = True
        S_r &= ~invalid
        fit1 = _fit_sp(gdet, gdot, gpred, invalid, walk, sos, B, thr_deg, dil, max_gap_s, 1.0, n_iter, tol)
        fit0 = _fit_sp(gdet, gdot, gpred, invalid, walk, sos, B, thr_deg, dil, max_gap_s, 0.0, n_iter, tol)
        use0 = fit1["status"] != "converged" and fit0["status"] == "converged"
        fit = fit0 if use0 else fit1
        conv = lambda f: f["k"] if f["status"] == "converged" else np.nan
        r = {"n_Sraw_s": float(S_r.sum() / FS), "fit_status": fit["status"], "n_iter": fit["n_iter"],
             "mask_change": fit["mask_change"], "k_update": fit["k_update"], "start_used": 0 if use0 else 1,
             "start1_status": fit1["status"], "start0_status": fit0["status"],
             "kappa_sp_start1": conv(fit1), "kappa_sp_start0": conv(fit0)}
        if fit["status"] == "no_fit":
            res[bn] = {**r, "ok": False, "status": "no_fit"}
            continue
        st = fit["st"]
        h, e_sp, S, Ssp, F = st["h"], st["e_sp"], st["S"], st["Ssp"], st["F"]
        h_r = _bp_segments(sos, gpred, segs_r)
        e_r = _bp_segments(sos, gdot, segs_r)
        e_raw_s = _bp_segments(sos, gdot, st["segs"])   # raw signal on the slow-phase segments
        r.update(n_S_s=float(S.sum() / FS), n_Ssp_s=float(Ssp.sum() / FS),
                 fast_frac=float(F[S].mean()), f_ep_rate=float(len(runs(F & S)) / (S.sum() / FS)),
                 gap_frac_walk=float(st["gap"][walk].mean()) if walk.any() else np.nan,
                 P=float(np.sum(h[S] ** 2) / S.sum()),
                 kappa_sp=fit["k"], kappa_sp_first=fit["k_first"],
                 kappa_raw=_gain(e_r, h_r, S_r) if S_r.sum() >= MIN_FIT_S * FS else np.nan,
                 kappa_sp_S=_gain(e_sp, h, S), kappa_raw_S=_gain(e_raw_s, h, S))
        r["c"] = r["kappa_raw"] - r["kappa_sp"]
        r["c_common"] = r["kappa_raw_S"] - r["kappa_sp_S"]
        for ax, k in (("cx", 0), ("cy", 1)):   # CPF x / y component projection gains (not rotation axes)
            r[f"kappa_sp_{ax}"] = _gain(e_sp, h, Ssp, k)
            r[f"kappa_raw_{ax}"] = _gain(e_r, h_r, S_r, k) if S_r.sum() >= MIN_FIT_S * FS else np.nan
            r[f"c_{ax}"] = r[f"kappa_raw_{ax}"] - r[f"kappa_sp_{ax}"]
        if gtr is not None:   # post-hoc 2-regressor fit on the rotation-only desaccaded response
            h_tr = _bp_segments(sos, gtr, st["segs"])
            X = np.stack([h[Ssp].ravel(), h_tr[Ssp].ravel()], 1)
            coef = np.linalg.lstsq(X, e_sp[Ssp].ravel(), rcond=None)[0]
            r["kappa_sp_rot_given_trans"], r["trans_coef"] = float(coef[0]), float(coef[1])
        if fit["status"] != "converged":
            status = "not_converged"
        elif Ssp.sum() < B["min_s"] * FS:
            status = "insufficient_final"
        elif S_r.sum() < B["min_s"] * FS:
            status = "insufficient_raw"
        else:
            status = "ok"
        prim = [r["kappa_sp"], r["kappa_raw"], r["c"]]
        r["ok"] = bool(status == "ok" and np.all(np.isfinite(prim)))
        r["status"] = status if (status != "ok" or r["ok"]) else "nonfinite"
        res[bn] = r
    return res


def step_freq(vz: np.ndarray, walk: np.ndarray) -> float:
    """Dominant 1-3 Hz frequency of vertical head velocity: Welch PSD (4 s windows) computed within
    each walking bout of >= 4 s, averaged with bout-length weights."""
    acc, wsum, f = None, 0.0, None
    for a, b in runs(walk):
        if b - a < 4 * FS:
            continue
        f, p = welch(vz[a:b] - vz[a:b].mean(), fs=FS, nperseg=int(4 * FS))
        acc = p * (b - a) if acc is None else acc + p * (b - a)
        wsum += b - a
    if acc is None:
        return np.nan
    k = (f >= 1) & (f <= 3)
    return float(f[k][np.argmax(acc[k])])


def load_recording(root: str, rid, window=None, personalized_only: bool = False):
    """Load one recording onto the FS grid. window = (start, end) in s relative to the first gaze
    sample (M3 uses [whisper_start, whisper_end]); None = the whole overlap of gaze and trajectory.
    Returns a dict, or a str failure reason (no_dir | no_gaze_or_traj | not_personalized | few_gaze |
    no_traj | short_window)."""
    d = rio.find_recording_dir(root, rid)
    if d is None:
        return "no_dir"
    gp, gk = rio.gaze_path(d)
    tp, tk = rio.trajectory_path(d)
    if gp is None or tp is None:
        return "no_gaze_or_traj"
    if personalized_only and gk != "personalized":
        return "not_personalized"
    G = rio.load_gaze(gp)
    gv = G[["gx", "gy", "gz"]].to_numpy()
    fin = np.isfinite(gv).all(1)
    if fin.sum() <= MIN_GAZE_OBS:                  # also covers an empty gaze table
        return "few_gaze"
    T = rio.load_trajectory(tp, tk)
    if len(T["t_us"]) < 2:
        return "no_traj"
    tg = G.t_us.to_numpy() * 1e-6
    tt = T["t_us"] * 1e-6
    t0 = tg[0]
    ws, we = window if window is not None else (-np.inf, np.inf)
    tgf = tg[fin]
    lo = max(tg[0], tt[0], t0 + ws)
    hi = min(tg[-1], tt[-1], t0 + we)
    if hi - lo < 30:
        return "short_window"
    t = np.arange(lo, hi, 1 / FS)
    g = np.stack([np.interp(t, tgf, gv[fin, k]) for k in range(3)], 1)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    invalid = obs_invalid(t, tgf)                  # no valid gaze sample within NEAR_S = 50 ms
    gdot = np.gradient(g, t, axis=0)
    Rd = rio.DEFAULT_T_CPF_DEVICE[:3, :3]
    w_cpf = np.stack([np.interp(t, tt, T["w"][:, k]) for k in range(3)], 1) @ Rd.T
    vw = T["v"]
    up = T["up"]
    vup = np.einsum("ni,ni->n", vw, up)
    vh_tr = np.linalg.norm(vw - vup[:, None] * up, axis=1)
    vh = np.interp(t, tt, vh_tr)
    walk = walking_mask(vh)
    # eye (CPF origin) velocity in CPF: R_cpf_dev (R_wd^T v_world) + omega x r, r = CPF origin in device frame
    vb = np.einsum("nji,nj->ni", T["R"], vw)                    # device-frame velocity
    r_c = -Rd.T @ rio.DEFAULT_T_CPF_DEVICE[:3, 3]
    vb = vb + np.cross(T["w"], r_c)
    v_cpf = np.stack([np.interp(t, tt, vb[:, k]) for k in range(3)], 1) @ Rd.T
    dep = G.depth.to_numpy(float)
    dk = np.isfinite(dep) & (dep > 0)
    if dk.sum() > 10:                              # depth used only with > 10 valid depth samples
        td = tg[dk]
        depth = np.interp(t, td, dep[dk])
        depth[obs_invalid(t, td)] = np.nan
    else:
        depth = None
    pitch = np.degrees(np.arctan2(g[:, 1], np.hypot(g[:, 0], g[:, 2])))
    return dict(rid=int(rid), t0=t0, t=t, g=g, gdot=gdot, w_cpf=w_cpf, walk=walk, invalid=invalid, v_cpf=v_cpf,
                depth=depth, vh=vh, vz=np.interp(t, tt, vup), gaze_kind=gk, traj_kind=tk,
                gaze_rate=float(1 / np.median(np.diff(tgf))), pitch_deg=pitch)


def measure(D: dict, thr_deg=40.0, dil=2, depth_fill=None, min_s=None) -> dict:
    """Recording-level M3 measures. depth_fill=None: no translation term (primary). Otherwise the
    translation regressor uses MPS depth (clipped to 0.2-20 m) with missing samples set to depth_fill."""
    depth = None
    if depth_fill is not None:
        depth = np.full(len(D["t"]), float(depth_fill))
        if D["depth"] is not None:
            ok = np.isfinite(D["depth"])
            depth[ok] = np.clip(D["depth"][ok], 0.2, 20)
    res = band_gains(D["t"], D["g"], D["gdot"], D["w_cpf"], D["walk"], D["invalid"], thr_deg, dil,
                     v_cpf=D["v_cpf"] if depth is not None else None, depth=depth, min_s=min_s)
    w = D["walk"]
    res.update(rid=D["rid"], gaze_kind=D["gaze_kind"], traj_kind=D["traj_kind"], gaze_rate=D["gaze_rate"],
               speed=float(np.mean(D["vh"][w])) if w.any() else np.nan,
               gaze_pitch=float(np.mean(D["pitch_deg"][w & ~D["invalid"]])) if (w & ~D["invalid"]).any() else np.nan,
               step_freq=step_freq(D["vz"], w), invalid_frac_walk=float(D["invalid"][w].mean()) if w.any() else np.nan)
    return res
