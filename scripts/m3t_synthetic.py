"""M3-T pipeline check (not evidence): a trunk-fixed hand-held target with known gaze behaviour.
World head angular velocity omega = omega_trunk + omega_rel (CPF). The hand moves with the trunk, so in the
head frame every hand point moves as -omega_rel x x, plus a small own gait-locked hand translation. The hand
pose is written as MPS-like native frames (30 Hz, device frame, mm-level jitter, xyzw wrist->device quaternion)
and read back through handref.side_on_grid, exactly as the real data path.
Gaze scenarios (slow phase between reading fast phases):
  target  slow phase follows the material point on the hand-held page (kappa_T = 1)
  world   slow phase stabilises gaze in the world (-omega x g, gain 1)
  mix     0.5 target + 0.5 world
Reading fast phases: 3/s half-sine, 5 samples, re-fixating a moving point on the page (horizontal offset
cycling over +-8 deg, i.e. forward saccades with return sweeps).
Expected: target -> kappa_H ~ 1 (attenuated by jitter), kappa_W ~ lambda < 1, shift-surrogate means ~ 0,
joint b ~ 1 > a ~ 0; world -> kappa_W ~ 1, joint a ~ 1 > b ~ 0.
trunk_share is the share of the gait-band head yaw carried by the trunk (and so by the hand); trunk_share = 0
is labelled "zero gait-yaw trunk share" (the slow yaw and the pitch terms still move with the trunk).
Calibration gate (Codex round-1 probe, process/reviews/m3t_prefreeze_synthetic_probe.py): an exact
continuous-time 50:50 mixture of the hand-translation and world-rotation predictions at 1.0 / 1.8 / 2.0 /
3.0 Hz must give J_ok and |b - a| <= CAL_TOL, i.e. the matched hand / gaze differentiation meets the registered
tolerance in the clean calibration. Outputs: results/m3t/m3t_synthetic.json (rows), results/m3t/m3t_synthetic_summary.json"""
import json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa import handref as hr
from ritw_kappa import slowphase as sp

FS = sp.FS
CAL_TOL = 0.05
CAL_FREQS = (1.0, 1.8, 2.0, 3.0)


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def simulate(scenario, seed, T=600.0, jitter_mm=0.7, frag=False, trunk_share=0.5, dropout_rate=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(T * FS))) / FS
    n = len(t)
    walk = np.ones(n, bool)
    if frag:
        walk[:] = False
        i = 0
        while i < n:
            on = int(rng.uniform(15, 60) * FS); walk[i:i + on] = True; i += on + int(rng.uniform(3, 15) * FS)
    f = 1.8 + 0.1 * rng.standard_normal()
    ph = rng.uniform(0, 2 * np.pi, 6)
    gw = walk.astype(float)
    yaw_g = np.radians(15) * gw * np.sin(np.pi * f * t + ph[0])
    pit_g = np.radians(8) * gw * np.sin(2 * np.pi * f * t + ph[1])
    yaw_s = np.radians(30) * np.sin(2 * np.pi * 0.2 * t + ph[2])
    w = np.zeros((n, 3))
    w[:, 1] = yaw_g + yaw_s
    w[:, 0] = pit_g + np.radians(6) * np.sin(2 * np.pi * 0.3 * t + ph[3])
    w += np.radians(1.0) * rng.standard_normal(w.shape)
    w_tr = np.zeros((n, 3))
    w_tr[:, 1] = trunk_share * yaw_g + yaw_s
    w_tr[:, 0] = 0.3 * pit_g + np.radians(6) * np.sin(2 * np.pi * 0.3 * t + ph[3])
    w_rel = w - w_tr
    # hand: wrist position / orientation in CPF, page point offset in the wrist frame
    p0 = np.array([-0.08, -0.22, 0.25])
    off_w = np.array([0.0, 0.03, 0.06])
    v_own = 0.004 * 2 * np.pi * f * gw[:, None] * np.stack([np.sin(2 * np.pi * f * t + ph[4]), np.sin(2 * np.pi * f * t + ph[5]), 0 * t], 1)
    p = np.zeros((n, 3)); p[0] = p0
    R = np.zeros((n, 3, 3)); R[0] = np.eye(3)
    for i in range(1, n):
        wh = -w_rel[i - 1]
        p[i] = p[i - 1] + (np.cross(wh, p[i - 1]) + v_own[i - 1] - 0.3 * (p[i - 1] - p0)) / FS
        R[i] = Rot.from_rotvec(wh / FS).as_matrix() @ R[i - 1]
        R[i] = R[i] - 0.3 / FS * (R[i] - np.eye(3))       # weak recentring (bounded drift)
        U, _, Vt = np.linalg.svd(R[i]); R[i] = U @ Vt
    vp = np.gradient(p, t, axis=0)
    # gaze
    page = p + np.einsum("nij,j->ni", R, off_w)
    n_s, dur = 5, 5 / FS
    starts = np.sort(rng.choice(np.arange(60, n - 60), int(3.0 * T), replace=False))
    sacc_on = np.full(n, -1)
    for k, s in enumerate(starts):
        sacc_on[s:s + n_s] = k
    g = np.zeros((n, 3)); g[0] = page[0] / np.linalg.norm(page[0])
    dirs, amps = {}, {}
    for i in range(1, n):
        gi = g[i - 1]
        d = np.linalg.norm(page[i - 1])
        # velocity of the fixated material point (hand rigid body moving with the trunk) relative to the head
        xP = d * gi
        vP = np.cross(-w_rel[i - 1], xP) + v_own[i - 1] - 0.3 * (p[i - 1] - p0)
        q_t = (vP - vP.dot(gi) * gi) / d
        q_w = -np.cross(w[i - 1], gi)
        v = {"target": q_t, "world": q_w, "mix": 0.5 * q_t + 0.5 * q_w}[scenario]
        k = sacc_on[i - 1]
        if k >= 0:
            if k not in dirs:
                u = page[i - 1] / np.linalg.norm(page[i - 1])
                hoff = np.radians(8) * (((k % 6) / 2.5) - 1)              # forward saccades, return sweep every 6
                tgt = Rot.from_rotvec(hoff * np.array([0, 1.0, 0])).apply(u)
                dd = tgt - gi; dd -= dd.dot(gi) * gi
                amps[k] = float(np.linalg.norm(dd)); dirs[k] = dd / (amps[k] + 1e-12)
            j = i - 1 - starts[k]
            v = v + dirs[k] * amps[k] * (np.pi / (2 * dur)) * np.sin(np.pi * (j + 0.5) / n_s)
        gn = gi + v / FS
        g[i] = gn / np.linalg.norm(gn)
    g = g + np.radians(0.2) * rng.standard_normal(g.shape) * np.array([1, 1, 0])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    gdot = np.gradient(g, t, axis=0)
    # MPS-like native hand frames: 30 Hz, device frame, jitter, optional dropouts
    idx = np.arange(0, n, 2)
    p_dev = (p[idx] - hr.TC) @ hr.Rd + jitter_mm / 1000 * rng.standard_normal((len(idx), 3))
    R_dev = np.einsum("ij,njk->nik", hr.Rd.T, R[idx])
    conf = np.ones(len(idx))
    if dropout_rate > 0:
        for s0 in rng.choice(np.arange(30, len(idx) - 60), int(dropout_rate * T), replace=False):
            conf[s0:s0 + int(rng.uniform(0.1, 1.5) * 30)] = -1
    fr = {"t": t[idx], "left": {"conf": conf, "p_dev": p_dev, "q": Rot.from_matrix(R_dev).as_quat()},
          "right": {"conf": -np.ones(len(idx)), "p_dev": p_dev, "q": Rot.from_matrix(R_dev).as_quat()}}
    side, Gs = hr.primary_side(fr, t, walk)
    D = {"t": t, "g": g, "gdot": gdot, "w_cpf": w, "walk": walk, "invalid": np.zeros(n, bool), "depth": None}
    out = {"scenario": scenario, "seed": seed, "jitter_mm": jitter_mm, "frag": frag, "dropout_rate": dropout_rate,
           "trunk_share": trunk_share, "side": side}
    out.update(hr.measure_hand(D, Gs[side], mode="trans"))
    rr = hr.measure_hand(D, Gs[side], mode="rigid", n_shift=0, world=False, joint=False)
    out["Hrigid_kappa_sp"], out["Hrigid_ok"] = rr["H_kappa_sp"], rr["H_ok"]
    return out


def calibration(frequency, drop_every=0):
    """Equal-mixture probe (copied from process/reviews/m3t_prefreeze_synthetic_probe.py). drop_every = k > 0
    drops every k-th native hand frame (brief-dropout stress test, Codex round 2; reported, no criterion)."""
    t = np.arange(120 * 60) / 60
    n = len(t)
    zero = np.zeros(n)
    om = 2 * np.pi * frequency
    A = 0.01
    gx, gy = A * np.sin(om * t), A * np.cos(om * t)
    g = np.stack([gx, gy, np.sqrt(1 - gx ** 2 - gy ** 2)], 1)
    gdot_true = np.stack([A * om * np.cos(om * t), -A * om * np.sin(om * t), zero], 1)
    p = np.stack([0.4 * 2 * A * np.sin(om * t), zero, 0.4 + zero], 1)
    vel = np.stack([0.4 * 2 * A * om * np.cos(om * t), zero, zero], 1)
    qh = (vel - np.sum(vel * g, 1)[:, None] * g) / np.linalg.norm(p, axis=1)[:, None]
    w_cpf = np.cross(2 * gdot_true - qh, g)
    fr = {"t": t[::2], "left": {"conf": np.ones(n // 2), "p_dev": (p[::2] - hr.TC) @ hr.Rd,
                                "q": np.tile([0.0, 0.0, 0.0, 1.0], (n // 2, 1))}}
    if drop_every:
        fr["left"]["conf"][drop_every::drop_every] = -1
    D = {"t": t, "g": g, "gdot": np.gradient(g, t, axis=0), "w_cpf": w_cpf, "invalid": np.zeros(n, bool),
         "walk": np.ones(n, bool), "depth": None}
    r = hr.measure_hand(D, hr.side_on_grid(fr, "left", t), n_shift=0)
    d = r["J_b_hand"] - r["J_a_world"]
    return {"frequency_hz": frequency, "drop_every": drop_every, "hand_valid": r["hand_valid_walk"], "a": r["J_a_world"], "b": r["J_b_hand"], "b_minus_a": d, "J_ok": bool(r["J_ok"]),
            "pass": bool(r["J_ok"] and np.isfinite(d) and abs(d) <= CAL_TOL)}


def summarise(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    df["cell"] = df.apply(lambda r: f"{r.scenario}|jit{r.jitter_mm}|frag{r.frag}|drop{r.dropout_rate}|"
                          + ("zero gait-yaw trunk share" if r.trunk_share == 0 else f"trunk{r.trunk_share}"), axis=1)
    out = {}
    for cell, g in df.groupby("cell", sort=False):
        J, H, W = g[g.J_ok.astype(bool)], g[g.H_ok.astype(bool)], g[g.W_ok.astype(bool)]
        e = {"n": int(len(g)), "n_J_ok": int(len(J)), "n_H_ok": int(len(H)), "n_W_ok": int(len(W)),
             "Jshift_n_ok": g.Jshift_n_ok.tolist(), "Hshift_n_ok": g.Hshift_n_ok.tolist()}
        for col, sub in (("J_b_hand", J), ("J_a_world", J), ("d_ba", J), ("Jshift_b_hand", J), ("Jshift_a_world", J),
                         ("J_rho", J), ("H_kappa_sp", H), ("Hshift_kappa_sp", H), ("W_kappa_sp", W), ("lambda", g),
                         ("rho_HW", g), ("rigid_minus_trans", H)):
            x = sub[col].astype(float)
            e[col] = {"mean": float(x.mean()) if x.notna().any() else None,
                      "absmax": float(x.abs().max()) if x.notna().any() else None, "n": int(x.notna().sum())}
        out[cell] = e
    return out


if __name__ == "__main__":
    from joblib import Parallel, delayed
    jobs = [((sc, 100 + s), dict(jitter_mm=jit)) for sc in ("target", "world", "mix") for jit in (0.0, 0.7, 2.0)
            for s in range(3)]
    for s in range(3):
        jobs += [(("target", 300 + s), dict(frag=True, dropout_rate=0.5)), (("target", 400 + s), dict(trunk_share=0.0)),
                 (("target", 500 + s), dict(trunk_share=0.8))]
    rows = Parallel(n_jobs=10)(delayed(simulate)(*a, **k) for a, k in jobs)
    for r in rows:
        r["d_ba"] = r["J_b_hand"] - r["J_a_world"]
        r["rigid_minus_trans"] = r["Hrigid_kappa_sp"] - r["H_kappa_sp"]
    cal = [calibration(f) for f in CAL_FREQS]
    cal_drop = [calibration(f, k) for k in (15, 6) for f in CAL_FREQS]
    os.makedirs("results/m3t", exist_ok=True)
    keys = ["scenario", "jitter_mm", "frag", "trunk_share", "seed", "W_kappa_sp", "H_kappa_sp", "Hrigid_kappa_sp", "lambda",
            "J_a_world", "J_b_hand", "Hshift_kappa_sp", "Hshift_n_ok", "Jshift_b_hand", "Jshift_a_world", "Jshift_n_ok", "J_ok", "J_rho", "rho_HW", "W_status", "H_status"]
    for r in rows:
        print({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in keys}, flush=True)
    conv = lambda x: x.item() if hasattr(x, "item") else str(x)
    json.dump(rows, open("results/m3t/m3t_synthetic.json", "w"), indent=1, default=conv)
    summ = {"calibration_equal_mixture": {"tol": CAL_TOL, "all_pass": all(c["pass"] for c in cal), "cells": cal},
            "brief_dropout_stress_reported": cal_drop,
            "cells": summarise(rows)}
    json.dump(summ, open("results/m3t/m3t_synthetic_summary.json", "w"), indent=1, default=conv)
    print(json.dumps(summ["calibration_equal_mixture"], default=conv))
