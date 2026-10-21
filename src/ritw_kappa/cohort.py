"""Recording objects: real (MPS CSV) or synthetic, with events, features and streams."""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from . import FS
from . import io as rio
from .events import detect_events, eligible_mask, label_stream
from .features import per_sample_features, steady_walking_mask, take
from .signal import build_session, add_bandpassed

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cache")


class Rec:
    def __init__(self, rec_id, pid, scenario, s, reading, meta=None):
        self.rec_id, self.pid, self.scenario, self.s, self.reading = rec_id, pid, scenario, s, reading
        self.meta = meta or {}
        self._f, self._steady, self._ev = {}, {}, {}

    def feats(self, W):
        if W not in self._f:
            self._f[W] = per_sample_features(self.s, W)
        return self._f[W]

    def steady(self, W):
        if W not in self._steady:
            self._steady[W] = steady_walking_mask(self.s, W)
        return self._steady[W]

    def events(self, W):
        if W not in self._ev:
            self._ev[W] = detect_events(self.s["t"], self.s["v_h"], self.s["fs"], self.reading, W)[0]
        return self._ev[W]

    def in_reading(self, t):
        r = np.zeros(len(t), bool)
        for a, b in self.reading:
            r |= (t >= a) & (t <= b)
        return r


SIGNAL_VERSION = "v2-causal"   # bump whenever the signal path changes; old caches are ignored


def _cache_path(root: str, rec_id) -> str:
    slug = os.path.abspath(root).strip(os.sep).replace(os.sep, "_")
    return os.path.join(CACHE, SIGNAL_VERSION, slug, f"{rec_id}.pkl")


def load_real(root: str, rec_id, pid, scenario, rec_ann, t16, use_cache=True) -> Rec | None:
    """Signals are cached per (source root, signal version); participant id, scenario and reading
    intervals always come from the current annotations, never from the cache."""
    cp = _cache_path(root, rec_id)
    reading = rio.reading_intervals(rec_id, rec_ann, t16)
    if use_cache and os.path.exists(cp):
        with open(cp, "rb") as fh:
            s, meta = pickle.load(fh)
        return Rec(rec_id, pid, scenario, s, reading, meta)
    d = rio.find_recording_dir(root, rec_id)
    if d is None:
        return None
    gp, gk = rio.gaze_path(d)
    tp, tk = rio.trajectory_path(d)
    if gp is None or tp is None:
        return None
    gaze = rio.load_gaze(gp)
    traj = rio.load_trajectory(tp, tk)
    s = build_session(gaze, traj)
    for k in ("gdot_bp", "gpred_bp"):
        s[k] = s[k].astype(np.float32)
    tt = (traj["t_us"] - gaze["t_us"].iloc[0]) / 1e6
    meta = {"gaze_kind": gk, "traj_kind": tk, "depth_available": bool(np.isfinite(s["depth"]).any()),
            "depth_coverage": float(np.mean(np.isfinite(s["depth"]))),
            "gaze_missing_rate": float(np.mean(s["gaze_missing"])),
            "traj_coverage": float(np.mean(s["traj_ok"])),
            "start_offset_ms": float(tt[0] * 1e3),           # trajectory start minus first gaze sample
            "overlap_frac": float(np.mean((s["t"] >= tt[0]) & (s["t"] <= tt[-1]))),
            "extrinsics": "default T_CPF_device (official RitW projection_utils); no per-device calibration",
            "source_root": os.path.abspath(root), "signal_version": SIGNAL_VERSION,
            "duration_s": float(s["t"][-1])}
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    with open(cp, "wb") as fh:
        pickle.dump((s, meta), fh)
    return Rec(rec_id, pid, scenario, s, reading, meta)


def stream(rec: Rec, W: float, H: float, dt: float = 0.1) -> dict:
    """Continuous-replay query grid, t >= W. The frozen eligible exposure (events.eligible_mask:
    walking bouts from start + W plus [t_s - 5, t_s + 2] around every QUALIFIED stop, valid
    trajectory only) is the authoritative replay mask; positives are qualified by definition, so
    their windows are inside it. Excluded intervals are labelled within this exposure, never added."""
    s = rec.s
    fs = s["fs"]
    ev = rec.events(W)
    tq = np.arange(W, s["t"][-1], dt)
    el = eligible_mask(s["t"], s["v_h"], fs, ev, W)
    tq = tq[el[np.clip(np.floor(tq * fs + 1e-9).astype(int), 0, len(el) - 1)]]
    lab, eid = label_stream(tq, ev, H)
    X = take(rec.feats(W), tq, fs)
    return {"rec_id": rec.rec_id, "pid": rec.pid, "copy": 0, "scenario": rec.scenario, "t": tq, "lab": lab, "eid": eid, "X": X,
            "in_read": rec.in_reading(tq), "t_s_of": ev.t_s.to_dict(), "n_pos": int((ev.cls == "pos").sum()),
            "dt": dt}


# ---------------------------------------------------------------- synthetic cohort (sanity only)

def _gait_omega(t, v, rng, amp=0.25):
    """Head angular velocity (CPF axes) with gait-band content scaled by walking speed."""
    f = 1.8 + 0.1 * rng.standard_normal()
    ph = rng.uniform(0, 2 * np.pi, 3)
    w = np.stack([np.sin(2 * np.pi * f * t + ph[0]),
                  0.8 * np.sin(np.pi * f * t + ph[1]),
                  0.5 * np.sin(np.pi * f * t + ph[2])], 1)
    scale = amp * np.clip(v / 1.2, 0.05, 1.5)[:, None]
    noise = rng.standard_normal((len(t), 3)) * 0.02
    return w * scale + noise


def synthetic_rec(pid: str, seed: int, effect: float = 0.0, dur: float = 300.0, fs: float = FS,
                  kappa_base: float = 0.9, gaze_noise: float = 0.003, energy_matched: bool = True) -> Rec:
    """RitW-like recording: walk/stop bouts; some stops are followed by reading (positives).
    With effect>0 the true gain drops by `effect` over the 5 s before reading stops. With
    energy_matched, the lost compensatory energy is replaced by uncorrelated gait-band eye motion,
    so eye-only energy features carry (almost) no signal and only coupling does."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    v = np.zeros(len(t))
    reading, stops = [], []
    cur = 3.0
    while cur < dur - 25:
        walk = rng.uniform(6, 14)
        a, b = cur, cur + walk
        ramp = (t >= a) & (t < b)
        v[ramp] = 1.2 * np.clip(np.minimum(t[ramp] - a, b - t[ramp]) / 0.8, 0, 1)
        stop_len = rng.uniform(4, 9)
        ts = b
        is_read = rng.random() < 0.5
        if is_read:
            reading.append((ts + rng.uniform(-1.0, 2.0), ts + stop_len - 0.5))
        stops.append((ts, is_read))
        cur = b + stop_len
    w = _gait_omega(t, v, rng)
    ktrue = np.full(len(t), kappa_base)
    for ts, is_read in stops:
        if is_read and effect:
            m = (t >= ts - 5) & (t < ts + 1)
            ktrue[m] = kappa_base - effect
    from scipy.signal import butter, sosfiltfilt
    nz = sosfiltfilt(butter(2, (0.8, 3.0), btype="bandpass", fs=fs, output="sos"),
                     rng.standard_normal((len(t), 3)), axis=0)
    nz *= np.sqrt(1.5) / np.sqrt(np.mean(np.sum(nz**2, 1)))
    extra = np.sqrt(np.clip(kappa_base**2 - ktrue**2, 0, None)) if energy_matched else 0 * ktrue
    g = np.zeros((len(t), 3))
    g[0] = np.array([0.0, -0.3, 1.0]) / np.linalg.norm([0.0, -0.3, 1.0])
    for i in range(1, len(t)):
        c = np.cross(w[i - 1], g[i - 1])
        gd = -ktrue[i - 1] * c
        if extra[i - 1] > 0:
            nn = nz[i - 1] - np.dot(nz[i - 1], g[i - 1]) * g[i - 1]
            gd += extra[i - 1] * np.linalg.norm(c) * nn
        gd += 0.5 * (np.array([0.0, -0.3, 1.0]) / 1.044 - g[i - 1])  # slow re-centring
        gi = g[i - 1] + gd / fs
        g[i] = gi / np.linalg.norm(gi)
    gm = g + gaze_noise * rng.standard_normal(g.shape)
    gm /= np.linalg.norm(gm, axis=1, keepdims=True)
    gdot = np.full_like(gm, np.nan)
    gdot[1:] = (gm[1:] - gm[:-1]) * fs
    s = {"t": t, "fs": fs, "g": gm, "gdot": gdot, "gpred": -np.cross(w, gm), "w": w,
         "v_eye": np.stack([0 * t, 0.02 * np.sin(2 * np.pi * 1.8 * t), v], 1),
         "v_h": v + 0.02 * rng.standard_normal(len(t)), "turn": 0.05 * rng.standard_normal(len(t)),
         "pitch_h": -0.2 + 0.02 * rng.standard_normal(len(t)), "valid": np.r_[False, np.ones(len(t) - 1, bool)],
         "depth": np.full(len(t), np.nan), "vergence": np.full(len(t), 0.05), "q_width": np.full(len(t), 0.05),
         "pitch_g": np.arcsin(gm[:, 1]), "yaw_g": np.arctan2(gm[:, 0], gm[:, 2]), "gaze_missing": np.zeros(len(t), bool)}
    s["valid"][:int(fs)] = False
    add_bandpassed(s)
    return Rec(f"syn{seed}", pid, 6, s, reading, {"synthetic": True, "effect": effect})


# ---------------------------------------------------------------- real cohort

def load_cohort(root: str, rec_ids, n_jobs: int = 2, use_cache: bool = True, strict: bool = False,
                grouping: str = "date_age"):
    """Load recordings (MPS CSVs under `root`). Returns (recs, missing_ids, errors).
    strict=True (all post-freeze runs): every listed id must load, else raise."""
    from joblib import Parallel, delayed
    rec_ann, t16 = rio.load_annotations(grouping=grouping)
    pid_of = dict(zip(rec_ann.id, rec_ann.pid))
    scen_of = dict(zip(t16.id, t16.scenario))

    def one(i):
        try:
            return load_real(root, i, pid_of.get(i, f"U{i}"), scen_of.get(i), rec_ann, t16, use_cache)
        except Exception as e:  # report, never silently drop
            return ("error", i, repr(e))

    out = Parallel(n_jobs=n_jobs)(delayed(one)(int(i)) for i in rec_ids)
    recs = [r for r in out if isinstance(r, Rec)]
    errors = [r for r in out if isinstance(r, tuple)]
    missing = [int(i) for i, r in zip(rec_ids, out) if r is None]
    if strict and (missing or errors or len(recs) != len(rec_ids)):
        raise RuntimeError(f"frozen cohort changed: missing={missing} errors={errors}")
    return recs, missing, errors
