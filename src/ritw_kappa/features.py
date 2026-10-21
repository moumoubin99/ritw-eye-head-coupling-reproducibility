"""Per-sample trailing-window features (all causal) and fold-level kappa calibration.

B1 head : v_mean, a_mean, turn_abs, pitch_h, head_bp
B2 eye  : gpitch, gyaw, gspeed, gaze_bp, logdepth, vergence
B3      : B1 + B2 + {m, logP, q_width, q_missing}
B4      : B3 + kappa
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import smooth_speed, walk_runs
from .signal import trailing_mean, window_stats, kappa_from_stats

B1 = ["v_mean", "a_mean", "turn_abs", "pitch_h", "head_bp"]
B2 = ["gpitch", "gyaw", "gspeed", "gaze_bp", "logdepth", "vergence"]
B3_EXTRA = ["m", "logP", "q_width", "q_missing"]
SETS = {"B1": B1, "B2": B2, "B3": B1 + B2 + B3_EXTRA, "B4": B1 + B2 + B3_EXTRA + ["kappa"]}
P_MIN_PCT = 10.0
EPS = 1e-6


def per_sample_features(s: dict, W: float) -> pd.DataFrame:
    fs = s["fs"]
    n = int(round(W * fs))
    ws = window_stats(s, W)
    # causal 0.25 s smoothing of speed before differencing for acceleration
    vs, _ = trailing_mean(s["v_h"], max(1, int(0.25 * fs)), min_frac=0.5)
    acc = np.full_like(vs, np.nan)
    acc[1:] = np.diff(vs) * fs
    tm = lambda x, mf=0.5: trailing_mean(x, n, min_frac=mf)[0]
    gsp = np.linalg.norm(s["gdot"], axis=1)
    depth = s["depth"]
    f = pd.DataFrame({
        "v_mean": tm(s["v_h"]), "a_mean": tm(acc), "turn_abs": tm(np.abs(s["turn"])),
        "pitch_h": tm(s["pitch_h"]), "head_bp": ws["Hb"],
        "gpitch": tm(s["pitch_g"]), "gyaw": tm(s["yaw_g"]), "gspeed": tm(gsp),
        "gaze_bp": ws["E"], "logdepth": tm(np.log(np.clip(depth, 0.05, 50))) if np.isfinite(depth).any() else np.nan,
        "vergence": tm(s["vergence"]),
        "logP": np.log(ws["P"] + EPS), "q_width": tm(s["q_width"]),
        "q_missing": trailing_mean(s["gaze_missing"].astype(float), n, min_frac=0.0)[0],
        "S": ws["S"], "P": ws["P"], "frac_valid": ws["frac_valid"],
    })
    f["v_end"] = vs                              # used only for matching
    return f


def steady_walking_mask(s: dict, W: float) -> np.ndarray:
    """Inside a walking bout (>= 2 s) with the whole window inside the bout."""
    fs = s["fs"]
    v = smooth_speed(s["v_h"], fs)
    m = np.zeros(len(v), bool)
    for a, b in walk_runs(v, fs):
        m[a + int(round(W * fs)):b] = True
    return m


class FoldKappa:
    """P_min = 10th percentile of steady-walking P in the training recordings;
    lambda = 0.05 * P_min; kappa0 = median kappa over training steady walking with m = 1."""

    def __init__(self, lam_frac: float = 0.05):
        self.lam_frac = lam_frac

    def fit(self, S_steady: np.ndarray, P_steady: np.ndarray):
        ok = np.isfinite(S_steady) & np.isfinite(P_steady)
        S_steady, P_steady = S_steady[ok], P_steady[ok]
        self.P_min = float(np.percentile(P_steady, P_MIN_PCT))
        _, m, k = kappa_from_stats(S_steady, P_steady, self.P_min, 0.0, self.lam_frac)
        self.kappa0 = float(np.median(k[m > 0])) if (m > 0).any() else 0.0
        return self

    @property
    def lam(self):
        return self.lam_frac * self.P_min

    def transform(self, df: pd.DataFrame, S: np.ndarray | None = None) -> pd.DataFrame:
        """Adds kappa and m. `S` optionally replaces the observed S column (surrogate runs)."""
        S = df["S"].to_numpy(float) if S is None else np.asarray(S, float)
        kf, m, _ = kappa_from_stats(S, df["P"].to_numpy(float), self.P_min, self.kappa0, self.lam_frac)
        df = df.copy()
        df["kappa"], df["m"] = kf, m
        return df


class KappaPool:
    """Steady-walking (S, P) per participant cluster, so a FoldKappa can be fitted for any
    training set with bootstrap multiplicity (counts: pid -> number of copies)."""

    def __init__(self, recs, W):
        self.SP = {}
        for r in recs:
            f, m = r.feats(W), r.steady(W)
            S0, P0 = self.SP.get(r.pid, (np.empty(0), np.empty(0)))
            self.SP[r.pid] = (np.r_[S0, f["S"].to_numpy()[m]], np.r_[P0, f["P"].to_numpy()[m]])
        self._cache = {}

    def fit(self, counts: dict) -> FoldKappa:
        key = tuple(sorted((p, c) for p, c in counts.items() if c > 0))
        if key not in self._cache:
            S = np.concatenate([np.tile(self.SP[p][0], c) for p, c in key])
            P = np.concatenate([np.tile(self.SP[p][1], c) for p, c in key])
            self._cache[key] = FoldKappa().fit(S, P)
        return self._cache[key]


def take(f: pd.DataFrame, t_query: np.ndarray, fs: float) -> pd.DataFrame:
    """Causal sampling: value at the last 60 Hz sample <= t."""
    idx = np.clip(np.floor(np.asarray(t_query) * fs + 1e-9).astype(int), 0, len(f) - 1)
    out = f.iloc[idx].reset_index(drop=True)
    out["t"] = np.asarray(t_query)
    out["_i"] = idx
    return out
