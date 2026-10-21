"""Event definitions from SLAM speed + existing reading annotations (no new labels).

walking : v_h > 0.5 m/s for >= 2 s
stop    : v_h < 0.15 m/s for >= 1.5 s;  t_s = first sample of the stop run
positive: walking bout >= 4 s ending <= MAX_DECEL_S before t_s, and an annotated
          reading onset t_r in [t_s - 3, t_s + 5]
strict negative: walking-qualified stop whose window [t_s - 5 - W, t_s + 2] lies
          outside every reading interval expanded by 2 s
excluded: every other stop that touches reading (reading-related, onset not associated)
Event definition is offline labelling, so the speed used here is centrally smoothed;
predictive features never use it. Samples without a valid trajectory are neither walking nor
stopped, so trajectory gaps can never create stops; a stop whose approach contains a gap is
not qualified.

Eligible exposure for continuous replay (frozen definition):
  E = walking-bout samples from bout start + W  U  [t_s - 5, t_s + 2] around every qualified stop,
restricted to valid trajectory. FA/h uses the label-0 part of E (eligible, event-free time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WALK_V, WALK_MIN_S = 0.5, 2.0
STOP_V, STOP_MIN_S = 0.15, 1.5
PRE_WALK_S = 4.0
MAX_DECEL_S = 3.0          # implementation detail: walking bout must end <= 3 s before t_s
ONSET_WIN = (-3.0, 5.0)
LEAD_S = 5.0               # I_e = [t_s - 5, t_s - H]
POST_S = 2.0
READ_PAD_S = 2.0
SMOOTH_S = 0.5


def smooth_speed(v_h: np.ndarray, fs: float) -> np.ndarray:
    """Centred running median over finite samples; NaN where the trajectory is invalid."""
    v = pd.Series(np.asarray(v_h, float))
    out = v.rolling(max(1, int(round(SMOOTH_S * fs))), center=True, min_periods=1).median().to_numpy().copy()
    out[~np.isfinite(v.to_numpy())] = np.nan
    return out


def walk_runs(v: np.ndarray, fs: float):
    return [(a, b) for a, b in runs(v > WALK_V) if (b - a) / fs >= WALK_MIN_S]


def stop_runs(v: np.ndarray, fs: float):
    return [(a, b) for a, b in runs(v < STOP_V) if (b - a) / fs >= STOP_MIN_S]


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _overlaps(a0, a1, intervals, pad):
    return any(a0 < f + pad and a1 > s - pad for s, f in intervals)


def detect_events(t: np.ndarray, v_h: np.ndarray, fs: float, reading: list[tuple[float, float]], W: float = 2.0):
    v = smooth_speed(v_h, fs)
    ok = np.isfinite(v)
    walks = [(a, b) for a, b in walk_runs(v, fs)]          # [a, b) sample indices
    stops = stop_runs(v, fs)
    onsets = np.array([s for s, _ in reading])
    rows = []
    for a, b in stops:
        if a == 0 or not ok[a - 1]:
            continue  # recording or valid segment starts stopped: no observed approach
        ts = t[a]
        prev = [w for w in walks if w[1] <= a and (a - w[1]) / fs <= MAX_DECEL_S]
        walk_dur = ((prev[-1][1] - prev[-1][0]) / fs) if prev else 0.0
        gap_free = bool(prev) and ok[prev[-1][0]:a].all()
        qual = walk_dur >= PRE_WALK_S and gap_free and ts - LEAD_S - W >= 0
        rel = onsets - ts if len(onsets) else np.array([])
        assoc = rel[(rel >= ONSET_WIN[0]) & (rel <= ONSET_WIN[1])]
        touches = _overlaps(ts - LEAD_S - W, ts + POST_S, reading, READ_PAD_S)
        if qual and len(assoc):
            cls = "pos"
        elif not touches:
            cls = "neg" if qual else "neg_unqual"
        else:
            cls = "exc"
        rows.append({"t_s": ts, "stop_dur": (b - a) / fs, "walk_dur": walk_dur, "qual": qual,
                     "t_r": float(ts + assoc[np.argmin(np.abs(assoc))]) if len(assoc) else np.nan,
                     "cls": cls})
    return (pd.DataFrame(rows, columns=["t_s", "stop_dur", "walk_dur", "qual", "t_r", "cls"]),
            [(t[a], t[b - 1]) for a, b in walks])


def eligible_mask(t: np.ndarray, v_h: np.ndarray, fs: float, ev: pd.DataFrame, W: float = 2.0) -> np.ndarray:
    """Frozen eligible exposure: walking-bout samples from bout start + W, plus [t_s - 5, t_s + 2]
    around every qualified stop (pos, neg or exc with a qualified approach); valid trajectory only."""
    v = smooth_speed(v_h, fs)
    elig = np.zeros(len(t), bool)
    for a, b in walk_runs(v, fs):
        elig[a + int(round(W * fs)):b] = True
    for _, e in ev[ev.qual.astype(bool)].iterrows():
        elig |= (t >= e.t_s - LEAD_S) & (t <= e.t_s + POST_S)
    return elig & np.isfinite(v)


def label_stream(tq: np.ndarray, ev: pd.DataFrame, H: float):
    """Per-time labels for continuous replay: 0 neg, 1 early (I_e), 2 late, 3 excluded.
    Also returns the positive-event index for early/late samples (-1 otherwise)."""
    lab = np.zeros(len(tq), int)
    eid = -np.ones(len(tq), int)
    for _, e in ev[ev.cls == "exc"].iterrows():
        lab[(tq >= e.t_s - LEAD_S) & (tq <= e.t_s + POST_S)] = 3
    for k, e in ev[ev.cls == "pos"].iterrows():
        late = (tq > e.t_s - H) & (tq <= e.t_s + POST_S)
        early = (tq >= e.t_s - LEAD_S) & (tq <= e.t_s - H)
        lab[late], eid[late] = 2, k
        lab[early], eid[early] = 1, k
    return lab, eid


def match_events(df: pd.DataFrame, cols: list[str], caliper: float = 1.0, strata: list[str] = ("pid", "scenario")):
    """Greedy 1:1 nearest-neighbour matching of 'pos' to 'neg' within strata on `cols` standardised
    over the whole frozen pos+neg pool (Euclidean distance / sqrt(len(cols)) <= caliper SD units).
    Returns df with `matched` and `pair`."""
    df = df.copy()
    df["matched"], df["pair"] = False, -1
    X = df[cols].to_numpy(float)
    sd = np.nanstd(X, 0)
    sd[~(sd > 0)] = 1.0
    Z = (X - np.nanmean(X, 0)) / sd
    pair = 0
    for _, g in df.groupby(list(strata), dropna=False):
        P = g.index[g.cls == "pos"].tolist()
        N = g.index[g.cls == "neg"].tolist()
        cand = []
        for i in P:
            for j in N:
                zi, zj = Z[df.index.get_loc(i)], Z[df.index.get_loc(j)]
                if np.all(np.isfinite(zi)) and np.all(np.isfinite(zj)):
                    cand.append((np.linalg.norm(zi - zj) / np.sqrt(len(cols)), i, j))
        used = set()
        for d, i, j in sorted(cand):
            if d > caliper or i in used or j in used:
                continue
            used |= {i, j}
            df.loc[[i, j], "matched"] = True
            df.loc[[i, j], "pair"] = pair
            pair += 1
    return df
