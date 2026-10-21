"""Continuous replay: refractory alarms, four-way alarm scoring, FA/h-early-hit curve, T_E2.

Labels per query time (events.label_stream): 0 neg, 1 early (I_e), 2 late, 3 excluded.
Every alarm starts a 10 s refractory period. An alarm is
  * an early hit for event k if it falls in I_e(k) and k has not been hit yet
    (later alarms in an already-hit I_e count as neither hit nor FA),
  * late if it falls in (t_s - H, t_s + 2],
  * excluded if it falls in an excluded window,
  * a false alarm otherwise.
FA/h denominator: time labelled 0 (eligible, event-free) in hours.
"""
from __future__ import annotations

import numpy as np

REFRACTORY_S = 10.0
TARGET_FAH = 2.0


def fire(tq: np.ndarray, score: np.ndarray, theta: float, refractory: float = REFRACTORY_S) -> np.ndarray:
    above = np.flatnonzero(score >= theta)
    if len(above) == 0:
        return above
    ta = tq[above]
    out, i = [], 0
    while i < len(above):
        out.append(above[i])
        i = np.searchsorted(ta, ta[i] + refractory, side="left")
    return np.asarray(out, int)


def score_alarms(idx, lab, eid, in_read):
    hit_events, late, exc, fa, fa_read, lead_idx = set(), 0, 0, 0, 0, []
    for i in idx:
        L = lab[i]
        if L == 1:
            if eid[i] not in hit_events:
                hit_events.add(eid[i])
                lead_idx.append(i)
        elif L == 2:
            late += 1
        elif L == 3:
            exc += 1
        else:
            fa += 1
            fa_read += int(in_read[i])
    return hit_events, late, exc, fa, fa_read, lead_idx


def stream_counts(st: dict, theta: float):
    idx = fire(st["t"], st["score"], theta)
    hits, late, exc, fa, fa_read, lead_idx = score_alarms(idx, st["lab"], st["eid"], st["in_read"])
    leads = [st["t_s_of"][st["eid"][i]] - st["t"][i] for i in lead_idx]
    return len(hits), late, exc, fa, fa_read, leads


def neg_hours(st):
    return float(np.sum(st["lab"] == 0)) * st["dt"] / 3600.0


def curve(streams: list[dict], thetas: np.ndarray):
    """Returns arrays (fa_per_h, hit_rate) per threshold, pooled over streams."""
    n_pos = sum(st["n_pos"] for st in streams)
    hours = sum(neg_hours(st) for st in streams)
    fa_h, hit = np.zeros(len(thetas)), np.zeros(len(thetas))
    for j, th in enumerate(thetas):
        h = f = 0
        for st in streams:
            c = stream_counts(st, th)
            h += c[0]
            f += c[3]
        fa_h[j] = f / max(hours, 1e-9)
        hit[j] = h / max(n_pos, 1)
    return fa_h, hit


def _n_alarms(streams, theta):
    return sum(len(fire(st["t"], st["score"], theta)) for st in streams)


def thresholds_for(streams, max_fah: float = 2 * TARGET_FAH, n_coarse: int = 400, max_exact: int = 20000):
    """Descending thresholds: a coarse quantile grid plus EVERY distinct score down to a certified
    cut theta*, below which FA/h > max_fah is guaranteed.

    Certificate: greedy refractory firing selects a maximum-cardinality refractory-separated subset
    of the above-threshold samples, so the total alarm count N(theta) is non-decreasing as theta
    falls. Non-FA alarms lie on label != 0 samples and are themselves refractory-separated, so
    their number is at most C = N computed on the label != 0 samples alone (threshold-free).
    Hence FA(theta) >= N(theta) - C, and every theta with N(theta) - C > max_fah * hours has
    FA/h > max_fah. theta* is the highest such distinct score (binary search, N monotone).

    If no exact tail point has FA/h in (TARGET_FAH, max_fah], max_fah is doubled and the cut
    recomputed until one does (or the whole score range is the tail): the upper interpolation
    bracket at TARGET_FAH is then an exact curve point, so no approximation remains from the cut.
    Returns (thresholds, approx). approx=True (reported, never silent) only if the certified tail had
    more than `max_exact` distinct scores and was quantile-subsampled."""
    streams = [st for st in streams if len(st["score"])]
    if not streams:
        return np.array([np.inf]), False
    s = np.concatenate([st["score"] for st in streams])
    s = s[np.isfinite(s)]
    if not len(s):
        return np.array([np.inf]), False
    hours = sum(neg_hours(st) for st in streams)
    C = sum(len(fire(st["t"], (st["lab"] != 0).astype(float), 0.5)) for st in streams)
    u = np.unique(s)[::-1]                         # descending distinct scores
    n_last = _n_alarms(streams, u[-1])
    while True:
        if n_last - C <= max_fah * hours:
            cut_i = len(u)                         # never certified: whole range is the tail
        else:
            lo, hi = 0, len(u) - 1                 # first index with N - C > max_fah * hours
            while lo < hi:
                mid = (lo + hi) // 2
                if _n_alarms(streams, u[mid]) - C > max_fah * hours:
                    hi = mid
                else:
                    lo = mid + 1
            cut_i = lo + 1                         # include theta* itself
        tail = u[:cut_i]
        if len(tail) > max_exact or cut_i == len(u):
            break
        fa_t, _ = curve(streams, tail)
        if ((fa_t > TARGET_FAH) & (fa_t <= max_fah)).any() or not (fa_t <= TARGET_FAH).any():
            break
        max_fah *= 2                               # bracket not yet exact: widen the certified range
    approx = len(tail) > max_exact
    if approx:
        tail = np.unique(np.quantile(tail, np.linspace(0, 1, max_exact)))
    coarse = np.unique(np.quantile(s, np.linspace(0, 1, n_coarse)))
    th = np.concatenate([[np.inf], np.unique(np.concatenate([coarse, tail]))[::-1]])
    return th, bool(approx)


def hit_at(fa_h: np.ndarray, hit: np.ndarray, target: float = TARGET_FAH) -> float:
    """Upper envelope then linear interpolation at `target`:
    collapse equal FA/h values to their max hit rate, cumulative max over increasing FA/h, then
    interpolate between the two distinct FA/h values bracketing `target`. NaN if the curve never
    reaches `target` (no extrapolation)."""
    fa = np.r_[0.0, np.asarray(fa_h, float)]
    h = np.r_[0.0, np.asarray(hit, float)]
    u, inv = np.unique(fa, return_inverse=True)
    hm = np.full(len(u), -np.inf)
    np.maximum.at(hm, inv, h)
    env = np.maximum.accumulate(hm)
    if u[-1] < target:
        return float("nan")
    eq = np.flatnonzero(u == target)
    if len(eq):
        return float(env[eq[0]])
    k = np.flatnonzero(u > target)[0]
    i = k - 1
    w = (target - u[i]) / (u[k] - u[i])
    return float(env[i] + w * (env[k] - env[i]))


def calibrate_threshold(streams, target: float = TARGET_FAH):
    """Lowest threshold whose pooled FA/h on these (inner held-out) streams is <= target.
    Returns (theta, approx) with the threshold-sweep approximation flag."""
    th, approx = thresholds_for(streams)
    fa_h, _ = curve(streams, th)
    ok = np.flatnonzero(fa_h <= target)
    return (float(th[ok[-1]]) if len(ok) else float(np.inf)), approx


def operate(streams, theta=None):
    """Pooled performance at a fixed threshold (deployable-threshold support). An explicit `theta`
    applies to all streams; with theta=None each stream uses its own inner-calibrated `theta`."""
    n_pos = sum(st["n_pos"] for st in streams)
    hours = sum(neg_hours(st) for st in streams)
    H = L = E = F = FR = 0
    leads = []
    for st in streams:
        th = theta if theta is not None else st.get("theta", np.inf)
        h, l, e, f, fr, ld = stream_counts(st, th)
        H += h; L += l; E += e; F += f; FR += fr; leads += ld
    return {"hit_rate": H / max(n_pos, 1), "fa_per_h": F / max(hours, 1e-9), "n_pos": n_pos,
            "hours": hours, "late": L, "excluded": E, "fa": F, "fa_in_reading": FR,
            "lead_median": float(np.median(leads)) if leads else np.nan, "leads": leads}
