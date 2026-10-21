"""M3 pipeline check (not evidence): known slow-phase gain + gait head motion + fast phases.
Fast phases: half-sine velocity, 5 samples (83 ms), amplitude 5-15 deg. Onsets: a fixed count rate*T of
onset samples drawn uniformly without replacement (not a Poisson process); when pulses overlap, the later
onset overwrites the remainder of the earlier one, and merged active intervals are counted as one realized
episode. Direction = head direction (omega x g) with probability p_head = 0.7, else back toward straight ahead.
Gaze dropouts remove observations; the grid validity mask is then built exactly as load_recording builds
it (obs_invalid: no retained observation within 50 ms), and gaze is linearly filled from retained samples.
Dropouts use their own random stream, so the same seed with and without dropouts shares one latent trajectory.
Base grid: kappa in {0.6, 0.9} x rate in {0, 1, 3, 4}/s x 3 seeds, 600 s continuous walking, no dropout.
Stress scenarios: fragmented walking, gaze dropouts, strong head motion (a k = 0 start over-masks; the k = 0
check may fail there), time-varying gain without fast phases.
Start-regime check: low true gains {0, 0.3} x head motion {x1, x2} x
rate {0, 3}; both starts' availability and errors and the registered k = 1 -> k = 0 fallback.
Availability calibration: within-realization change when only observation dropouts are added to the same
latent trajectory (continuous and fragmented walking). A separately labelled behaviour+availability scenario
(different walking pattern, duration and seeds) is descriptive only; neither defines a numerical bound.
Every cell reports its status; failures are counted, not dropped. Output: results/m3_synthetic.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ritw_kappa.slowphase import band_gains, FS, obs_invalid, runs

M = ("kappa_sp", "kappa_sp_first", "kappa_sp_start1", "start_used", "start1_status", "kappa_sp_start0", "kappa_raw", "c", "c_common", "n_iter", "fast_frac",
     "f_ep_rate", "status", "fit_status", "mask_change", "start0_status", "n_Ssp_s")


def walk_pattern(T, rng, frag):
    n = int(round(T * FS))
    if not frag:
        return np.ones(n, bool)
    w = np.zeros(n, bool)
    i = 0
    while i < n:
        on = int(rng.uniform(6, 40) * FS)
        w[i:i + on] = True
        i += on + int(rng.uniform(3, 15) * FS)
    return w


def simulate(kappa, rate, seed, T=600.0, p_head=0.7, noise_deg=0.2, head_scale=1.0, frag=False,
             dropout_rate=0.0, kappa_period=None, kappa_alt=None, kappa_nonwalk=0.5):
    rng = np.random.default_rng(seed)
    t = np.arange(int(round(T * FS))) / FS
    n = len(t)
    walk = walk_pattern(T, rng, frag)
    fs_step = 1.8 + 0.1 * rng.standard_normal()
    ph = rng.uniform(0, 2 * np.pi, 4)
    gait = walk.astype(float)
    w = np.zeros((n, 3))
    w[:, 1] = np.radians(15) * gait * np.sin(np.pi * fs_step * t + ph[0]) + np.radians(30) * np.sin(2 * np.pi * 0.2 * t + ph[1])
    w[:, 0] = np.radians(8) * gait * np.sin(2 * np.pi * fs_step * t + ph[2]) + np.radians(6) * np.sin(2 * np.pi * 0.3 * t + ph[3])
    w += np.radians(1.0) * rng.standard_normal(w.shape)
    w *= head_scale
    kap = np.where(walk, kappa, kappa_nonwalk)      # a different gain between walking bouts
    if kappa_period is not None:                    # time-varying gain while walking
        kap = np.where(walk, np.where((t // kappa_period) % 2 == 0, kappa, kappa_alt), kappa_nonwalk)
    n_s = 5
    dur = n_s / FS
    starts = np.sort(rng.choice(np.arange(60, n - 60), int(rate * T), replace=False)) if rate > 0 else np.array([], int)
    sacc_on = np.full(n, -1)
    for k, s in enumerate(starts):
        sacc_on[s:s + n_s] = k
    amp = np.radians(rng.uniform(5, 15, len(starts)))
    g = np.zeros((n, 3))
    g[0] = [0, 0, 1]
    dirs = {}
    for i in range(1, n):
        gi = g[i - 1]
        v = -kap[i - 1] * np.cross(w[i - 1], gi)
        k = sacc_on[i - 1]
        if k >= 0:
            if k not in dirs:
                d = np.cross(w[i - 1], gi) if rng.random() < p_head else np.array([0, 0, 1.0]) - gi
                d -= d.dot(gi) * gi
                dirs[k] = d / (np.linalg.norm(d) + 1e-12)
            j = i - 1 - starts[k]
            v = v + dirs[k] * amp[k] * (np.pi / (2 * dur)) * np.sin(np.pi * (j + 0.5) / n_s)
        gn = gi + v / FS
        g[i] = gn / np.linalg.norm(gn)
    g = g + np.radians(noise_deg) * rng.standard_normal(g.shape) * np.array([1, 1, 0])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    invalid = np.zeros(n, bool)
    if dropout_rate > 0:                            # gaze dropouts: removed observations, as load_recording sees them
        rng_obs = np.random.default_rng(seed + 10 ** 6)
        removed = np.zeros(n, bool)
        for s0 in rng_obs.choice(np.arange(60, n - 120), int(dropout_rate * T), replace=False):
            removed[s0:s0 + int(rng_obs.uniform(0.05, 1.0) * FS)] = True
        tobs = t[~removed]
        for c in range(3):
            g[:, c] = np.interp(t, tobs, g[~removed, c])
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        invalid = obs_invalid(t, tobs)
    gdot = np.gradient(g, t, axis=0)
    r = band_gains(t, g, gdot, w, walk, invalid)
    r["head_speed_p90"] = float(np.degrees(np.percentile(np.linalg.norm(w, axis=1), 90)))
    r["realized_events"] = int(len(runs(sacc_on >= 0)))
    return r


def row(r, **kw):
    return {**kw, **{f"{b}_{m}": r[b].get(m) for b in ("gait", "slow") for m in M},
            "head_speed_p90": r["head_speed_p90"], "realized_events": r["realized_events"]}


def ok(x):
    return x["gait_status"] == "ok"


def mean(v):
    return float(np.mean(v)) if len(v) else None


# ---- base grid
RATES = (0.0, 1.0, 3.0, 4.0)
base = []
for kappa in (0.6, 0.9):
    for rate in RATES:
        for seed in range(3):
            base.append(row(simulate(kappa, rate, 1000 * seed + int(10 * rate) + int(100 * kappa)),
                            kappa_true=kappa, rate=rate, seed=seed))
            print({k: base[-1][k] for k in ("kappa_true", "rate", "gait_kappa_sp", "gait_kappa_raw", "gait_c_common",
                                            "gait_status", "realized_events")}, flush=True)
est = [x for x in base if ok(x)]
err = max((abs(x["gait_kappa_sp"] - x["kappa_true"]) for x in est), default=np.inf)
# agreement of two successful starts (distinct from base_all_ok, which validates the fallback estimator)
gaps = np.asarray([x["gait_kappa_sp_start1"] - x["gait_kappa_sp_start0"] for x in base], dtype=float)
n_s0_conv = sum(x["gait_start0_status"] == "converged" for x in base)
start_ok = bool(all(x["gait_start1_status"] == "converged" and x["gait_start0_status"] == "converged" for x in base)
                and np.all(np.isfinite(gaps)))
start_gap = float(np.max(np.abs(gaps))) if start_ok else np.inf
raw = {(k, r): mean([x["gait_kappa_raw"] for x in est if x["kappa_true"] == k and x["rate"] == r])
       for k in (0.6, 0.9) for r in RATES}
mono = all(raw[(k, RATES[i])] is not None and raw[(k, RATES[i + 1])] is not None
           and raw[(k, RATES[i])] > raw[(k, RATES[i + 1])] for k in (0.6, 0.9) for i in range(3))

# ---- stress scenarios (kappa 0.9, rate 3/s)
SC = {"fragmented_walking": dict(frag=True, T=900.0),
      "dropouts_0.2_per_s": dict(dropout_rate=0.2),
      "strong_head_x2": dict(head_scale=2.0),
      "fragmented_plus_dropouts": dict(frag=True, T=900.0, dropout_rate=0.2)}
stress = {}
for name, kw in SC.items():
    rows = [row(simulate(0.9, 3.0, 50 + s, **kw), seed=s) for s in range(4)]
    e = [x for x in rows if ok(x)]
    stress[name] = {"n": len(rows), "n_ok": len(e), "statuses": [x["gait_status"] for x in rows],
                    "max_abs_err": max((abs(x["gait_kappa_sp"] - 0.9) for x in e), default=None),
                    "n_start1_converged": sum(x["gait_start1_status"] == "converged" for x in rows),
                    "n_start0_converged": sum(x["gait_start0_status"] == "converged" for x in rows),
                    "max_start_gap_among_both_converged": max((abs(x["gait_kappa_sp_start1"] - x["gait_kappa_sp_start0"]) for x in rows
                                                               if x["gait_start1_status"] == "converged"
                                                               and x["gait_start0_status"] == "converged"), default=None),
                    "mean_fast_frac": mean([x["gait_fast_frac"] for x in e]),
                    "mean_n_Ssp_s": mean([x["gait_n_Ssp_s"] for x in e]), "rows": rows}
    print(name, {k: v for k, v in stress[name].items() if k != "rows"}, flush=True)
# time-varying gain (0.7 / 1.0 alternating every 15 s), no fast phases
tv = [row(simulate(0.7, 0.0, 70 + s, kappa_period=15.0, kappa_alt=1.0), seed=s) for s in range(4)]
tv_ok = [x for x in tv if ok(x)]
stress["time_varying_gain_no_fast"] = {
    "n": len(tv), "n_ok": len(tv_ok), "statuses": [x["gait_status"] for x in tv],
    "max_abs_c": max((abs(x["gait_c"]) for x in tv_ok), default=None),
    "max_abs_c_common": max((abs(x["gait_c_common"]) for x in tv_ok), default=None),
    "mean_kappa_sp": mean([x["gait_kappa_sp"] for x in tv_ok]), "rows": tv}
print("time_varying", {k: v for k, v in stress["time_varying_gain_no_fast"].items() if k != "rows"}, flush=True)
# start-regime check: low true gains, normal and doubled head motion; both starts reported
regime = []
for kappa in (0.0, 0.3):
    for hs in (1.0, 2.0):
        for rate in (0.0, 3.0):
            for sd in range(3):
                x = row(simulate(kappa, rate, 400 + 10 * sd + int(10 * kappa) + int(hs), head_scale=hs),
                        kappa_true=kappa, head_scale=hs, rate=rate, seed=sd)
                regime.append(x)
regime_summary = []
for kappa in (0.0, 0.3):
    for hs in (1.0, 2.0):
        for rate in (0.0, 3.0):
            c = [x for x in regime if x["kappa_true"] == kappa and x["head_scale"] == hs and x["rate"] == rate]
            e = [abs(x["gait_kappa_sp"] - kappa) for x in c if ok(x)]
            e1 = [abs(x["gait_kappa_sp_start1"] - kappa) for x in c if x["gait_start1_status"] == "converged"]
            e0 = [abs(x["gait_kappa_sp_start0"] - kappa) for x in c if x["gait_start0_status"] == "converged"]
            regime_summary.append({"kappa_true": kappa, "head_scale": hs, "rate": rate, "n": len(c),
                                   "status": [x["gait_status"] for x in c], "start_used": [x["gait_start_used"] for x in c],
                                   "max_abs_err": max(e) if e else None,
                                   "start1_status": [x["gait_start1_status"] for x in c],
                                   "start0_status": [x["gait_start0_status"] for x in c],
                                   "start1_max_abs_err": max(e1) if e1 else None,
                                   "start0_max_abs_err": max(e0) if e0 else None})
            print("regime", regime_summary[-1], flush=True)
stress["start_regime_low_gain"] = {"summary": regime_summary, "rows": regime}

# availability calibration: same latent trajectory, observation dropouts added (within realization)
calib = {}
for lab, kw in (("continuous", dict()), ("fragmented", dict(frag=True, T=900.0))):
    pairs = []
    for sd in range(8):
        x0 = row(simulate(0.9, 3.0, 500 + sd, **kw), seed=sd)
        x1 = row(simulate(0.9, 3.0, 500 + sd, dropout_rate=0.2, **kw), seed=sd)
        pairs.append((x0, x1))
    both = [(x0, x1) for x0, x1 in pairs if ok(x0) and ok(x1)]
    d = {"n": len(pairs), "n_both_ok": len(both)}
    for m in ("kappa_sp", "kappa_raw", "c", "c_common", "fast_frac"):
        v = np.array([x1[f"gait_{m}"] - x0[f"gait_{m}"] for x0, x1 in both])
        d[f"delta_{m}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
                           "max": float(v.max())} if len(v) > 1 else None
    calib[lab] = {**d, "rows": [[x0, x1] for x0, x1 in pairs]}
    print("calib", lab, {k: v for k, v in d.items()}, flush=True)
stress["availability_within_realization"] = calib
# behaviour + availability scenario (descriptive): different walking pattern, duration, seeds and dropouts
A = [row(simulate(0.9, 3.0, 200 + s), seed=s) for s in range(6)]
Bc = [row(simulate(0.9, 3.0, 300 + s, frag=True, T=900.0, dropout_rate=0.2), seed=s) for s in range(6)]
Ao, Bo = [x for x in A if ok(x)], [x for x in Bc if ok(x)]
null = {"n_ok_A": len(Ao), "n_ok_B": len(Bo)}
for m in ("kappa_sp", "kappa_raw", "c", "c_common"):
    null[f"delta_{m}"] = (mean([x[f"gait_{m}"] for x in Bo]) - mean([x[f"gait_{m}"] for x in Ao])) if Ao and Bo else None
stress["behaviour_plus_availability_descriptive"] = {**null, "rows_A": A, "rows_B": Bc}
print("behaviour+availability", null, flush=True)

tvs = stress["time_varying_gain_no_fast"]
checks = {
    "base_all_ok": len(est) == len(base), "base_err_lt_0.05": bool(err < 0.05),
    "base_raw_monotone_incl_rate0": bool(mono), "base_start_gap_lt_0.02": bool(start_gap < 0.02),
    **{f"{k}_all_ok_err_lt_0.05": bool(stress[k]["n_ok"] == stress[k]["n"] and stress[k]["max_abs_err"] < 0.05) for k in SC},
    "time_varying_all_ok": tvs["n_ok"] == tvs["n"],
    "time_varying_abs_c_common_lt_0.05": bool(tv_ok and tvs["max_abs_c_common"] < 0.05),
    "start_regime_all_ok_err_lt_0.05": all(x["gait_status"] == "ok" and abs(x["gait_kappa_sp"] - x["kappa_true"]) < 0.05
                                           for x in regime),
    "calib_all_ok": all(calib[k]["n_both_ok"] == calib[k]["n"] for k in calib),
    "calib_abs_delta_kappa_sp_lt_0.05": all(calib[k]["delta_kappa_sp"] is not None and
                                            max(abs(calib[k]["delta_kappa_sp"]["min"]), abs(calib[k]["delta_kappa_sp"]["max"])) < 0.05
                                            for k in calib),
}
out = {"base_rows": base, "max_abs_err_gait_kappa_sp": err, "max_start1_vs_start0": start_gap,
       "n_base_start0_converged": n_s0_conv,
       "mean_gait_kappa_raw": {f"{k}_{r}": v for (k, r), v in raw.items()}, "n_base_not_ok": len(base) - len(est),
       "stress": stress, "checks": checks, "pass": bool(all(checks.values()))}
os.makedirs("results", exist_ok=True)
with open("results/m3_synthetic.json", "w") as fh:
    json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else x)
print("max |err| base:", round(err, 4), "| start gap:", round(start_gap, 5), "| raw means:", {f"{k}_{r}": round(v, 3) for (k, r), v in raw.items() if v is not None})
print("checks:", json.dumps(checks, indent=0), "\nPASS:", out["pass"])
