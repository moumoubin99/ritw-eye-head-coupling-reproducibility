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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
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
             dropout_rate=0.0, kappa_period=None, kappa_alt=None, kappa_nonwalk=0.5, wnoise_dps=0.0):
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
    w_meas = w + np.radians(wnoise_dps) * np.random.default_rng(seed + 2 * 10 ** 6).standard_normal(w.shape)
    r = band_gains(t, g, gdot, w_meas, walk, invalid)
    r["head_speed_p90"] = float(np.degrees(np.percentile(np.linalg.norm(w, axis=1), 90)))
    r["realized_events"] = int(len(runs(sacc_on >= 0)))
    return r



# POST HOC (after the M3 task 11/12 results): does lower head-motion amplitude bias kappa_sp downward?
# head_scale scales the true head rotation; wnoise_dps adds measurement noise to the omega regressor only
# (errors in variables). Not part of the frozen plan.
import json
from joblib import Parallel, delayed
def one(hs, wn, rate, sd):
    r = simulate(1.0, rate, 700 + sd, head_scale=hs, wnoise_dps=wn)["gait"]
    return dict(head_scale=hs, wnoise_dps=wn, rate=rate, seed=sd, status=r.get("status"), kappa_sp=r.get("kappa_sp"),
                kappa_raw=r.get("kappa_raw"), P=r.get("P"))
jobs = [(hs, wn, rate, sd) for hs in (1.0, 0.5, 0.33) for wn in (0.0, 1.0, 3.0) for rate in (1.7, 2.8) for sd in range(3)]
R = Parallel(n_jobs=10)(delayed(one)(*j) for j in jobs)
import pandas as pd
df = pd.DataFrame(R)
print(df.groupby(["head_scale", "wnoise_dps", "rate"]).agg(n_ok=("status", lambda x: (x == "ok").sum()),
      kappa_sp=("kappa_sp", "mean"), kappa_raw=("kappa_raw", "mean"), P=("P", "mean")).round(3))
with open("results/posthoc/m3_attenuation_check.json", "w") as fh:
    json.dump(R, fh, indent=1, default=float)
