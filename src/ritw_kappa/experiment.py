"""E1 (matched stops), E2 (continuous replay), refit cluster bootstrap, V2 surrogates.

Participant handling: every E1 row / E2 stream carries its original participant cluster `pid`
and a bootstrap `copy` index (0 outside the bootstrap). Outer folds AND inner groups are `pid`,
so copies of one participant never straddle a split. Bootstrap multiplicity enters model fitting
as duplicated rows and the fold kappa calibration through KappaPool counts.

Matching identity: participant clusters are merge-only (several people may share one), so they
are valid split/bootstrap groups but not person identities; matched pairs are formed within a
RECORDING (one wearer), never across recordings of a cluster.

Nesting: the kappa calibration (P_min, lambda, kappa0, mask m) -- and, in surrogate runs, the
nuisance generator (kappa0, sigma) -- is refitted on the training participants of EVERY inner
split, and again on the full outer-training set for the outer prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .events import match_events, label_stream
from .features import SETS, KappaPool, take
from .models import CONFIGS, cfg_key, family, fit_model, safe_auc
from .replay import curve, hit_at, thresholds_for, calibrate_threshold, operate
from .signal import trailing_mean, causal_bp, kappa_from_stats
from .cohort import stream as make_stream

MATCH_COLS = ["turn_abs", "v_m4", "v_m3", "v_m2", "v_m1"]      # turning + approach-speed curve
MATCH_SENS_COLS = MATCH_COLS + ["pitch_h", "vergence"]          # R018 sensitivity
TRAIN_STRIDE_S = 0.5  # label-0 samples subsampled to 2 Hz for fitting (scoring stays at 10 Hz)


def _counts(pids, copies) -> dict:
    c = {}
    for p, _ in set(zip(pids, copies)):
        c[p] = c.get(p, 0) + 1
    return c


def _without(counts, g):
    return {p: c for p, c in counts.items() if p != g}


def _ckey(counts):
    return tuple(sorted(counts.items()))


def _chunks(n, k):
    k = max(1, min(k, n))
    return [list(range(i, n, k)) for i in range(k)]


# ------------------------------------------------------------------ E1

def e1_rows(recs, W, H, caliper=1.0, match_cols=MATCH_COLS):
    rows = []
    for r in recs:
        ev = r.events(W)
        ev = ev[ev.cls.isin(["pos", "neg"])]
        if not len(ev):
            continue
        f, fs = r.feats(W), r.s["fs"]
        X = take(f, ev.t_s.to_numpy() - H, fs)
        vs = f["v_end"].to_numpy()
        for k in (1, 2, 3, 4):
            X[f"v_m{k}"] = vs[np.clip(np.floor((ev.t_s.to_numpy() - k) * fs + 1e-9).astype(int), 0, len(vs) - 1)]
        X["cls"], X["t_s"], X["rec_id"], X["pid"], X["scenario"] = ev.cls.values, ev.t_s.values, r.rec_id, r.pid, r.scenario
        X["copy"] = 0
        rows.append(X)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = match_events(df, list(match_cols), caliper=caliper, strata=["rec_id"])
    df["y"] = (df.cls == "pos").astype(int)
    return df


def run_e1(df, pool: KappaPool, sets=("B1", "B2", "B3", "B4"), seed=0, configs=CONFIGS, fixed_cfg=None,
           s_surr=None):
    """Outer leave-one-participant-cluster-out with nested (inner LOPO) config selection.
    fixed_cfg: {(fold, set): cfg} reuses configs (sensitivity runs); a fold whose fixed config is
    absent is reported unavailable (no predictions), never re-tuned. s_surr: callable(counts) ->
    {rec_id: S array at 60 Hz}, replaces the observed S (V2-E1). Returns (res, pred_df, chosen)."""
    d = df[df.matched].reset_index(drop=True)
    pid = d.pid.to_numpy()
    cp = d["copy"].to_numpy() if "copy" in d else np.zeros(len(d), int)
    y = d.y.to_numpy()
    rid, ii = d.rec_id.to_numpy(), d["_i"].to_numpy()
    cache = {}

    def prep(counts):
        key = _ckey(counts)
        if key not in cache:
            fk = pool.fit(counts)
            S = None
            if s_surr is not None:
                Sd = s_surr(counts)
                S = np.array([Sd[r][i] for r, i in zip(rid, ii)])
            cache[key] = fk.transform(d, S)
        return cache[key]

    pred = {k: np.full(len(d), np.nan) for k in sets}
    chosen, unavailable = {}, []
    for f in np.unique(pid):
        tr, te = pid != f, pid == f
        if len(np.unique(y[tr])) < 2:
            continue
        cnt = _counts(pid[tr], cp[tr])
        if fixed_cfg is not None and any((f, k) not in fixed_cfg for k in sets):
            unavailable.append(str(f))
            continue
        for k in sets:
            cols = SETS[k]
            cfg = fixed_cfg.get((f, k)) if fixed_cfg is not None else None
            if cfg is None:
                best, best_auc = configs[0], -np.inf
                for c in configs:
                    p = np.full(len(d), np.nan)
                    for g in np.unique(pid[tr]):
                        X = prep(_without(cnt, g))[cols].to_numpy(float)
                        a, b = tr & (pid != g), pid == g
                        p[b] = fit_model(c, X[a], y[a], seed)(X[b])
                    ok = tr & np.isfinite(p)
                    au = safe_auc(y[ok], p[ok])
                    if np.isfinite(au) and au > best_auc + 1e-12:
                        best, best_auc = c, au
                cfg = best
            chosen[(f, k)] = cfg
            X = prep(cnt)[cols].to_numpy(float)
            pred[k][te] = fit_model(cfg, X[tr], y[tr], seed)(X[te])
    res = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "n_pid": int(len(np.unique(pid))),
           "folds_unavailable": unavailable}
    for k in sets:
        ok = np.isfinite(pred[k])
        res[f"auc_{k}"] = safe_auc(y[ok], pred[k][ok])
        pa = [safe_auc(y[(pid == f) & ok], pred[k][(pid == f) & ok]) for f in np.unique(pid)]
        res[f"auc_pidmean_{k}"] = float(np.nanmean(pa)) if np.isfinite(pa).any() else np.nan
    if "B4" in sets and "B3" in sets:
        res["dAUC"] = res["auc_B4"] - res["auc_B3"]
        res["dAUC_pidmean"] = res["auc_pidmean_B4"] - res["auc_pidmean_B3"]
    return res, d.assign(**{f"p_{k}": pred[k] for k in sets}), chosen


def boot_resample(units, rng):
    """Cluster bootstrap over participants. Returns list of (original pid, copy index)."""
    pick = rng.choice(units, size=len(units), replace=True)
    seen, out = {}, []
    for p in pick:
        seen[p] = seen.get(p, 0) + 1
        out.append((p, seen[p]))
    return out


def e1_bootstrap(df, pool, B=200, seed=0, sets=("B3", "B4"), n_jobs=2, fixed_cfg=None):
    """Refit cluster bootstrap of dAUC. fixed_cfg (sensitivities) is reused by ORIGINAL cluster id,
    so the bootstrap replicates the same frozen-configuration estimator as the point estimate."""
    from joblib import Parallel, delayed
    d0 = df[df.matched]
    units = np.unique(d0.pid)
    by = {p: d0[d0.pid == p] for p in units}

    def chunk(bs):
        out = []
        for b in bs:
            rng = np.random.default_rng(seed + 1000 + b)
            db = pd.concat([by[p].assign(copy=c) for p, c in boot_resample(units, rng)], ignore_index=True)
            out.append((b, run_e1(db, pool, sets, seed, fixed_cfg=fixed_cfg)[0].get("dAUC", np.nan)))
        return out

    res = Parallel(n_jobs=n_jobs)(delayed(chunk)(c) for c in _chunks(B, n_jobs))
    vals = np.full(B, np.nan)
    for part in res:
        for b, v in part:
            vals[b] = v
    return vals


# ------------------------------------------------------------------ E2

def build_streams(recs, W, H, dt=0.1):
    return [make_stream(r, W, H, dt) for r in recs]


def _train_mask(st, stride=TRAIN_STRIDE_S):
    lab = st["lab"]
    k = int(round(stride / st["dt"]))
    sub = (np.arange(len(lab)) % k) == 0
    return (lab == 1) | ((lab == 0) & sub)


def _mat(st, cols, fk, S_by_rec=None):
    """Feature matrix of one stream with kappa/m filled by FoldKappa `fk` (optionally surrogate S)."""
    key = tuple(cols)
    base = st.setdefault("_base", {})
    if key not in base:
        base[key] = st["X"].reindex(columns=list(cols)).to_numpy(float)
    M = base[key].copy()
    X = st["X"]
    S = X["S"].to_numpy(float) if S_by_rec is None else S_by_rec[st["rec_id"]][X["_i"].to_numpy()]
    kf, m, _ = kappa_from_stats(S, X["P"].to_numpy(float), fk.P_min, fk.kappa0, fk.lam_frac)
    if "kappa" in cols:
        M[:, cols.index("kappa")] = kf
    if "m" in cols:
        M[:, cols.index("m")] = m
    return M


def _stack_train(streams, cols, fk, S_by_rec=None):
    Xs, ys = [], []
    for st in streams:
        M = _train_mask(st)
        Xs.append(_mat(st, cols, fk, S_by_rec)[M])
        ys.append((st["lab"][M] == 1).astype(int))
    return np.vstack(Xs), np.concatenate(ys)


def _pred_streams(model, streams, cols, fk, S_by_rec=None):
    if not streams:
        return []
    Ms = [_mat(st, cols, fk, S_by_rec) for st in streams]
    p = model(np.vstack(Ms))
    out, o = [], 0
    for M in Ms:
        out.append(p[o:o + len(M)]); o += len(M)
    return out


def run_e2_pair(streams, pool: KappaPool, seed=0, configs=CONFIGS, fixed=None, b3_fixed=None, s_surr=None):
    """Outer LOPO continuous scorers for B3 and B4 with a COMMON model family per outer fold.

    Inner LOPO (nested kappa calibration per inner split) gives full-stream held-out predictions
    for every (method, config). Family rule (prespecified, revised after code review round 2): the
    family of the best B3 config by pooled inner AUC (label 0 vs 1) -- it depends on B3 only, so
    it is invariant when only kappa is replaced and V2 surrogates may legitimately reuse it with
    the observed B3 scores. B4 then takes its best config inside that family; each method's
    threshold is calibrated to 2 FA/h on its own inner held-out streams.
    fixed: {(fold, method): cfg} reuse (sensitivities and their bootstraps); a fold missing a
    fixed config is marked unavailable and dropped from both methods' summaries, never re-tuned.
    b3_fixed: observed {"scored_b3", "info"} -- B3 scores and per-fold family reused (V2).
    Returns (scored_b3, scored_b4, info)."""
    pid = np.array([st["pid"] for st in streams])
    cp = np.array([st.get("copy", 0) for st in streams])
    methods = ["B4"] if b3_fixed is not None else ["B3", "B4"]
    out = {m: [dict(st) for st in streams] for m in methods}
    info = {}
    for f in np.unique(pid):
        tr_idx, te_idx = np.flatnonzero(pid != f), np.flatnonzero(pid == f)
        train = [streams[i] for i in tr_idx]
        if sum(st["n_pos"] for st in train) == 0:
            continue
        cnt = _counts(pid[tr_idx], cp[tr_idx])
        if fixed is not None and any((f, m) not in fixed for m in ("B3", "B4")):
            info[f] = {"unavailable": True}
            for m in methods:
                for i in te_idx:
                    out[m][i]["unavailable"] = True
            continue
        if fixed is not None:
            cands = {m: [fixed[(f, m)]] for m in methods}
        elif b3_fixed is not None:
            fam = b3_fixed["info"][f]["family"]
            cands = {m: [c for c in configs if family(c) == fam] for m in methods}
        else:
            cands = {m: list(configs) for m in methods}
        # inner held-out predictions, per (method, cfg), aligned with `train`
        ip = {(m, cfg_key(c)): [None] * len(train) for m in methods for c in cands[m]}
        tr_pid = pid[tr_idx]
        for g in np.unique(tr_pid):
            cg = _without(cnt, g)
            fk = pool.fit(cg)
            Sg = s_surr(cg) if s_surr is not None else None
            a_st = [st for st, p in zip(train, tr_pid) if p != g]
            b_pos = [j for j, p in enumerate(tr_pid) if p == g]
            for m in methods:
                cols = SETS[m]
                Sm = Sg if m == "B4" else None
                Xa, ya = _stack_train(a_st, cols, fk, Sm)
                for c in cands[m]:
                    model = fit_model(c, Xa, ya, seed)
                    for j, pr in zip(b_pos, _pred_streams(model, [train[j] for j in b_pos], cols, fk, Sm)):
                        ip[(m, cfg_key(c))][j] = pr
        lab_full = np.concatenate([st["lab"] for st in train])
        sel = np.isin(lab_full, [0, 1])
        auc = {}
        for (m, k), parts in ip.items():
            p = np.concatenate(parts)
            ok = sel & np.isfinite(p)
            auc[(m, k)] = safe_auc((lab_full[ok] == 1).astype(int), p[ok])
        fams = []
        for m in methods:
            for c in cands[m]:
                if family(c) not in fams:
                    fams.append(family(c))

        def best_in(m, fam):
            cs = [c for c in cands[m] if family(c) == fam]
            a = [auc[(m, cfg_key(c))] for c in cs]
            a = [x if np.isfinite(x) else -np.inf for x in a]
            j = int(np.argmax(a)) if cs else None
            return (cs[j], a[j]) if cs else (None, -np.inf)

        if b3_fixed is not None:
            fam = b3_fixed["info"][f]["family"]
        elif fixed is not None:
            fam = family(fixed[(f, "B3")])
        else:
            fam = max(fams, key=lambda fa: (best_in("B3", fa)[1], -fams.index(fa)))
        fi = {"family": fam, "inner_auc": {f"{m}|{k}": float(v) for (m, k), v in auc.items()}}
        fk_out = pool.fit(cnt)
        S_out = s_surr(cnt) if s_surr is not None else None
        test = [streams[i] for i in te_idx]
        for m in methods:
            cfg = best_in(m, fam)[0]
            inner = [dict(st, score=pr) for st, pr in zip(train, ip[(m, cfg_key(cfg))])]
            theta, th_approx = calibrate_threshold(inner)
            cols = SETS[m]
            Sm = S_out if m == "B4" else None
            Xtr, ytr = _stack_train(train, cols, fk_out, Sm)
            model = fit_model(cfg, Xtr, ytr, seed)
            for i, pr in zip(te_idx, _pred_streams(model, test, cols, fk_out, Sm)):
                out[m][i].update(score=pr, theta=theta, cfg=cfg)
            fi[f"cfg_{m}"], fi[f"theta_{m}"], fi[f"theta_approx_{m}"] = cfg, float(theta), bool(th_approx)
        info[f] = fi
    for m in methods:
        for st in out[m]:
            st.pop("_base", None)
            st.setdefault("score", np.full(len(st["lab"]), -np.inf))
            st.setdefault("theta", np.inf)
    s3 = b3_fixed["scored_b3"] if b3_fixed is not None else out["B3"]
    if b3_fixed is not None:
        bad = {st["rec_id"] for st in out["B4"] if st.get("unavailable")}
        s3 = [dict(st, unavailable=True) if st["rec_id"] in bad else st for st in s3]
    return s3, out["B4"], info


def summarize_method(sc):
    sc = [st for st in sc if not st.get("unavailable")]
    if not sc:                                     # every stream unavailable: NaN, never a silent 0
        return {"hit_at2": np.nan, "thresholds_approx": False, "unavailable": True}, [], (None, None, None)
    th, approx = thresholds_for(sc)
    fa, hit = curve(sc, th)
    op = operate(sc, None)
    r = {"hit_at2": hit_at(fa, hit), "thresholds_approx": approx}
    r.update({f"deploy_{k}": v for k, v in op.items() if k != "leads"})
    return r, op["leads"], (th, fa, hit)


def e2_summary(scored_b3, scored_b4, b3_summary=None):
    """b3_summary: reuse a precomputed B3 summary (invariant across surrogate replicates)."""
    res, curves = {}, {}
    for name, sc in (("B3", scored_b3), ("B4", scored_b4)):
        if name == "B3" and b3_summary is not None:
            r, leads, cv = b3_summary
        else:
            r, leads, cv = summarize_method(sc)
        res.update({f"{k}_{name}": v for k, v in r.items()})
        res[f"leads_{name}"] = leads
        curves[name] = cv
    res["T_E2"] = res["hit_at2_B4"] - res["hit_at2_B3"]
    res["n_streams_unavailable"] = int(sum(bool(st.get("unavailable")) for st in scored_b4))
    return res, curves


def _boot_streams(streams, rng):
    units = np.unique([st["pid"] for st in streams])
    out = []
    for p, c in boot_resample(units, rng):
        out += [dict(st, copy=c) for st in streams if st["pid"] == p]
    return out


def e2_bootstrap(streams, pool, B=200, seed=0, n_jobs=2, fixed=None):
    """Refit cluster bootstrap of T_E2 (fixed configs, if given, reused by original cluster id).
    Returns (T values, approx flags) -- approx marks replicates whose threshold sweep or
    calibration was approximate."""
    from joblib import Parallel, delayed

    def chunk(bs):
        out = []
        for b in bs:
            rng = np.random.default_rng(seed + 5000 + b)
            s3, s4, info = run_e2_pair(_boot_streams(streams, rng), pool, seed, fixed=fixed)
            r = e2_summary(s3, s4)[0]
            ap = (r["thresholds_approx_B3"] or r["thresholds_approx_B4"]
                  or any(v.get(f"theta_approx_{m}", False) for v in info.values() for m in ("B3", "B4")))
            out.append((b, r["T_E2"], bool(ap)))
        return out

    res = Parallel(n_jobs=n_jobs)(delayed(chunk)(c) for c in _chunks(B, n_jobs))
    vals, approx = np.full(B, np.nan), np.zeros(B, bool)
    for part in res:
        for b, v, ap in part:
            vals[b], approx[b] = v, ap
    return vals, approx


# ------------------------------------------------------------------ V2 surrogate

def phase_randomize(x, rng):
    """Multivariate phase randomisation (same random phases for all columns)."""
    n = len(x)
    F = np.fft.rfft(x, axis=0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape[0]))
    ph[0] = 1.0
    if n % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(F * ph[:, None], n=n, axis=0)


def depth_series(rec, d_const=None):
    """Constant depth, or measured MPS depth with causal forward fill (leading gap: first value)."""
    n = len(rec.s["t"])
    if d_const is not None:
        return np.full(n, float(d_const))
    d = pd.Series(rec.s["depth"]).ffill().bfill().to_numpy()
    return np.clip(d, 0.2, 20.0)


class SurrogateGen:
    """V2 null generator in the RAW (pre-filter) domain, then the observed signal path:

        gdot_surr = kappa0 * gpred + trans + sigma(t) * eps~,   trans = -(I - g g^T) v_eye / d
        eps~ = phase-randomised (resid / sigma),  resid = gdot - kappa0 * gpred - trans
        log sigma(t) = beta . [1, log P, v_h, q_width]   (least squares of log rms(resid, 0.5 s)
                                                          on event-free valid training samples)
        S_surr = trailing mean_W < BP(gdot_surr), BP(gpred) >     (BP = the causal band-pass)

    (kappa0, beta) are refitted for every training set; random phases are fixed per
    (replicate, recording) so all splits of one replicate see the same surrogate noise draw."""

    def __init__(self, recs, W, H, d_const=None, win_s=0.5, sub=6):
        self.W, self.recs = W, {r.rec_id: r for r in recs}
        self.c = {}
        for r in recs:
            s, f = r.s, r.feats(W)
            g, v = s["g"], s["v_eye"]
            d = depth_series(r, d_const)
            trans = -(v - np.sum(v * g, 1, keepdims=True) * g) / d[:, None]
            bad = ~np.isfinite(s["gdot"]).all(1)
            A = np.nan_to_num(s["gdot"] - trans)
            Bm = np.nan_to_num(s["gpred"])
            trans = np.nan_to_num(trans)
            n = int(win_s * s["fs"])
            va = s["valid"]
            mAA = trailing_mean(np.sum(A * A, 1), n, va)[0]
            mAB = trailing_mean(np.sum(A * Bm, 1), n, va)[0]
            mBB = trailing_mean(np.sum(Bm * Bm, 1), n, va)[0]
            P = f["P"].to_numpy()
            qw = s["q_width"]
            lab, _ = label_stream(s["t"], r.events(W), H)
            ok = (lab == 0) & va & np.isfinite(P) & np.isfinite(s["v_h"]) & np.isfinite(mAA)
            if np.isfinite(qw).any():
                ok &= np.isfinite(qw)
            Pf = np.where(np.isfinite(P), P, np.nanmedian(P) if np.isfinite(P).any() else 1e-3)
            qf = np.where(np.isfinite(qw), qw, np.nanmedian(qw) if np.isfinite(qw).any() else 0.0)
            Z = np.column_stack([np.ones(len(P)), np.log(Pf + 1e-6), np.nan_to_num(s["v_h"]), qf])
            idx = np.flatnonzero(ok)[::sub]
            self.c[r.rec_id] = {"pid": r.pid, "A": A, "B": Bm, "trans": trans, "bad": bad, "Z": Z,
                                "fit": (Z[idx], mAA[idx], mAB[idx], mBB[idx])}

    def fit(self, counts, fk):
        Zs, ys = [], []
        for c in self.c.values():
            k = counts.get(c["pid"], 0)
            if not k:
                continue
            Z, aa, ab, bb = c["fit"]
            rms2 = aa - 2 * fk.kappa0 * ab + fk.kappa0 ** 2 * bb
            ok = rms2 > 0
            Zs += [Z[ok]] * k
            ys += [0.5 * np.log(rms2[ok])] * k
        beta, *_ = np.linalg.lstsq(np.vstack(Zs), np.concatenate(ys), rcond=None)
        return beta

    def S(self, rec_id, fk, beta, rng):
        c, r = self.c[rec_id], self.recs[rec_id]
        k0 = fk.kappa0
        sig = np.exp(np.clip(c["Z"] @ beta, -20, 20))[:, None]
        eps = (c["A"] - k0 * c["B"]) / sig
        eps[c["bad"]] = 0.0
        gd = k0 * c["B"] + c["trans"] + sig * phase_randomize(eps, rng)
        gd[c["bad"]] = np.nan
        gb = causal_bp(gd, r.s["fs"])
        dot = np.sum(gb * r.s["gpred_bp"], 1)
        return trailing_mean(dot, int(round(self.W * r.s["fs"])), r.s["valid"])[0]


def make_s_surr(gen: SurrogateGen, pool: KappaPool, rep: int, seed=0):
    """callable(counts) -> {rec_id: S_surr}; one replicate (fixed phases per recording).
    Only the scalar fits (kappa0, beta) are cached per training set; the 60 Hz surrogate arrays are
    regenerated on demand and released by the caller after its split, so memory does not grow
    with the number of distinct training sets."""
    base = np.random.default_rng(seed + 90000 + rep)
    seeds = {rid: int(base.integers(1 << 31)) for rid in sorted(gen.c)}
    fits = {}

    def f(counts):
        key = _ckey(counts)
        if key not in fits:
            fk = pool.fit(counts)
            fits[key] = (fk, gen.fit(counts, fk))
        fk, beta = fits[key]
        return {rid: gen.S(rid, fk, beta, np.random.default_rng(seeds[rid])) for rid in gen.c}
    return f


def v2op_rep(streams, pool, gen, rep, obs, seed=0):
    """One surrogate replicate of the full E2 B4 pipeline (B3 scores, B3 summary and per-fold
    family reused from the observed run). Returns (T_surr, approx): approx marks a replicate whose
    B4 threshold sweep or any B4 inner calibration was approximate (B3 is the observed run's)."""
    s_surr = make_s_surr(gen, pool, rep, seed)
    s3, s4, info = run_e2_pair(streams, pool, seed, b3_fixed=obs, s_surr=s_surr)
    r = e2_summary(s3, s4, b3_summary=obs["b3_summary"])[0]
    ap = bool(r["thresholds_approx_B4"] or any(v.get("theta_approx_B4", False) for v in info.values()))
    return r["T_E2"], ap


def v2e1_rep(df, pool, gen, rep, seed=0, fixed_cfg=None):
    s_surr = make_s_surr(gen, pool, rep, seed)
    return run_e1(df, pool, ("B3", "B4"), seed, fixed_cfg=fixed_cfg, s_surr=s_surr)[0]["dAUC"]


# ------------------------------------------------------------------ supporting (R010)

def fixed_prediction_bootstrap(pred, B=2000, seed=0):
    """Participant-cluster bootstrap of dAUC on the frozen outer predictions (no refit; supporting only)."""
    rng = np.random.default_rng(seed)
    units = np.unique(pred.pid)
    groups = {p: pred[pred.pid == p] for p in units}
    out = np.full(B, np.nan)
    for b in range(B):
        d = pd.concat([groups[p] for p in rng.choice(units, len(units))], ignore_index=True)
        ok = np.isfinite(d.p_B3) & np.isfinite(d.p_B4)
        out[b] = safe_auc(d.y[ok], d.p_B4[ok]) - safe_auc(d.y[ok], d.p_B3[ok])
    return out


def gee_kappa(df, pool, covars=("v_mean", "gaze_bp", "logP")):
    """Low-dimensional logistic GEE y ~ kappa + covars, exchangeable within participant,
    Mancl-DeRouen bias-reduced sandwich. kappa from a pooled FoldKappa (descriptive, supporting only)."""
    import statsmodels.api as sm
    d = df[df.matched].reset_index(drop=True)
    d = pool.fit({p: 1 for p in d.pid.unique()}).transform(d)
    cols = ["kappa", *covars]
    d = d.dropna(subset=cols)
    Z = (d[cols] - d[cols].mean()) / d[cols].std(ddof=0).replace(0, 1)
    X = sm.add_constant(Z)
    m = sm.GEE(d.y.astype(float), X, groups=d.pid, family=sm.families.Binomial(),
               cov_struct=sm.cov_struct.Exchangeable())
    r = m.fit(cov_type="bias_reduced")
    return {"beta_kappa_per_sd": float(r.params["kappa"]), "se": float(r.bse["kappa"]),
            "p": float(r.pvalues["kappa"]), "n": int(len(d)), "n_pid": int(d.pid.nunique())}


def cfg_table(chosen: dict) -> dict:
    """JSON-safe {"fold|set": [kind, params]}"""
    return {f"{f}|{k}": [c[0], c[1]] for (f, k), c in chosen.items()}


def cfg_from_table(t: dict) -> dict:
    return {tuple(key.split("|", 1)): (v[0], v[1]) for key, v in t.items()}
