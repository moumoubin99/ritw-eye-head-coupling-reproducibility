"""Public Figure 3 generator. Requires authorized raw data, regenerated recording-level
tables and an explicitly supplied example recording. The real trace is not bundled.
All fitting operations and frozen-coefficient assertions are retained."""
import argparse
import json, sys
import pandas as pd
from scipy.signal import butter, savgol_filter
from paper_plot_style import *

sys.path.insert(0, path("src"))
from ritw_kappa import io as rio, slowphase as sp, handref as hr

m = pd.read_csv(path("results", "m3t", "m3t_measures_main.csv"))
J = m[(m.task == 12) & (m.J_ok.astype(str).str.lower() == "true")].copy()
J["d"] = J.J_b_hand - J.J_a_world
st = json.load(open(path("results", "m3t", "m3t_stats.json")))
assert len(J) == st["T1_primary_b_minus_a"]["n_rec"] == 14

# ---------- panel a: reload one recording ----------
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--example-recording-id", type=int, required=True,
                    help="An authorized local task-12 recording in the regenerated J_ok cohort")
args = parser.parse_args()
rid = args.example_recording_id
selected = J[J.rid == rid]
if len(selected) != 1:
    parser.error("Example recording must be present once in the eligible regenerated cohort")
row = selected.iloc[0]
HAVE = os.path.exists(os.path.join(rio.find_recording_dir(path("data", "raw", "seattle"), rid),
                                   "mps", "hand_tracking", "hand_tracking_frames.jsonl"))
fig = plt.figure(figsize=(DOUBLE, 4.9))
gs0 = fig.add_gridspec(2, 1, height_ratios=[2, 2.1], hspace=0.3)
gt = gs0[0].subgridspec(2, 1, hspace=0.55)
gs = gs0[1].subgridspec(1, 2, wspace=0.3)
a1 = fig.add_subplot(gt[0]); a2 = fig.add_subplot(gt[1], sharex=a1)
panel_label(a1, "a", x=-0.1, y=1.1)


def panel_a():
    THR, DIL = 40.0, 2
    rec, _ = rio.load_annotations()
    r = rec[rec.id == rid].iloc[0]
    D = sp.load_recording(path("data", "raw", "seattle"), rid, (float(r.whisper_start), float(r.whisper_end)))
    fr = hr.read_frames(rio.find_recording_dir(path("data", "raw", "seattle"), rid))
    side, G = hr.primary_side(fr, D["t"], D["walk"])
    G = G[side]
    t, g = D["t"], D["g"]
    inv = D["invalid"] | G["invalid"]
    qH = hr.gpred_hand(g, G, "trans")
    rr = sp.band_gains(t, g, D["gdot"], D["w_cpf"], D["walk"], inv, THR, DIL, bands=("gait",),
                       v_cpf=-qH, depth=np.ones(len(t)))["gait"]
    B = sp.BANDS["gait"]
    sos = butter(2, [B["lo"], B["hi"]], "bandpass", fs=sp.FS, output="sos")
    gdet = savgol_filter(g, 5, 2, deriv=1, delta=1 / sp.FS, axis=0)
    gpred = -np.cross(D["w_cpf"], g)
    k0 = 0.0 if rr["start_used"] == 0 else 1.0
    fit = sp._fit_sp(gdet, D["gdot"], gpred, inv, D["walk"], sos, B, THR, DIL, 0.5, k0, 50, 1e-3)
    S_ = fit["st"]
    S, segs, F = S_["Ssp"], S_["segs"], S_["F"]
    xHb = sp._bp_segments(sos, qH, segs)
    coef = np.linalg.lstsq(np.c_[S_["h"][S].ravel(), xHb[S].ravel()], S_["e_sp"][S].ravel(), rcond=None)[0]
    err = max(abs(coef[0] - row.J_a_world), abs(coef[1] - row.J_b_hand))
    assert err < 1e-14, err
    print(f"Fig 3a: authorized example, a={coef[0]:.4f} b={coef[1]:.4f}, |diff| vs frozen CSV = {err:.1e}")

    bridge, _ = sp.short_runs(F, 0.5)
    imp = np.zeros(len(t), bool)
    for a0, b0 in bridge:
        imp[a0:b0] = True
    gsp = sp.impute(D["gdot"], gpred, fit["k"], bridge)
    # window: the 6-s span inside the analysed support S (all samples in S or F-in-S) with most imputed runs
    W = int(6 * sp.FS)
    inS = S_["S"]
    best, i0 = -1, None
    for a0, b0 in segs:
        for s0 in range(a0, b0 - W, int(sp.FS / 2)):
            if inS[s0:s0 + W].all():
                n = int(np.sum(np.diff(imp[s0:s0 + W].astype(int)) == 1))
                if n > best:
                    best, i0 = n, s0
    w = slice(i0, i0 + W)
    tw = t[w] - t[i0]
    print(f"Fig 3a window: t = {t[i0] - t[0]:.2f}-{t[i0 + W - 1] - t[0]:.2f} s from window start; imputed runs {best}")
    ax_i = 1  # CPF y (vertical) component: gait-band head pitch dominates during walking
    deg = np.degrees

    a1.plot(tw, deg(D["gdot"][w, ax_i]), color=C["grey"], lw=0.7, label="Measured gaze velocity")
    yi = deg(gsp[w, ax_i]).copy(); yi[~imp[w]] = np.nan
    a1.plot(tw, yi, color=C["read"], lw=1.1, label="Imputed (bridged) samples")
    Fw = F[w]
    for a0, b0 in sp.runs(Fw):
        a1.axvspan(tw[a0] - 0.5 / sp.FS, tw[min(b0, len(tw)) - 1] + 0.5 / sp.FS, color=C["grey"], alpha=0.18, lw=0)
    a1.fill_between([], [], color=C["grey"], alpha=0.18, label="Fast-phase mask")
    a1.set_ylabel("Vertical gaze\nvelocity (deg s$^{-1}$)")
    a1.legend(loc="lower right", ncol=3, bbox_to_anchor=(1.0, 0.98), borderaxespad=0, handlelength=1.4, columnspacing=0.8)
    plt.setp(a1.get_xticklabels(), visible=False)
    a2.plot(tw, deg(S_["e_sp"][w, ax_i]), color=C["black"], lw=1.1, label=r"Slow-phase response $e_{\mathrm{sp}}$")
    a2.plot(tw, deg(S_["h"][w, ax_i]), color=C["world"], lw=0.8, label=r"World predictor $x_W$")
    a2.plot(tw, deg(xHb[w, ax_i]), color=C["wrist"], lw=0.8, label=r"Wrist-translation proxy $x_H$")
    yS = np.where(Fw, np.nan, 0.0)
    a2.set_ylabel("Gait band\n(deg s$^{-1}$)")
    a2.set_xlabel("Time within the illustrative 6-s window (s)")
    a2.legend(loc="lower right", ncol=3, bbox_to_anchor=(1.0, 0.98), borderaxespad=0, handlelength=1.4, columnspacing=0.8)
    return {"a": coef[0], "b": coef[1], "check_abs_err": err,
            "window_s": [float(t[i0] - t[0]), float(t[i0 + W - 1] - t[0])], "component": "CPF y"}


if HAVE:
    meta = panel_a()
else:
    meta = None
    for ax in (a1, a2):
        ax.set_xticks([]); ax.set_yticks([])
    a1.text(0.5, 0.5, f"Panel a pending: hand-tracking file for configured example not present locally",
            transform=a1.transAxes, ha="center", va="center", fontsize=7, color=C["read"])

# ---------- panel b: a vs b ----------
ab = fig.add_subplot(gs[0, 0])
ab.plot([-0.2, 1.1], [-0.2, 1.1], color=C["grey"], lw=0.5, ls="--")
ab.scatter(J.J_a_world, J.J_b_hand, s=12, color=C["diff"], lw=0, alpha=0.85, zorder=2)
ab.scatter([row.J_a_world], [row.J_b_hand], s=30, facecolor="none", edgecolor="k", lw=0.7, zorder=3)
ma, ca, ncl, nrec = cluster_mean_ci(J.J_a_world, J.pid)
mb, cb, _, _ = cluster_mean_ci(J.J_b_hand, J.pid)
t5 = st["T5_descriptive_task12"]
assert abs(ma - t5["J_a_world"]["cluster_mean"]) < 1e-12 and abs(mb - t5["J_b_hand"]["cluster_mean"]) < 1e-12
ab.errorbar(ma, mb, xerr=[[ma - ca[0]], [ca[1] - ma]], yerr=[[mb - cb[0]], [cb[1] - mb]], fmt="s", ms=4,
            color="k", mfc="white", capsize=1.5, lw=0.8, zorder=4)
ab.set_xlim(-0.1, 1.05); ab.set_ylim(-0.05, 1.0)
ab.set_xlabel(r"World-rotation coefficient $a$"); ab.set_ylabel(r"Wrist-proxy coefficient $b$")
ab.text(0.97, 0.03, f"{nrec} rec / {ncl} clusters\n{int((J.d > 0).sum())} of {nrec} above identity",
        transform=ab.transAxes, ha="right", va="bottom", fontsize=6)
panel_label(ab, "b", x=-0.16)

# ---------- panel c: b vs b_shift ----------
ac = fig.add_subplot(gs[0, 1])
for _, q in J.iterrows():
    ac.plot([0, 1], [q.J_b_hand, q.Jshift_b_hand], color=C["grey"], lw=0.5, alpha=0.7)
ac.scatter(np.zeros(len(J)), J.J_b_hand, s=10, color=C["wrist"], lw=0, zorder=2)
ac.scatter(np.ones(len(J)), J.Jshift_b_hand, s=10, color=C["grey"], lw=0, zorder=2)
for x, col in ((0, "J_b_hand"), (1, "Jshift_b_hand")):
    mm, cc, _, _ = cluster_mean_ci(J[col], J.pid)
    ac.errorbar(x + (-0.18 if x == 0 else 0.18), mm, yerr=[[mm - cc[0]], [cc[1] - mm]], fmt="o", ms=4,
                color="k", mfc="white", capsize=2, lw=0.8, zorder=3)
ac.axhline(0, color=C["grey"], lw=0.5, ls=":")
ac.set_xticks([0, 1]); ac.set_xticklabels(["Aligned\n$b$", "Time-shifted\n$b_{\\mathrm{shift}}$"])
ac.set_xlim(-0.45, 1.6); ac.set_ylim(-0.1, 1.0)
ac.set_ylabel(r"Wrist-proxy coefficient")
ac.text(0.97, 0.97, f"{len(J)} rec / {J.pid.nunique()} clusters\nshift surrogates per rec: median {int(J.Jshift_n_ok.median())}",
        transform=ac.transAxes, ha="right", va="top", fontsize=6)
panel_label(ac, "c", x=-0.16)
save(fig, "fig3_m3t_joint")
if meta is None:
    sys.exit(f"Fig 3a NOT produced: missing hand-tracking file for configured example")
json.dump(meta, open(os.path.join(FIG_DIR, "fig3a_trace_meta.json"), "w"), indent=1, default=float)
