"""Shared style and helpers for the paper figures (Scientific Reports: 88 mm single, 180 mm double column).
Every script reads only the frozen evidence inventory (PAPER_ACCEPTANCE_CONTRACT.md, D2); paths are
resolved relative to the project root so the scripts run from any working directory."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4
SINGLE, DOUBLE = 88 * MM, 180 * MM

matplotlib.rcParams.update({
    "font.size": 7.5, "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
})

# Okabe-Ito (colour-blind safe; distinct in greyscale through marker shape and fill)
C = {"read": "#D55E00", "walk": "#0072B2", "world": "#0072B2", "wrist": "#E69F00", "diff": "#009E73",
     "grey": "#7F7F7F", "black": "#000000", "purple": "#CC79A7", "sky": "#56B4E9"}
COND = {12: "Walking while reading", 11: "Walking, not reading"}


def path(*p):
    return os.path.join(ROOT, *p)


def panel_label(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left",
            family="sans-serif")


def cluster_mean_ci(values, clusters, B=5000, seed=0):
    """Mean of cluster means and a percentile cluster-bootstrap 95% interval (display summary only)."""
    v = np.asarray(values, float)
    cl = np.asarray(clusters)
    ok = np.isfinite(v)
    u = np.unique(cl[ok])
    cm = np.array([v[ok][cl[ok] == k].mean() for k in u])
    rng = np.random.default_rng(seed)
    bm = cm[rng.integers(0, len(cm), (B, len(cm)))].mean(1)
    return float(cm.mean()), (float(np.quantile(bm, .025)), float(np.quantile(bm, .975))), len(cm), int(ok.sum())


def jitter(n, w=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    print("Saved:", os.path.join(FIG_DIR, name + ".pdf"))
