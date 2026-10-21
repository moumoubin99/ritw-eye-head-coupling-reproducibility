"""Scorers (logistic / shallow GBM) with nested participant-level selection."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONFIGS = [
    ("logit", {"C": 0.1}),
    ("logit", {"C": 1.0}),
    ("gbm", {"n_estimators": 100}),
    ("gbm", {"n_estimators": 300}),
]


def make_model(cfg, seed: int = 0):
    kind, p = cfg
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=p["C"], max_iter=2000))
    import lightgbm as lgb
    return make_pipeline(SimpleImputer(strategy="median"),
                         lgb.LGBMClassifier(n_estimators=p["n_estimators"], learning_rate=0.05, max_depth=2,
                                            num_leaves=4, min_child_samples=10, subsample=1.0,
                                            random_state=seed, verbose=-1, n_jobs=1))


def _usable(X):
    keep = ~np.all(~np.isfinite(X), axis=0)
    return keep


def family(cfg) -> str:
    return cfg[0]


def cfg_key(cfg) -> str:
    return f"{cfg[0]}:" + ",".join(f"{k}={v}" for k, v in sorted(cfg[1].items()))


def fit_model(cfg, Xtr, ytr, seed=0):
    """Returns predict(X) -> P(y=1)."""
    keep = _usable(Xtr)
    if len(np.unique(ytr)) < 2:
        c = float(np.mean(ytr)) if len(ytr) else 0.5
        return lambda X: np.full(len(X), c)
    m = make_model(cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr[:, keep], ytr)

    def predict(X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return m.predict_proba(X[:, keep])[:, 1]
    return predict


def fit_predict(cfg, Xtr, ytr, Xte, seed=0):
    keep = _usable(Xtr)
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), float(np.mean(ytr)) if len(ytr) else 0.5)
    m = make_model(cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr[:, keep], ytr)
        return m.predict_proba(Xte[:, keep])[:, 1]


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def inner_lopo_predictions(cfg, X, y, groups, seed=0, Xeval=None, eval_groups=None):
    """Leave-one-group-out predictions. If Xeval given, predictions are produced for Xeval rows
    (grouped by eval_groups) using models trained on X rows of the other groups."""
    if Xeval is None:
        Xeval, eval_groups = X, groups
    out = np.full(len(Xeval), np.nan)
    for g in np.unique(eval_groups):
        tr = groups != g
        te = eval_groups == g
        if tr.sum() == 0:
            continue
        out[te] = fit_predict(cfg, X[tr], y[tr], Xeval[te], seed)
    return out


def select_config(X, y, groups, configs=CONFIGS, seed=0):
    """Pick the config with the best pooled inner-LOPO AUC (ties -> earlier, simpler config)."""
    best, best_auc, preds = configs[0], -np.inf, {}
    for cfg in configs:
        p = inner_lopo_predictions(cfg, X, y, groups, seed)
        preds[cfg[0] + str(cfg[1])] = p
        ok = np.isfinite(p)
        a = safe_auc(y[ok], p[ok])
        if np.isfinite(a) and a > best_auc + 1e-12:
            best, best_auc = cfg, a
    return best, best_auc
