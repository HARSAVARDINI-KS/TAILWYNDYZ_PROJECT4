"""Models, time-aware validation and evaluation utilities."""
import itertools
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import SEED
from features import CATEGORICAL

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
def _split_cols(cols):
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in CATEGORICAL]
    return num, cat


def make_logit(cols, C=0.3):
    num, cat = _split_cols(cols)
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=3000))])


def make_rf(cols):
    num, cat = _split_cols(cols)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=SEED))])


LGBM_BASE = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                 class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1)


def make_lgbm(params=None):
    p = dict(LGBM_BASE, n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=30)
    p.update(params or {})
    return LGBMClassifier(**p)


def to_lgbm_frame(X, cat_levels):
    X = X.copy()
    for c in X.columns:
        if c in cat_levels:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def category_levels(F):
    return {c: sorted(F[c].dropna().astype(str).unique()) for c in CATEGORICAL}


def fit_predict(kind, Xtr, ytr, Xte, cols, cat_levels, params=None):
    """Fit one model type on (Xtr, ytr), return P(churn) for Xte."""
    Xtr, Xte = Xtr[cols], Xte[cols]
    if kind == "rule":
        return np.clip(Xte["recency_months"].fillna(0).values, 0, 6) / 6.0
    if kind == "logit":
        m = make_logit(cols).fit(Xtr, ytr)
    elif kind == "rf":
        m = make_rf(cols).fit(Xtr, ytr)
    elif kind == "lgbm":
        m = make_lgbm(params).fit(to_lgbm_frame(Xtr, cat_levels), ytr)
        return m.predict_proba(to_lgbm_frame(Xte, cat_levels))[:, 1]
    else:
        raise ValueError(kind)
    return m.predict_proba(Xte)[:, 1]


def fit_model(kind, Xtr, ytr, cols, cat_levels, params=None):
    """Return a fitted model object (for SHAP / what-if)."""
    Xtr = Xtr[cols]
    if kind == "lgbm":
        return make_lgbm(params).fit(to_lgbm_frame(Xtr, cat_levels), ytr)
    return {"logit": make_logit, "rf": make_rf}[kind](cols).fit(Xtr, ytr)


# --------------------------------------------------------------------------
# Time-aware validation: forward chaining over signup cohorts
# --------------------------------------------------------------------------
def forward_folds(train_df, n_folds=3):
    """Expanding-window folds ordered by signup quarter (earlier cohorts train,
    the next cohort validates). No future cohort ever trains a past one."""
    q = train_df.signup_date.dt.to_period("Q")
    quarters = sorted(q.unique())
    folds = []
    for vq in quarters[-n_folds:]:
        tr = np.where(q < vq)[0]
        va = np.where(q == vq)[0]
        folds.append((tr, va))
    return folds


def oof_scores(kind, Xtr, ytr, cols, cat_levels, params=None, n_folds=3):
    folds = forward_folds(Xtr, n_folds)
    idx, sc = [], []
    for tr, va in folds:
        s = fit_predict(kind, Xtr.iloc[tr], ytr.iloc[tr], Xtr.iloc[va], cols, cat_levels, params)
        idx.append(Xtr.index[va]); sc.append(s)
    return pd.Series(np.concatenate(sc), index=np.concatenate(idx))


def tune_lgbm(Xtr, ytr, cols, cat_levels, n_iter=8):
    """Small random search scored by forward-chaining PR-AUC."""
    grid = dict(num_leaves=[7, 15, 31], learning_rate=[0.03, 0.05], n_estimators=[150, 300],
                min_child_samples=[20, 40, 60])
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    rng = np.random.RandomState(SEED)
    pick = [combos[i] for i in rng.choice(len(combos), size=n_iter, replace=False)]
    rows = []
    for prm in pick:
        o = oof_scores("lgbm", Xtr, ytr, cols, cat_levels, prm)
        y = ytr.loc[o.index]
        rows.append(dict(prm, pr_auc=average_precision_score(y, o), roc_auc=roc_auc_score(y, o)))
    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best = {k: (int(res.loc[0, k]) if k != "learning_rate" else float(res.loc[0, k])) for k in grid}
    return best, res


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.nanargmax(f1))])


def metrics(y, s, thr):
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= thr).astype(int)
    out = dict(n=len(y), prevalence=y.mean(), threshold=thr,
               accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0))
    if 0 < y.sum() < len(y):
        out.update(pr_auc=average_precision_score(y, s), roc_auc=roc_auc_score(y, s))
        k = max(1, int(0.1 * len(y)))
        top = np.argsort(-s)[:k]
        out["top_decile_precision"] = y[top].mean()
        out["top_decile_lift"] = y[top].mean() / max(y.mean(), 1e-9)
    return out


def bootstrap_ci(y, s, fn, n=300, seed=SEED):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(fn(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def calibrate_isotonic(oof, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(oof, y.loc[oof.index])
    return iso


class PlattCalibrator:
    """Monotone sigmoid recalibration of raw model scores (keeps the ranking intact,
    unlike isotonic regression which creates ties on small samples)."""

    @staticmethod
    def _z(s):
        s = np.clip(np.asarray(s, float), 1e-4, 1 - 1e-4)
        return np.log(s / (1 - s)).reshape(-1, 1)

    def fit(self, oof, y):
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(oof.values), y.loc[oof.index])
        return self

    def predict(self, s):
        return self.lr.predict_proba(self._z(s))[:, 1]


def calibrate_platt(oof, y):
    return PlattCalibrator().fit(oof, y)


def paired_bootstrap_diff(y, s_a, s_b, fn=average_precision_score, n=500, seed=SEED):
    """95% CI of metric(a) - metric(b) on the same resamples (paired)."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(fn(y[i], s_a[i]) - fn(y[i], s_b[i]))
    d = np.array(d)
    return float(fn(y, s_a) - fn(y, s_b)), *np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())


def calibration_table(y, p, bins=8):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d.p.rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
