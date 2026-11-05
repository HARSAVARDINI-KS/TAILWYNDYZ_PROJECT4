"""Validate the forward-scoring procedure (run after run_pipeline.py).

No label exists for Jul-Sep 2024, so the forward scores cannot be scored directly. Two substitutes:
 1. PROTOCOL REPLAY - repeat the exact forward procedure one period earlier, using only what
    was known then: labels derived from activity up to Mar-2024 (snapshot Dec-2023, label Jan-Mar),
    Platt calibrator and threshold from forward-chaining out-of-fold scores, then score the
    Mar-2024 snapshot and compare with the official Apr-Jun 2024 label.
 2. DRIFT CHECK - compare the Jun-2024 scoring snapshot with the Mar-2024 snapshot the model was
    trained on (PSI on key features, share of values outside the training range).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from config import *
from data_prep import load_raw, clean
from features import build_features, feature_columns
from modeling import *

p, a, l, _ = clean(load_raw(), save_audit=False)
cols = feature_columns(["demographic", "lifecycle", "behavioural", "engagement", "support"])

# ---- 1. protocol replay ---------------------------------------------------
# Training labels are derived from activity rows only up to Mar-2024 (what is known when scoring Mar-2024)
Fb = build_features(p, a[a.MONTH <= FEATURE_CUTOFF], BACKTEST_CUTOFF)
Eb = Fb[Fb.eligible == 1].copy(); Eb["y"] = Eb.churn_derived
Fm = build_features(p, a, FEATURE_CUTOFF).join(l.set_index("CUSTOMER_ID").CHURNED)
Em = Fm[Fm.eligible == 1].copy(); Em["y"] = Em.CHURNED.astype(int)
lv = category_levels(pd.concat([Eb, Em]))

import json
params = json.load(open(TAB / "run_summary.json"))["best_params"]
oof = oof_scores("lgbm", Eb, Eb.y, cols, lv, params)
cal = calibrate_platt(oof, Eb.y)
thr = best_f1_threshold(Eb.y.loc[oof.index], cal.predict(oof.values))
model = fit_model("lgbm", Eb, Eb.y, cols, lv, params)
s = cal.predict(model.predict_proba(to_lgbm_frame(Em[cols], lv))[:, 1])

rows = []
for pop, m in (("All eligible", np.ones(len(Em), bool)), ("Active base (bought Feb/Mar-24)", (Em.recency_months <= 1).values)):
    y = Em.y[m]; sc = s[m]
    met = metrics(y, sc, thr)
    rows.append(dict(population=pop, n=int(m.sum()), prevalence=y.mean(), mean_predicted=sc.mean(),
                     brier=brier_score_loss(y, sc), pr_auc=met["pr_auc"], roc_auc=met["roc_auc"],
                     precision=met["precision"], recall=met["recall"], f1=met["f1"], top_decile_lift=met["top_decile_lift"]))
replay = pd.DataFrame(rows); replay.to_csv(TAB / "09_forward_protocol_replay.csv", index=False)

# risk-band hit rates in the active base (the bands used in the forward file)
act = (Em.recency_months <= 1).values
band = np.where(s >= thr, "High", np.where(s >= 0.15, "Medium", "Low"))
bt = pd.DataFrame({"band": band[act], "y": Em.y[act].values}).groupby("band").agg(members=("y", "size"), observed_churn=("y", "mean"), churners=("y", "sum")).reset_index()
bt.to_csv(TAB / "09_forward_replay_band_hit_rates.csv", index=False)
cal_tab = calibration_table(Em.y[act], s[act], bins=5); cal_tab.to_csv(TAB / "09_forward_replay_calibration_active.csv", index=False)

# ---- 2. drift check --------------------------------------------------------
Fs = build_features(p, a, SCORING_CUTOFF); Fs = Fs[Fs.eligible == 1]
def psi(x, y, bins=10):
    x, y = x.dropna(), y.dropna()
    q = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(q) < 3: return np.nan
    q[0], q[-1] = -np.inf, np.inf
    e = np.histogram(x, q)[0] / len(x); o = np.histogram(y, q)[0] / len(y)
    e, o = np.clip(e, 1e-4, None), np.clip(o, 1e-4, None)
    return float(((o - e) * np.log(o / e)).sum())
num = [c for c in cols if c not in ("gender", "city", "membership_tier", "preferred_category", "home_price_tier")]
rows = []
for c in num:
    lo, hi = Em[c].min(), Em[c].max()
    rows.append(dict(feature=c, psi=psi(Em[c], Fs[c]), share_outside_training_range=float(((Fs[c] < lo) | (Fs[c] > hi)).mean()),
                     train_mean=Em[c].mean(), scoring_mean=Fs[c].mean()))
drift = pd.DataFrame(rows).sort_values("psi", ascending=False); drift.to_csv(TAB / "09_forward_drift_check.csv", index=False)

fwd = pd.read_csv(ROOT / "outputs" / "forward_risk_scores_jul_sep_2024.csv")
act_f = fwd[fwd.already_lapsed_2plus_months == 0]
print("REPLAY\n", replay.round(3).to_string()); print(bt.round(3).to_string()); print(cal_tab.round(3).to_string())
print("DRIFT (top 10)\n", drift.head(10).round(3).to_string())
print("share of features with PSI>0.25:", (drift.psi > 0.25).mean().round(3), "| PSI>0.1:", (drift.psi > 0.1).mean().round(3))
print("forward active mean prob", act_f.churn_probability_jul_sep_2024.mean().round(3),
      "| replay active mean predicted", replay.mean_predicted.iloc[1].round(3), "observed", replay.prevalence.iloc[1].round(3))

# ---- 3. effect of clipping scoring features to the training range ---------------------
from scipy.stats import spearmanr
from features import CATEGORICAL
oof2 = oof_scores("lgbm", Em.assign(), Em.y, cols, lv, params)
cal2 = calibrate_platt(oof2, Em.y)
thr2 = best_f1_threshold(Em.y.loc[oof2.index], cal2.predict(oof2.values))
m2 = fit_model("lgbm", Em, Em.y, cols, lv, params)
Fc = Fs.copy()
for c in cols:
    if c not in CATEGORICAL: Fc[c] = Fc[c].clip(Em[c].min(), Em[c].max())
s_raw = cal2.predict(m2.predict_proba(to_lgbm_frame(Fs[cols], lv))[:, 1])
s_clip = cal2.predict(m2.predict_proba(to_lgbm_frame(Fc[cols], lv))[:, 1])
b = lambda x: np.where(x >= thr2, "High", np.where(x >= 0.15, "Medium", "Low"))
act_m = (Fs.recency_months <= 1).values
sens = pd.DataFrame([dict(spearman_rank_corr=spearmanr(s_raw, s_clip)[0], members_changing_band=int((b(s_raw) != b(s_clip)).sum()),
                          active_members_changing_band=int((b(s_raw)[act_m] != b(s_clip)[act_m]).sum()), max_abs_prob_change=float(np.abs(s_raw - s_clip).max()))])
sens.to_csv(TAB / "09_forward_clipping_sensitivity.csv", index=False); print(sens.round(4).to_string())
