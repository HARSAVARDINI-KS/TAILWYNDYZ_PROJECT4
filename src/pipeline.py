"""End-to-end, reproducible pipeline.   Run:  python src/pipeline.py

Stages: clean -> features -> EDA -> cohort-split model comparison -> imbalance study ->
ablation -> out-of-time backtest -> champion + calibration -> SHAP -> segments ->
retention scenarios -> forward scoring. Everything is written to outputs/ and data/processed/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *            # noqa
from data_prep import load_raw, clean
from features import build_features, feature_columns, GROUPS, CATEGORICAL
from modeling import *          # noqa
from eda import run_eda, rate_table, wilson, C_RET, C_CHURN, C_NEUT
from explain import shap_values, global_importance, top_drivers, shap_figures, nice
from scenario import (run_what_if, cross_fitted_scores, targeting_economics, scenario_figure)

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)

summary = {}

# =========================================================================
# 1. Clean + features
# =========================================================================
raw = load_raw()
p, a, l, audit = clean(raw)
F = build_features(p, a, FEATURE_CUTOFF).join(l.set_index("CUSTOMER_ID").CHURNED)
n_all = len(p)
inelig = F[F.eligible == 0]
extra = [dict(table="label", issue="members with no purchase on/before Mar-2024 (signed up Apr-Jun 2024 or never bought)",
              rows_affected=n_all - int(F.eligible.sum()), action="excluded from modelling population: churn is undefined without prior purchase history"),
         dict(table="label", issue="official CHURNED disagrees with churn re-derived from activity rows (eligible members)",
              rows_affected=int((F[F.eligible == 1].CHURNED != F[F.eligible == 1].churn_derived).sum()),
              action="official label kept (re-derivation used only for the backtest snapshot)")]
audit = pd.concat([audit, pd.DataFrame(extra)], ignore_index=True)
audit.to_csv(TAB / "01_data_quality_audit.csv", index=False)

E = F[F.eligible == 1].copy()
E["y"] = E.CHURNED.astype(int)
E["split"] = np.where(E.signup_date < COHORT_SPLIT, "train", "validation")
E["active_base"] = (E.recency_months <= 1).astype(int)
E.drop(columns=["eligible"]).reset_index().to_csv(PROC / "modelling_dataset.csv", index=False)
tr, va = E[E.split == "train"], E[E.split == "validation"]
ew = (va.recency_months <= 1).values
cat_levels = category_levels(E)
ALL = ["demographic", "lifecycle", "behavioural", "engagement", "support"]
cols_all = feature_columns(ALL)
cols_nolife = feature_columns([g for g in ALL if g != "lifecycle"])
log(f"eligible={len(E)}  train={len(tr)} (churn {tr.y.mean():.1%})  validation={len(va)} (churn {va.y.mean():.1%})  active-base val={ew.sum()} (churn {va.y[ew].mean():.1%})")
summary.update(n_members=n_all, n_eligible=len(E), n_ineligible=n_all - len(E), churn_rate_eligible=E.y.mean(),
               n_train=len(tr), n_val=len(va), churn_train=tr.y.mean(), churn_val=va.y.mean(),
               n_val_active=int(ew.sum()), churn_val_active=va.y[ew].mean(),
               n_active_all=int(E.active_base.sum()), churn_active_all=E[E.active_base == 1].y.mean(),
               lapsed_n=int((E.recency_months >= 2).sum()), lapsed_churn=E.y[E.recency_months >= 2].mean())

# =========================================================================
# 2. EDA
# =========================================================================
segtab, eda_sum = run_eda(E, a, audit)
log("EDA done")

# =========================================================================
# 3. Model comparison (cohort-based split)
# =========================================================================
best_params, tune_tab = tune_lgbm(tr, tr.y, cols_all, cat_levels)
tune_tab.to_csv(TAB / "03_lgbm_tuning_forward_cv.csv", index=False)
log(f"best LGBM params {best_params}")

SPECS = [("Rule: no purchase in last 2 months", "rule", cols_all, None),
         ("Logistic regression", "logit", cols_nolife, None),
         ("Random forest", "rf", cols_all, None),
         ("LightGBM (tuned)", "lgbm", cols_all, best_params)]
comp, comp_ew, scores_val, thrs = [], [], {}, {}
for name, kind, cols, prm in SPECS:
    if kind == "rule":
        thr = 2 / 6
    else:
        oof = oof_scores(kind, tr, tr.y, cols, cat_levels, prm)
        thr = best_f1_threshold(tr.y.loc[oof.index], oof)
    s = fit_predict(kind, tr, tr.y, va, cols, cat_levels, prm)
    scores_val[name], thrs[name] = s, thr
    m = metrics(va.y, s, thr)
    lo, hi = bootstrap_ci(va.y, s, average_precision_score)
    comp.append(dict(model=name, **m, pr_auc_ci_low=lo, pr_auc_ci_high=hi))
    m2 = metrics(va.y[ew], s[ew], thr)
    lo2, hi2 = bootstrap_ci(va.y[ew], s[ew], average_precision_score)
    comp_ew.append(dict(model=name, **m2, pr_auc_ci_low=lo2, pr_auc_ci_high=hi2))
comp, comp_ew = pd.DataFrame(comp), pd.DataFrame(comp_ew)
comp.to_csv(TAB / "03_model_comparison_validation_all.csv", index=False)
comp_ew.to_csv(TAB / "03_model_comparison_validation_active_base.csv", index=False)
# paired bootstrap: is LightGBM's PR-AUC gain over each rival real?
pb = []
for rival in [n for n in scores_val if n != "LightGBM (tuned)"]:
    for pop, msk in (("All validation", np.ones(len(va), bool)), ("Active base", ew)):
        d, lo, hi, p1 = paired_bootstrap_diff(va.y[msk], scores_val["LightGBM (tuned)"][msk], scores_val[rival][msk])
        pb.append(dict(comparison=f"LightGBM minus {rival}", population=pop, pr_auc_diff=d, ci_low=lo, ci_high=hi, share_resamples_lgbm_not_better=p1))
pd.DataFrame(pb).to_csv(TAB / "03_paired_bootstrap_pr_auc_differences.csv", index=False)
log("model comparison done\n" + comp[["model", "precision", "recall", "f1", "pr_auc", "roc_auc"]].round(3).to_string())

# logistic regression sensitivity to lifecycle features (extrapolation under cohort shift)
s_lr_life = fit_predict("logit", tr, tr.y, va, cols_all, cat_levels)
summary["logit_with_lifecycle_prauc"] = average_precision_score(va.y, s_lr_life)
summary["logit_with_lifecycle_ew_prauc"] = average_precision_score(va.y[ew], s_lr_life[ew])

# per-segment performance of the tuned LightGBM (validation)
s_l = scores_val["LightGBM (tuned)"]; thr_l = thrs["LightGBM (tuned)"]
V = va.assign(score=s_l, pred=(s_l >= thr_l).astype(int))
perf_rows = []
for nm, col in [("Membership tier", "membership_tier"), ("City", "city"), ("Signup quarter", None)]:
    keys = V.signup_date.dt.to_period("Q").astype(str) if col is None else V[col]
    for k, g in V.groupby(keys):
        if g.y.sum() < 3 or g.y.sum() == len(g):
            pr = np.nan
        else:
            pr = average_precision_score(g.y, g.score)
        perf_rows.append(dict(segment=nm, level=k, n=len(g), churn_rate=g.y.mean(), pr_auc=pr,
                              recall=(g.pred[g.y == 1].mean() if g.y.sum() else np.nan),
                              precision=(g.y[g.pred == 1].mean() if g.pred.sum() else np.nan)))
pd.DataFrame(perf_rows).to_csv(TAB / "03_lgbm_performance_by_segment.csv", index=False)

# =========================================================================
# 4. Class imbalance handling
# =========================================================================
imb = []
for lab, prm, use_oof in [("Unweighted, threshold 0.50", dict(best_params, class_weight=None), False),
                          ("Unweighted, OOF-tuned threshold", dict(best_params, class_weight=None), True),
                          ("Class-weighted ('balanced'), threshold 0.50", best_params, False),
                          ("Class-weighted + OOF-tuned threshold", best_params, True)]:
    if use_oof:
        o = oof_scores("lgbm", tr, tr.y, cols_all, cat_levels, prm); thr = best_f1_threshold(tr.y.loc[o.index], o)
    else:
        thr = 0.5
    s = fit_predict("lgbm", tr, tr.y, va, cols_all, cat_levels, prm)
    m = metrics(va.y, s, thr)
    m2 = metrics(va.y[ew], s[ew], thr)
    imb.append(dict(strategy=lab, **{k: m[k] for k in ["threshold", "precision", "recall", "f1", "pr_auc"]},
                    active_base_recall=m2["recall"], active_base_precision=m2["precision"], active_base_pr_auc=m2["pr_auc"]))
pd.DataFrame(imb).to_csv(TAB / "04_imbalance_handling.csv", index=False)
log("imbalance study done")

# =========================================================================
# 5. Ablation study (feature groups, LightGBM, identical split/params)
# =========================================================================
abl_spec = [("Demographics only", ["demographic"]),
            ("Demographics + lifecycle", ["demographic", "lifecycle"]),
            ("+ Behavioural", ["demographic", "lifecycle", "behavioural"]),
            ("+ Behavioural + Engagement", ["demographic", "lifecycle", "behavioural", "engagement"]),
            ("+ Behavioural + Support", ["demographic", "lifecycle", "behavioural", "support"]),
            ("All features", ALL),
            ("All minus behavioural", ["demographic", "lifecycle", "engagement", "support"]),
            ("All minus engagement", ["demographic", "lifecycle", "behavioural", "support"]),
            ("All minus support", ["demographic", "lifecycle", "behavioural", "engagement"]),
            ("All minus demographics", ["lifecycle", "behavioural", "engagement", "support"]),
            ("All minus lifecycle", ["demographic", "behavioural", "engagement", "support"]),
            ("Engagement only", ["engagement"]), ("Support only", ["support"])]
abl = []
for nm, g in abl_spec:
    c = feature_columns(g)
    s = fit_predict("lgbm", tr, tr.y, va, c, cat_levels, best_params)
    lo, hi = bootstrap_ci(va.y, s, average_precision_score)
    lo2, hi2 = bootstrap_ci(va.y[ew], s[ew], average_precision_score)
    abl.append(dict(feature_set=nm, n_features=len(c), pr_auc=average_precision_score(va.y, s), pr_auc_ci_low=lo, pr_auc_ci_high=hi,
                    roc_auc=roc_auc_score(va.y, s), active_base_pr_auc=average_precision_score(va.y[ew], s[ew]),
                    active_base_ci_low=lo2, active_base_ci_high=hi2, active_base_roc_auc=roc_auc_score(va.y[ew], s[ew])))
abl = pd.DataFrame(abl); abl.to_csv(TAB / "05_ablation_feature_groups.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (col, lo, hi, ttl, base) in zip(axs, [("pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "All validation members", va.y.mean()),
                                              ("active_base_pr_auc", "active_base_ci_low", "active_base_ci_high", "Active base (bought Feb/Mar-24)", va.y[ew].mean())]):
    y = np.arange(len(abl))[::-1]
    ax.barh(y, abl[col], color=[C_CHURN if n == "All features" else C_RET for n in abl.feature_set], alpha=.9)
    ax.errorbar(abl[col], y, xerr=[abl[col] - abl[lo], abl[hi] - abl[col]], fmt="none", ecolor="#222", capsize=2, lw=1)
    ax.axvline(base, color="#555", ls="--", lw=1); ax.text(base, len(abl) - .4, " prevalence", fontsize=7, color="#555")
    ax.set_title(f"PR-AUC: {ttl}", loc="left", fontsize=10); ax.set_yticks(y); ax.set_yticklabels(abl.feature_set)
plt.tight_layout(); plt.savefig(FIG / "ablation_01_feature_groups.png"); plt.close()
log("ablation done")

# =========================================================================
# 6. Out-of-time backtest  (train on earlier snapshot, test on later one)
# =========================================================================
Fb = build_features(p, a, BACKTEST_CUTOFF)
Eb = Fb[Fb.eligible == 1].copy(); Eb["y"] = Eb.churn_derived
sb = fit_predict("lgbm", Eb, Eb.y, E, cols_all, cat_levels, best_params)
ob = oof_scores("lgbm", Eb.assign(signup_date=Eb.signup_date), Eb.y, cols_all, cat_levels, best_params)
thr_b = best_f1_threshold(Eb.y.loc[ob.index], ob)
ewE = (E.recency_months <= 1).values
bt = []
for nm, sc, th in [("LightGBM trained on Dec-2023 snapshot", sb, thr_b),
                   ("Rule: no purchase in last 2 months", np.clip(E.recency_months.values, 0, 6) / 6, 2 / 6)]:
    for pop, msk in (("All eligible", np.ones(len(E), bool)), ("Active base", ewE)):
        m = metrics(E.y[msk], sc[msk], th); bt.append(dict(model=nm, population=pop, **m))
bt = pd.DataFrame(bt); bt.to_csv(TAB / "05_out_of_time_backtest.csv", index=False)
summary.update(backtest_n_train=len(Eb), backtest_churn_train=Eb.y.mean())
log("backtest done\n" + bt[["model", "population", "n", "pr_auc", "roc_auc", "precision", "recall"]].round(3).to_string())

# =========================================================================
# 7. Champion: calibration, threshold, predictions, SHAP
# =========================================================================
oof = oof_scores("lgbm", tr, tr.y, cols_all, cat_levels, best_params)
iso = calibrate_platt(oof, tr.y)
oof_cal = pd.Series(iso.predict(oof.values), index=oof.index)
thr_cal = best_f1_threshold(tr.y.loc[oof_cal.index], oof_cal)
champ = fit_model("lgbm", tr, tr.y, cols_all, cat_levels, best_params)
def predict_cal(X):
    return iso.predict(champ.predict_proba(to_lgbm_frame(X[cols_all], cat_levels))[:, 1])
p_val = predict_cal(va)
m_champ = metrics(va.y, p_val, thr_cal)
m_champ_ew = metrics(va.y[ew], p_val[ew], thr_cal)
cal_tab = calibration_table(va.y, p_val); cal_tab.to_csv(TAB / "06_calibration_validation.csv", index=False)
cal_tab_ew = calibration_table(va.y[ew], p_val[ew], bins=4)
summary.update(champ_val=m_champ, champ_val_ew=m_champ_ew, thr_cal=thr_cal, brier_val=brier_score_loss(va.y, p_val),
               brier_val_baseline=brier_score_loss(va.y, np.full(len(va), tr.y.mean())),
               best_params=best_params)
cm = confusion_matrix(va.y, (p_val >= thr_cal).astype(int)); summary["confusion_matrix"] = cm.tolist()
cm_ew = confusion_matrix(va.y[ew], (p_val[ew] >= thr_cal).astype(int), labels=[0, 1]); summary["confusion_matrix_active"] = cm_ew.tolist()

sv, Xl = shap_values(champ, va[cols_all], cat_levels)
imp = global_importance(sv, cols_all)
grp_of = {c: g for g, cs in GROUPS.items() for c in cs}
imp["group"] = imp.feature.map(grp_of); imp["label"] = imp.feature.map(nice)
imp.to_csv(TAB / "06_shap_global_importance.csv", index=False)
grp_imp = imp.groupby("group").mean_abs_shap.sum().sort_values(ascending=False)
(grp_imp / grp_imp.sum() * 100).rename("share_pct").to_csv(TAB / "06_shap_importance_by_group.csv")
shap_figures(sv, Xl, cols_all, imp)
# active-base-only importance (where early warning matters)
imp_ew = global_importance(sv[ew], cols_all); imp_ew["group"] = imp_ew.feature.map(grp_of); imp_ew["label"] = imp_ew.feature.map(nice)
imp_ew.to_csv(TAB / "06_shap_global_importance_active_base.csv", index=False)
grp_ew = imp_ew.groupby("group").mean_abs_shap.sum(); (grp_ew / grp_ew.sum() * 100).rename("share_pct").to_csv(TAB / "06_shap_importance_by_group_active_base.csv")
shap_figures(sv[ew], Xl[ew], cols_all, imp_ew, suffix="_active_base", scope="active members, validation cohort")

drv = top_drivers(sv, va[cols_all], cols_all)
band = np.where(p_val >= thr_cal, "High", np.where(p_val >= 0.15, "Medium", "Low"))
pred = pd.DataFrame({"customer_id": va.index, "signup_date": va.signup_date.dt.date.values, "membership_tier": va.membership_tier.values,
                     "city": va.city.values, "months_since_last_purchase": va.recency_months.values,
                     "churn_probability": np.round(p_val, 4), "predicted_label": (p_val >= thr_cal).astype(int),
                     "risk_band": band, "actual_churned": va.y.values}).set_index("customer_id").join(drv)
pred.sort_values("churn_probability", ascending=False).to_csv(ROOT / "outputs" / "churn_predictions_validation_cohort.csv")

# figures: PR curves, calibration, lift
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
cols_pal = {"Rule: no purchase in last 2 months": C_NEUT, "Logistic regression": "#E9A03B", "Random forest": "#5FA55A", "LightGBM (tuned)": C_CHURN}
for ax, msk, ttl in ((axs[0], np.ones(len(va), bool), "All validation members"), (axs[1], ew, "Active base (bought Feb/Mar-24)")):
    for nm, s in scores_val.items():
        pr, rc, _ = precision_recall_curve(va.y[msk], s[msk]); ax.plot(rc, pr, label=nm, color=cols_pal[nm], lw=1.8)
    ax.axhline(va.y[msk].mean(), color="#555", ls="--", lw=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Precision-recall: {ttl}", loc="left", fontsize=10)
axs[0].legend(frameon=False, fontsize=7)
axs[2].plot([0, 1], [0, 1], color="#555", ls="--", lw=1); axs[2].plot(cal_tab.mean_pred, cal_tab.observed, "o-", color=C_CHURN, label="All validation")
axs[2].plot(cal_tab_ew.mean_pred, cal_tab_ew.observed, "s-", color=C_RET, label="Active base")
axs[2].set_xlabel("Predicted probability"); axs[2].set_ylabel("Observed churn rate"); axs[2].set_title("Calibration (Platt-scaled)", loc="left", fontsize=10); axs[2].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "model_01_pr_calibration.png"); plt.close()

# confusion matrix figure
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
for ax, c, ttl in ((axs[0], cm, "All validation"), (axs[1], cm_ew, "Active base")):
    ax.imshow(c, cmap="Blues"); ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Pred retain", "Pred churn"]); ax.set_yticklabels(["Retained", "Churned"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(c[i, j]), ha="center", va="center", color="white" if c[i, j] > c.max() / 2 else "black", fontsize=12)
    ax.set_title(ttl, fontsize=10, loc="left"); ax.grid(False)
plt.tight_layout(); plt.savefig(FIG / "model_02_confusion.png"); plt.close()
log("champion + SHAP done")

# =========================================================================
# 8. Segment risk (observed vs predicted) + adjusted odds ratios
# =========================================================================
seg_pred = []
for nm, key in [("Membership tier", va.membership_tier), ("City", va.city), ("Signup quarter", va.signup_date.dt.to_period("Q").astype(str))]:
    d = pd.DataFrame({"k": key.values, "y": va.y.values, "p": p_val}).groupby("k").agg(members=("y", "size"), observed_churn=("y", "mean"), predicted_churn=("p", "mean")).reset_index()
    d.insert(0, "segment", nm); seg_pred.append(d)
pd.concat(seg_pred).to_csv(TAB / "07_validation_observed_vs_predicted_by_segment.csv", index=False)

A = E[E.active_base == 1].copy()
X = pd.DataFrame({"Gold vs Silver": (A.membership_tier == "Gold").astype(float), "Platinum vs Silver": (A.membership_tier == "Platinum").astype(float),
                  "Marketing opt-in": A.marketing_opt_in.astype(float), "Age (per 10 yrs)": A.age / 10,
                  "Tenure (per year)": A.tenure_months / 12, "Recency = 1 month (vs 0)": (A.recency_months == 1).astype(float),
                  "Support tickets last 6m (per ticket)": A.tickets_l6, "Complaint months last 6m (per month)": A.complaint_months_l6,
                  "App sessions/mo last 6m (per session)": A.app_l6, "Emails opened/mo last 6m (per email)": A.emails_l6,
                  "Coupons redeemed last 6m (per coupon)": A.coupons_l6, "Promo share of txns last 6m (per +10pp)": A.promo_pct_l6.fillna(A.promo_pct_l6.median()) * 10,
                  "Txns last 6m (per 5 txns)": A.txn_l6 / 5, "Txn trend ratio (last 3m / prior 3m)": A.txn_ratio_l3_prev3}, index=A.index)
res = sm.Logit(A.y, sm.add_constant(X)).fit(disp=0, maxiter=200)
ci = res.conf_int()
orr = pd.DataFrame({"odds_ratio": np.exp(res.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]), "p_value": res.pvalues}).drop("const")
orr.to_csv(TAB / "07_adjusted_odds_ratios_active_base.csv")
summary["logit_active_n"] = len(A); summary["logit_active_events"] = int(A.y.sum())
fig, ax = plt.subplots(figsize=(7, 4.4)); o = orr.iloc[::-1]
ax.errorbar(o.odds_ratio, range(len(o)), xerr=[o.odds_ratio - o.ci_low, o.ci_high - o.odds_ratio], fmt="o", color=C_CHURN, ecolor="#555", capsize=2)
ax.axvline(1, color="#333", lw=1); ax.set_yticks(range(len(o))); ax.set_yticklabels(o.index); ax.set_xscale("log")
ax.set_xlabel("Adjusted odds ratio (log scale)"); ax.set_title("Adjusted odds ratios (active members)", loc="left", fontsize=10)
plt.tight_layout(); plt.savefig(FIG / "segments_01_odds_ratios.png"); plt.close()
log("segment analysis done")

# =========================================================================
# 9. Retention scenarios
# =========================================================================
Xa = va[ew]
what = run_what_if(predict_cal, Xa); what.to_csv(TAB / "08_scenario_model_what_if.csv", index=False)
oof_all = cross_fitted_scores(E, cols_all, cat_levels, best_params)
oof_all_cal = pd.Series(iso.predict(oof_all.values), index=oof_all.index)
econ = targeting_economics(E[E.active_base == 1], oof_all_cal)
econ.to_csv(TAB / "08_scenario_targeting_economics.csv", index=False)
scenario_figure(econ)
summary["xfit_active_pr_auc"] = average_precision_score(E.y[E.active_base == 1], oof_all.loc[E.index[E.active_base == 1]])
log("scenarios done\n" + econ[econ.uplift_assumption == 0.2][["policy", "contacted", "pct_of_all_churners_reached", "net_benefit_usd", "roi"]].round(2).to_string())

# =========================================================================
# 10. Forward scoring (Jul-Sep 2024) with model refit on all labelled members
# =========================================================================
champ_all = fit_model("lgbm", E, E.y, cols_all, cat_levels, best_params)
Fs = build_features(p, a, SCORING_CUTOFF); Fs = Fs[Fs.eligible == 1]
for c in cols_all:                       # guard: keep numeric features inside the range seen in training
    if c not in CATEGORICAL and c in E:
        Fs[c] = Fs[c].clip(E[c].min(), E[c].max())
ps = iso.predict(champ_all.predict_proba(to_lgbm_frame(Fs[cols_all], cat_levels))[:, 1])
svs, _ = shap_values(champ_all, Fs[cols_all], cat_levels)
drs = top_drivers(svs, Fs[cols_all], cols_all)
fwd = pd.DataFrame({"customer_id": Fs.index, "membership_tier": Fs.membership_tier.values, "city": Fs.city.values,
                    "months_since_last_purchase": Fs.recency_months.values, "churn_probability_jul_sep_2024": np.round(ps, 4),
                    "predicted_label": (ps >= thr_cal).astype(int),
                    "risk_band": np.where(ps >= thr_cal, "High", np.where(ps >= 0.15, "Medium", "Low")),
                    "already_lapsed_2plus_months": (Fs.recency_months >= 2).astype(int)}).set_index("customer_id").join(drs)
fwd.sort_values("churn_probability_jul_sep_2024", ascending=False).to_csv(ROOT / "outputs" / "forward_risk_scores_jul_sep_2024.csv")
summary["fwd_n"] = len(fwd); summary["fwd_active_n"] = int((fwd.already_lapsed_2plus_months == 0).sum())
summary["fwd_active_high"] = int(((fwd.already_lapsed_2plus_months == 0) & (fwd.risk_band == "High")).sum())
summary["fwd_active_medium"] = int(((fwd.already_lapsed_2plus_months == 0) & (fwd.risk_band == "Medium")).sum())

def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o
(ROOT / "outputs" / "tables" / "run_summary.json").write_text(json.dumps(_clean(summary), indent=2, default=str))
log("No error")