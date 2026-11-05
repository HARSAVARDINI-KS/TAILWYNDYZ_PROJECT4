"""Retention-intervention scenarios.

Two complementary views (both are *simulations driven by explicit assumptions*;
the data contain no randomised intervention, so effects are NOT causal estimates):

A. Model what-if   - shift modifiable engagement/support features to the state
                     expected after a successful intervention and read the change
                     in predicted churn probability for the targeted members.
B. Targeting economics - given cross-fitted risk scores for the *active base*
                     (members who bought in Feb/Mar-2024, i.e. still reachable),
                     compare blanket vs risk-targeted campaigns for an assumed
                     relative churn reduction among contacted would-be churners.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SCENARIO, SEED, FIG, TAB
from modeling import fit_predict


# ------------------------------- A. what-if -------------------------------
def service_recovery(X):
    X = X.copy()
    t3, c3 = X.tickets_l3.copy(), X.complaint_l3.copy()
    X["tickets_l3"] = 0
    X["tickets_l6"] = (X.tickets_l6 - t3).clip(lower=0)
    X["tickets_l12"] = (X.tickets_l12 - t3).clip(lower=0)
    X["ticket_change"] = 0
    X["complaint_l3"] = 0
    X["complaint_months_l6"] = (X.complaint_months_l6 - c3).clip(lower=0)
    return X


def engagement_nudge(X):
    X = X.copy()
    X["coupons_l3"] += 1; X["coupons_l6"] += 1
    X["emails_l3"] += 1; X["emails_l6"] += 0.5
    X["app_l3"] += 1; X["app_l6"] += 0.5
    return X


WHAT_IFS = {
    "Proactive service recovery (resolve tickets/complaints)": (
        lambda X: (X.tickets_l3 >= 1) | (X.complaint_l3 >= 1), service_recovery),
    "Engagement nudge (coupon + email + app push)": (
        lambda X: (X.app_change < 0) | (X.emails_l3 == 0), engagement_nudge),
}


def run_what_if(predict_fn, X_active):
    """predict_fn(X) -> calibrated churn probability."""
    rows = []
    base = predict_fn(X_active)
    for name, (sel, shift) in WHAT_IFS.items():
        m = sel(X_active).values
        if m.sum() == 0:
            continue
        after = predict_fn(shift(X_active[m]))
        rows.append(dict(scenario=name, targeted_members=int(m.sum()),
                         share_of_active_base=m.mean(),
                         mean_risk_before=base[m].mean(), mean_risk_after=after.mean(),
                         abs_change_pp=100 * (after.mean() - base[m].mean()),
                         expected_churners_avoided=float((base[m] - after).sum()),
                         relative_change=(after.mean() - base[m].mean()) / base[m].mean()))
    return pd.DataFrame(rows)


# --------------------------- B. targeting economics ------------------------
def cross_fitted_scores(E, cols, cat_levels, params, n_splits=5):
    """Out-of-fold raw scores for every eligible member (simulation use only)."""
    oof = pd.Series(index=E.index, dtype=float)
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(E, E.y):
        oof.iloc[va] = fit_predict("lgbm", E.iloc[tr], E.y.iloc[tr], E.iloc[va], cols, cat_levels, params)
    return oof


def targeting_economics(active, score, sc=SCENARIO):
    """active: active-base frame with y, avg_monthly_spend_l6, tickets_l3; score: risk score aligned to index."""
    n = len(active)
    value = (active.avg_monthly_spend_l6 * sc["margin"] * sc["horizon_months"]).clip(lower=0)
    order = score.loc[active.index].sort_values(ascending=False).index
    pols = {"Blanket (all active members)": active.index}
    for q in (0.10, 0.20, 0.30):
        pols[f"Top {int(q*100)}% by predicted risk"] = order[: int(round(q * n))]
    pols["Rule: >=1 support ticket in last 3m"] = active.index[active.tickets_l3 >= 1]
    rows = []
    for name, idx in pols.items():
        sub = active.loc[idx]
        for r in sc["relative_churn_reduction"]:
            saved = r * sub.y.sum()
            val = r * (sub.y * value.loc[idx]).sum()
            cost = len(sub) * sc["cost_per_contact"]
            rows.append(dict(policy=name, uplift_assumption=r, contacted=len(sub),
                             share_of_active_base=len(sub) / n,
                             churners_in_contacted=int(sub.y.sum()),
                             pct_of_all_churners_reached=sub.y.sum() / active.y.sum(),
                             hit_rate=sub.y.mean(),
                             members_saved=saved, retained_margin_usd=val, campaign_cost_usd=cost,
                             net_benefit_usd=val - cost, roi=(val - cost) / cost))
        # random targeting of the same size (analytic expectation)
    # random baseline with 20% reach, for reference
    k = int(round(0.2 * n))
    for r in sc["relative_churn_reduction"]:
        exp_churn = active.y.mean() * k
        exp_val = active.y.mean() * k * r * (active.y * value).sum() / max(active.y.sum(), 1)
        rows.append(dict(policy="Random 20% (expected)", uplift_assumption=r, contacted=k, share_of_active_base=k / n,
                         churners_in_contacted=exp_churn, pct_of_all_churners_reached=k / n, hit_rate=active.y.mean(),
                         members_saved=r * exp_churn, retained_margin_usd=exp_val,
                         campaign_cost_usd=k * sc["cost_per_contact"],
                         net_benefit_usd=exp_val - k * sc["cost_per_contact"],
                         roi=(exp_val - k * sc["cost_per_contact"]) / (k * sc["cost_per_contact"])))
    return pd.DataFrame(rows)


def scenario_figure(tab):
    t = tab[tab.uplift_assumption == 0.20]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].barh(t.policy[::-1], (t.pct_of_all_churners_reached * 100)[::-1], color="#2A6F97")
    axs[0].set_xlabel("% of all would-be churners reached"); axs[0].set_title("Coverage of at-risk members", loc="left", fontsize=10)
    axs[1].barh(t.policy[::-1], t.net_benefit_usd[::-1], color=["#D1495B" if v < 0 else "#2A6F97" for v in t.net_benefit_usd[::-1]])
    axs[1].set_xlabel("Net benefit (USD, 20% relative churn reduction)"); axs[1].set_title("Campaign economics", loc="left", fontsize=10)
    axs[1].set_yticklabels([])
    fig.tight_layout(); fig.savefig(FIG / "scenario_01_targeting.png"); plt.close(fig)
