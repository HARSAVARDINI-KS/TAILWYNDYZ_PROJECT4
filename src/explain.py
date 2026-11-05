"""SHAP-based driver attribution (global + per customer)."""
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIG, TAB
from modeling import to_lgbm_frame

NICE = {
    "recency_months": "Months since last purchase", "txn_l3": "Transactions (last 3m)", "txn_l6": "Transactions (last 6m)",
    "txn_l12": "Transactions (last 12m)", "spend_l3": "Spend (last 3m)", "spend_l6": "Spend (last 6m)", "spend_l12": "Spend (last 12m)",
    "active_months_l3": "Active months (last 3)", "active_months_l6": "Active months (last 6)",
    "txn_ratio_l3_prev3": "Txn trend (last 3m vs prior 3m)", "spend_ratio_l3_prev3": "Spend trend (last 3m vs prior 3m)",
    "txn_slope_l6": "Txn slope (6m)", "spend_slope_l6": "Spend slope (6m)", "tenure_months": "Tenure (months)",
    "history_months": "Months of history", "tickets_l3": "Support tickets (last 3m)", "tickets_l6": "Support tickets (last 6m)",
    "complaint_months_l6": "Complaint months (6m)", "months_since_ticket": "Months since last ticket",
    "app_l3": "App sessions/mo (3m)", "app_l6": "App sessions/mo (6m)", "emails_l3": "Emails opened/mo (3m)",
    "emails_l6": "Emails opened/mo (6m)", "coupons_l3": "Coupons redeemed (3m)", "coupons_l6": "Coupons redeemed (6m)",
    "promo_pct_l3": "Promo share of txns (3m)", "promo_pct_l6": "Promo share of txns (6m)", "basket_l3": "Avg basket (3m)",
    "membership_tier": "Membership tier", "age": "Age", "marketing_opt_in": "Marketing opt-in",
    "months_with_purchase": "Months with a purchase",
}


def nice(c):
    return NICE.get(c, c.replace("_", " ").capitalize())


def shap_values(model, X, cat_levels):
    Xl = to_lgbm_frame(X, cat_levels)
    sv = shap.TreeExplainer(model).shap_values(Xl)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xl


def global_importance(sv, cols):
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    imp["share_pct"] = 100 * imp.mean_abs_shap / imp.mean_abs_shap.sum()
    return imp.reset_index(drop=True)


def top_drivers(sv, X, cols, k=3):
    """Text drivers per customer, e.g. 'Months since last purchase = 3 (raises risk)'."""
    out = []
    A = np.abs(sv)
    for i in range(len(X)):
        order = np.argsort(-A[i])[:k]
        parts = []
        for j in order:
            v = X.iloc[i][cols[j]]
            v = f"{v:.1f}".rstrip("0").rstrip(".") if isinstance(v, (float, np.floating)) and not pd.isna(v) else v
            parts.append(f"{nice(cols[j])} = {v} ({'raises' if sv[i, j] > 0 else 'lowers'} risk)")
        out.append(parts)
    return pd.DataFrame(out, index=X.index, columns=[f"driver_{i+1}" for i in range(k)])


def shap_figures(sv, Xl, cols, imp, suffix="", scope="validation cohort"):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh([nice(f) for f in top.feature], top.mean_abs_shap, color="#2A6F97")
    ax.set_xlabel("Mean |SHAP| (log-odds)"); ax.set_title(f"Top churn drivers ({scope})", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"shap_01_global{suffix}.png"); plt.close(fig)

    # beeswarm on numeric-coded view
    Xn = Xl.copy()
    for c in Xn.columns:
        if str(Xn[c].dtype) == "category":
            Xn[c] = Xn[c].cat.codes
    Xn.columns = [nice(c) for c in Xn.columns]
    plt.figure(figsize=(7.2, 5.2))
    shap.summary_plot(sv, Xn, max_display=14, show=False, plot_size=None)
    plt.title("SHAP: how each driver moves churn risk", loc="left", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / f"shap_02_beeswarm{suffix}.png"); plt.close()