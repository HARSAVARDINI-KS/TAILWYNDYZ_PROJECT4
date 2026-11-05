"""Leakage-safe, cutoff-relative feature engineering.

`build_features(profile, activity, cutoff)` only reads activity rows with
MONTH <= cutoff. Nothing from the label window is ever touched, and all window
features are defined relative to the cutoff (L3 = last three months up to and
including the cutoff month). Because of this the same function is used for
  * the official snapshot   (cutoff = Mar-2024, label = Apr-Jun 2024),
  * the out-of-time backtest (cutoff = Dec-2023, label = Jan-Mar 2024), and
  * forward scoring          (cutoff = Jun-2024, label = unknown).
"""
import numpy as np
import pandas as pd

from config import WINDOW_START, LABEL_MONTHS

NUM_VARS = ["TRANSACTIONS", "TOTAL_SPEND", "DISTINCT_CATEGORIES", "PROMO_TXN_PCT",
            "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS", "COMPLAINT_FLAG"]

GROUPS = {
    "demographic": ["age", "age_missing", "gender", "city", "membership_tier", "marketing_opt_in",
                    "preferred_category", "home_price_tier"],
    "lifecycle": ["tenure_months", "history_months", "short_history_flag", "missing_months"],
    "behavioural": ["recency_months", "txn_l3", "txn_l6", "txn_l12", "spend_l3", "spend_l6", "spend_l12",
                    "active_months_l3", "active_months_l6", "txn_ratio_l3_prev3", "spend_ratio_l3_prev3",
                    "txn_slope_l6", "spend_slope_l6", "basket_l3", "basket_change", "categories_l3",
                    "promo_pct_l3", "promo_pct_l6", "promo_change", "months_with_purchase"],
    "engagement": ["app_l3", "app_l6", "app_change", "emails_l3", "emails_l6", "email_change",
                   "coupons_l3", "coupons_l6", "coupon_change", "app_per_txn_l6"],
    "support": ["tickets_l3", "tickets_l6", "tickets_l12", "complaint_months_l6", "complaint_l3",
                "ticket_change", "months_since_ticket"],
}
CATEGORICAL = ["gender", "city", "membership_tier", "preferred_category", "home_price_tier"]


def _panel(profile: pd.DataFrame, activity: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Customer x month grid up to the cutoff with explicit handling of gaps.

    * months before the member joined         -> excluded
    * months after the member's last data row  -> zeros (inactive, not 'missing')
    * months missing in the middle of history  -> NaN (flagged, ignored by means)
    """
    a = activity[activity.MONTH <= cutoff]
    months = pd.date_range(WINDOW_START, cutoff, freq="MS")
    start = profile.set_index("CUSTOMER_ID").SIGNUP_DATE.dt.to_period("M").dt.to_timestamp().clip(lower=WINDOW_START)
    start = start[start <= cutoff]

    grid = pd.MultiIndex.from_product([start.index, months], names=["CUSTOMER_ID", "MONTH"]).to_frame(index=False)
    grid = grid.merge(start.rename("start"), left_on="CUSTOMER_ID", right_index=True)
    grid = grid[grid.MONTH >= grid.start]
    grid = grid.merge(a, on=["CUSTOMER_ID", "MONTH"], how="left")
    grid["observed"] = grid["TRANSACTIONS"].notna()

    last_row = a.groupby("CUSTOMER_ID").MONTH.max().rename("last_row")
    grid = grid.merge(last_row, left_on="CUSTOMER_ID", right_index=True, how="left")
    trailing = grid.last_row.isna() | (grid.MONTH > grid.last_row)
    fill = (~grid.observed) & trailing
    grid.loc[fill, NUM_VARS] = 0.0                     # inactivity, not missing
    grid["interior_missing"] = ((~grid.observed) & (~trailing)).astype(int)
    grid["k"] = (cutoff.year - grid.MONTH.dt.year) * 12 + (cutoff.month - grid.MONTH.dt.month)  # 0 = cutoff month
    return grid


def _slope(df: pd.DataFrame, col: str) -> pd.Series:
    """OLS slope per customer over rows in df (x = -k, so positive = increasing)."""
    d = df[["CUSTOMER_ID", "k", col]].dropna()
    x = -d.k.astype(float)
    g = d.assign(x=x, y=d[col].astype(float))
    g["xm"] = g.groupby("CUSTOMER_ID").x.transform("mean")
    g["ym"] = g.groupby("CUSTOMER_ID").y.transform("mean")
    g["num"] = (g.x - g.xm) * (g.y - g.ym)
    g["den"] = (g.x - g.xm) ** 2
    s = g.groupby("CUSTOMER_ID")[["num", "den"]].sum()
    return (s.num / s.den.replace(0, np.nan)).fillna(0.0)


def build_features(profile: pd.DataFrame, activity: pd.DataFrame, cutoff: pd.Timestamp,
                   horizon: int = LABEL_MONTHS) -> pd.DataFrame:
    """Return one row per eligible customer (>=1 purchase on/before cutoff)."""
    cutoff = pd.Timestamp(cutoff)
    g = _panel(profile, activity, cutoff)
    idx = pd.Index(sorted(g.CUSTOMER_ID.unique()), name="CUSTOMER_ID")
    F = pd.DataFrame(index=idx)

    def win(lo, hi):               # months with lo <= k <= hi
        return g[(g.k >= lo) & (g.k <= hi)]

    def agg(frame, col, how):
        return frame.groupby("CUSTOMER_ID")[col].agg(how).reindex(idx)

    L3, P3, L6, L12 = win(0, 2), win(3, 5), win(0, 5), win(0, 11)

    # ---- behavioural -----------------------------------------------------
    purch = g[g.TRANSACTIONS > 0]
    F["recency_months"] = purch.groupby("CUSTOMER_ID").k.min().reindex(idx)       # NaN => never purchased
    for nm, fr in (("l3", L3), ("l6", L6), ("l12", L12)):
        F[f"txn_{nm}"] = agg(fr, "TRANSACTIONS", "sum")
        F[f"spend_{nm}"] = agg(fr, "TOTAL_SPEND", "sum")
    F["active_months_l3"] = agg(L3.assign(a=(L3.TRANSACTIONS > 0).astype(int)), "a", "sum")
    F["active_months_l6"] = agg(L6.assign(a=(L6.TRANSACTIONS > 0).astype(int)), "a", "sum")
    t_p3, s_p3 = agg(P3, "TRANSACTIONS", "sum"), agg(P3, "TOTAL_SPEND", "sum")
    F["txn_ratio_l3_prev3"] = ((F.txn_l3 + 1) / (t_p3.fillna(F.txn_l3) + 1)).clip(upper=5)   # new members: neutral = 1; clipped
    F["spend_ratio_l3_prev3"] = ((F.spend_l3 + 1) / (s_p3.fillna(F.spend_l3) + 1)).clip(upper=5)
    F["txn_slope_l6"] = _slope(L6, "TRANSACTIONS").reindex(idx).fillna(0)
    F["spend_slope_l6"] = _slope(L6, "TOTAL_SPEND").reindex(idx).fillna(0)
    F["basket_l3"] = (F.spend_l3 / F.txn_l3.replace(0, np.nan))
    basket_l6 = (F.spend_l6 / F.txn_l6.replace(0, np.nan))
    F["basket_change"] = F.basket_l3 / basket_l6
    act3 = L3[L3.TRANSACTIONS > 0]
    F["categories_l3"] = agg(act3, "DISTINCT_CATEGORIES", "mean")
    def wpromo(fr):
        w = fr[fr.TRANSACTIONS > 0]
        num = (w.PROMO_TXN_PCT * w.TRANSACTIONS).groupby(w.CUSTOMER_ID).sum()
        den = w.groupby("CUSTOMER_ID").TRANSACTIONS.sum()
        return (num / den).reindex(idx)
    F["promo_pct_l3"], F["promo_pct_l6"] = wpromo(L3), wpromo(L6)
    F["promo_change"] = F.promo_pct_l3 - F.promo_pct_l6
    F["months_with_purchase"] = purch.groupby("CUSTOMER_ID").size().reindex(idx).fillna(0)

    # ---- engagement ------------------------------------------------------
    F["app_l3"], F["app_l6"] = agg(L3, "APP_SESSIONS", "mean"), agg(L6, "APP_SESSIONS", "mean")
    F["app_change"] = F.app_l3 - agg(P3, "APP_SESSIONS", "mean").fillna(F.app_l3)
    F["emails_l3"], F["emails_l6"] = agg(L3, "EMAILS_OPENED", "mean"), agg(L6, "EMAILS_OPENED", "mean")
    F["email_change"] = F.emails_l3 - agg(P3, "EMAILS_OPENED", "mean").fillna(F.emails_l3)
    F["coupons_l3"], F["coupons_l6"] = agg(L3, "COUPONS_REDEEMED", "sum"), agg(L6, "COUPONS_REDEEMED", "sum")
    F["coupon_change"] = F.coupons_l3 - agg(P3, "COUPONS_REDEEMED", "sum").fillna(F.coupons_l3)
    F["app_per_txn_l6"] = agg(L6, "APP_SESSIONS", "sum") / F.txn_l6.replace(0, np.nan)

    # ---- support ---------------------------------------------------------
    F["tickets_l3"], F["tickets_l6"], F["tickets_l12"] = (agg(fr, "SUPPORT_TICKETS", "sum") for fr in (L3, L6, L12))
    F["complaint_months_l6"] = agg(L6, "COMPLAINT_FLAG", "sum")
    F["complaint_l3"] = agg(L3, "COMPLAINT_FLAG", "sum")
    F["ticket_change"] = F.tickets_l3 - agg(P3, "SUPPORT_TICKETS", "sum").fillna(F.tickets_l3)
    tk = g[g.SUPPORT_TICKETS > 0].groupby("CUSTOMER_ID").k.min().reindex(idx)
    F["months_since_ticket"] = tk.fillna(24)

    # ---- lifecycle -------------------------------------------------------
    prof = profile.drop_duplicates("CUSTOMER_ID").set_index("CUSTOMER_ID").reindex(idx)
    F["tenure_months"] = ((cutoff + pd.offsets.MonthEnd(0) - prof.SIGNUP_DATE).dt.days / 30.44).round(1)
    F["history_months"] = g.groupby("CUSTOMER_ID").size().reindex(idx)
    F["short_history_flag"] = (F.history_months < 4).astype(int)
    F["missing_months"] = agg(g, "interior_missing", "sum")

    # ---- demographic -----------------------------------------------------
    F["age"], F["age_missing"] = prof.AGE, prof.AGE_MISSING
    F["gender"], F["city"], F["membership_tier"] = prof.GENDER, prof.CITY, prof.MEMBERSHIP_TIER
    F["marketing_opt_in"] = prof.MARKETING_OPT_IN
    F["preferred_category"], F["home_price_tier"] = prof.PREFERRED_CATEGORY, prof.HOME_STORE_PRICE_TIER
    F["signup_date"] = prof.SIGNUP_DATE
    F["avg_monthly_spend_l6"] = F.spend_l6 / F.history_months.clip(upper=6)   # for scenario economics only

    # ---- eligibility + derived label (for audit & backtest) --------------
    F["eligible"] = F.recency_months.notna().astype(int)
    fut = activity[(activity.MONTH > cutoff) & (activity.MONTH <= cutoff + pd.DateOffset(months=horizon))]
    bought_after = fut[fut.TRANSACTIONS > 0].CUSTOMER_ID.unique()
    F["churn_derived"] = (~F.index.isin(bought_after)).astype(int)
    return F


def feature_columns(groups):
    cols = []
    for gname in groups:
        cols += GROUPS[gname]
    return cols
