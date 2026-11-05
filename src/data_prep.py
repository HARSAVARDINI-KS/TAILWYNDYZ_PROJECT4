"""Load the three tables, audit data quality, and return cleaned frames.

Every cleaning decision is logged to a DQ table so that the EDA notebook and the
report can quote exact counts.
"""
import numpy as np
import pandas as pd

from config import RAW_XLSX, TAB

SHEETS = {
    "profile": "fb Customer Profile",
    "activity": "fb Monthly Activity",
    "label": "fb Churn Label",
}


def load_raw() -> dict:
    """Read the raw sheets. Row 0 of each sheet is a title banner, so header=1."""
    return {k: pd.read_excel(RAW_XLSX, sheet_name=s, header=1) for k, s in SHEETS.items()}


def clean(raw: dict, save_audit: bool = True):
    """Return (profile, activity, label, audit_df)."""
    log = []

    def add(table, issue, n, action):
        log.append(dict(table=table, issue=issue, rows_affected=int(n), action=action))

    p, a, l = (raw[k].copy() for k in ("profile", "activity", "label"))

    # ---- duplicates -----------------------------------------------------
    n = p.duplicated().sum(); add("profile", "exact duplicate rows", n, "dropped")
    p = p.drop_duplicates().reset_index(drop=True)
    n = l.duplicated().sum(); add("label", "exact duplicate rows", n, "dropped")
    l = l.drop_duplicates().reset_index(drop=True)
    n = a.duplicated().sum(); add("activity", "exact duplicate rows", n, "dropped")
    a = a.drop_duplicates()
    n = a.duplicated(["CUSTOMER_ID", "MONTH"]).sum()
    add("activity", "same customer-month with conflicting values", n, "kept first")
    a = a.drop_duplicates(["CUSTOMER_ID", "MONTH"]).reset_index(drop=True)

    # ---- text casing / whitespace ----------------------------------------
    raw_tier = p["MEMBERSHIP_TIER"].nunique(); raw_city = p["CITY"].nunique()
    for c in ("MEMBERSHIP_TIER", "CITY", "GENDER", "PREFERRED_CATEGORY", "HOME_STORE_PRICE_TIER"):
        p[c] = p[c].astype("string").str.strip()
    p["MEMBERSHIP_TIER"] = p["MEMBERSHIP_TIER"].str.title()
    p["CITY"] = p["CITY"].str.title()
    add("profile", f"inconsistent casing/whitespace in MEMBERSHIP_TIER ({raw_tier} variants -> {p.MEMBERSHIP_TIER.nunique()})",
        raw_tier - p.MEMBERSHIP_TIER.nunique(), "strip + title-case")
    add("profile", f"inconsistent casing/whitespace in CITY ({raw_city} variants -> {p.CITY.nunique()})",
        raw_city - p.CITY.nunique(), "strip + title-case")

    # ---- missing profile values ------------------------------------------
    for c in ("AGE", "GENDER", "HOME_STORE_PRICE_TIER"):
        add("profile", f"missing {c}", p[c].isna().sum(), "median/'Unknown' imputation + missing flag")
    p["AGE_MISSING"] = p["AGE"].isna().astype(int)
    p["AGE"] = p["AGE"].fillna(p["AGE"].median())
    p["GENDER"] = p["GENDER"].fillna("Unknown")
    p["HOME_STORE_PRICE_TIER"] = p["HOME_STORE_PRICE_TIER"].fillna("Unknown")

    # ---- monthly activity: spend problems --------------------------------
    neg = (a.TOTAL_SPEND < 0)
    zero_with_txn = (a.TRANSACTIONS > 0) & (a.TOTAL_SPEND == 0)
    add("activity", "negative TOTAL_SPEND", neg.sum(), "rebuilt as AVG_BASKET_VALUE x TRANSACTIONS")
    add("activity", "zero TOTAL_SPEND despite TRANSACTIONS>0", zero_with_txn.sum(),
        "rebuilt as AVG_BASKET_VALUE x TRANSACTIONS")
    rebuilt = a.AVG_BASKET_VALUE * a.TRANSACTIONS
    a.loc[neg | zero_with_txn, "TOTAL_SPEND"] = rebuilt[neg | zero_with_txn]

    # outliers: spend wildly inconsistent with basket x transactions
    implied = a.AVG_BASKET_VALUE * a.TRANSACTIONS
    ratio = a.TOTAL_SPEND / implied.replace(0, np.nan)
    outl = (ratio > 5)
    add("activity", "outlier TOTAL_SPEND (>5x basket x transactions; max "
        f"${a.TOTAL_SPEND.max():,.0f} vs 99.9th pct ${a.TOTAL_SPEND.quantile(.999):,.0f})",
        outl.sum(), "replaced by AVG_BASKET_VALUE x TRANSACTIONS")
    a.loc[outl, "TOTAL_SPEND"] = implied[outl]

    zero_txn = (a.TRANSACTIONS == 0)
    add("activity", "zero-transaction months (valid inactivity, kept)", zero_txn.sum(),
        "kept; they are real signal (basket/promo fields are 0 in these months)")
    a["SPEND_FIXED"] = 0
    a.loc[neg | zero_with_txn | outl, "SPEND_FIXED"] = 1

    # ---- missing months ---------------------------------------------------
    months = pd.date_range("2023-01-01", "2024-06-01", freq="MS")
    sm = p.set_index("CUSTOMER_ID").SIGNUP_DATE.dt.to_period("M").dt.to_timestamp()
    first = sm.clip(lower=months[0])
    last_row = a.groupby("CUSTOMER_ID").MONTH.max()
    cnt = a.groupby("CUSTOMER_ID").MONTH.nunique()
    expected_to_last = ((last_row.dt.year - first.reindex(last_row.index).dt.year) * 12
                        + last_row.dt.month - first.reindex(last_row.index).dt.month + 1)
    interior = (expected_to_last - cnt).clip(lower=0)
    add("activity", f"interior missing months (gaps between first and last row) across {int((interior>0).sum())} members",
        interior.sum(), "left as NaN (excluded from averages) + count feature")
    trailing = (last_row < months[-1]).sum()
    add("activity", "members whose rows stop before Jun-2024 (no later rows)", trailing,
        "months after last row treated as zero activity (not missing)")

    # ---- label table ------------------------------------------------------
    add("label", "LAST_PURCHASE_DATE / MONTHS_OBSERVED / INSUFFICIENT_HISTORY_FLAG",
        len(l), "NOT used as features - computed using the label window (leakage)")

    audit = pd.DataFrame(log)
    if save_audit:
        audit.to_csv(TAB / "01_data_quality_audit.csv", index=False)
    return p, a, l, audit


if __name__ == "__main__":
    p, a, l, audit = clean(load_raw())
    print(audit.to_string())
