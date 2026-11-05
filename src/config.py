"""Central configuration: paths, dates, and modelling constants."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "FreshBasket_Loyalty_Churn_Dataset.xlsx"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for d in (PROC, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --- Time design -----------------------------------------------------------
# Label window (official definition): Apr-Jun 2024. Features may use <= Mar 2024.
FEATURE_CUTOFF = pd.Timestamp("2024-03-01")      # last month usable for features
LABEL_MONTHS = 3                                  # churn horizon (months)
WINDOW_START = pd.Timestamp("2023-01-01")         # first month of activity data

# Out-of-time backtest snapshot: features <= Dec 2023, label = Jan-Mar 2024
BACKTEST_CUTOFF = pd.Timestamp("2023-12-01")

# Cohort-based validation: members who signed up before this date train the
# model; members who signed up on/after it form the held-out validation cohort.
COHORT_SPLIT = pd.Timestamp("2023-04-01")

# Forward-looking scoring (production use): features through Jun 2024
SCORING_CUTOFF = pd.Timestamp("2024-06-01")

# --- Retention scenario assumptions (configurable, NOT estimated from data) -
SCENARIO = dict(
    cost_per_contact=2.0,          # USD per outreach (email/app push + coupon cost)
    margin=0.25,                   # gross margin on retained spend
    horizon_months=12,             # value horizon of a retained member
    relative_churn_reduction=[0.10, 0.20, 0.30],   # uplift among contacted at-risk members
)

FEATURE_GROUPS = ["demographic", "behavioural", "engagement", "support"]
