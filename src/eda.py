"""EDA helpers: produce the tables/figures quoted in the notebook and report."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from config import FIG, TAB

C_RET, C_CHURN, C_NEUT = "#2A6F97", "#D1495B", "#8D99AE"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "font.size": 9})


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def rate_table(df, by, target="y"):
    g = df.groupby(by, observed=True)[target].agg(["sum", "size"]).reset_index()
    g.columns = [by if isinstance(by, str) else "group", "churned", "members"]
    g["churn_rate"] = g.churned / g.members
    ci = [wilson(k, n) for k, n in zip(g.churned, g.members)]
    g["ci_low"], g["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    return g


def chi2_p(df, by, target="y"):
    ct = pd.crosstab(df[by], df[target])
    return stats.chi2_contingency(ct)[1]


def bar_rate(ax, tab, xcol, title, color=C_CHURN, rot=0):
    ax.bar(tab[xcol].astype(str), tab.churn_rate * 100, color=color, alpha=.9)
    ax.errorbar(range(len(tab)), tab.churn_rate * 100,
                yerr=[(tab.churn_rate - tab.ci_low) * 100, (tab.ci_high - tab.churn_rate) * 100],
                fmt="none", ecolor="#333", capsize=2, lw=1)
    for i, (r, h, n) in enumerate(zip(tab.churn_rate, tab.ci_high, tab.members)):
        ax.text(i, h * 100 + 0.8, f"{r*100:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=7)
    ax.set_title(title, fontsize=10, loc="left"); ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=rot); ax.set_ylim(0, max(tab.ci_high.max() * 100 * 1.35 + 4, 20))


def run_eda(E, a, audit):
    """E = eligible modelling frame with `y`; a = cleaned activity."""
    out = {}

    # 1. data-quality summary chart
    q = audit[audit.rows_affected > 0].copy()
    q = q[~q.issue.str.contains("zero-transaction|rows stop|LAST_PURCHASE")]
    q["label"] = q.table + ": " + q.issue.str.replace(r"\s*\(.*", "", regex=True).str.slice(0, 70)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.barh(q.label[::-1], q.rows_affected[::-1], color=C_NEUT)
    ax.set_xscale("log")
    for i, v in enumerate(q.rows_affected[::-1]):
        ax.text(v * 1.08, i, f"{v:,}", va="center", fontsize=8)
    ax.set_xlabel("Rows / values affected (log scale)")
    ax.set_title("Data-quality issues found and fixed", loc="left", fontsize=10)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(FIG / "eda_01_data_quality.png"); plt.close(fig)

    # 2. recency vs churn (the dominant signal)
    t = E.assign(rec=E.recency_months.clip(upper=6).astype(int).astype(str).replace("6", "6+"))
    rt = rate_table(t, "rec"); rt.to_csv(TAB / "02_churn_by_recency.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); bar_rate(ax, rt, "rec", "Churn rate by months since last purchase (as of Mar-2024)")
    ax.set_xlabel("Months since last purchase at cut-off"); ax.set_ylim(0, 115); ax.set_yticks(range(0, 101, 20)); fig.tight_layout(); fig.savefig(FIG / "eda_02_recency.png"); plt.close(fig)

    # 3. segments
    seg = {}
    E = E.copy()
    E["membership_tier"] = pd.Categorical(E.membership_tier, ["Silver", "Gold", "Platinum"], ordered=True)
    E["home_price_tier"] = pd.Categorical(E.home_price_tier, ["Value", "Mainstream", "Upscale", "Unknown"], ordered=True)
    E["signup_half"] = E.signup_date.dt.year.astype(str) + "-H" + ((E.signup_date.dt.month > 6) + 1).astype(str)
    E["age_band"] = pd.cut(E.age, [17, 29, 44, 59, 80], labels=["18-29", "30-44", "45-59", "60+"])
    E["ticket_bucket"] = pd.cut(E.tickets_l6, [-1, 0, 1, 99], labels=["0", "1", "2+"])
    E["email_quartile"] = pd.qcut(E.emails_l6.rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    E["app_quartile"] = pd.qcut(E.app_l6.rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    E["coupon_bucket"] = pd.cut(E.coupons_l6, [-1, 0, 2, 99], labels=["0", "1-2", "3+"])
    active = E[E.recency_months <= 1]
    rows = []
    for nm, col in [("Membership tier", "membership_tier"), ("City", "city"), ("Signup cohort (half-year)", "signup_half"),
                    ("Age band", "age_band"), ("Marketing opt-in", "marketing_opt_in"), ("Gender", "gender"),
                    ("Preferred category", "preferred_category"), ("Home store price tier", "home_price_tier"),
                    ("Support tickets (last 6m)", "ticket_bucket"), ("Email opens (last 6m, quartile)", "email_quartile"),
                    ("App sessions (last 6m, quartile)", "app_quartile"), ("Coupons redeemed (last 6m)", "coupon_bucket")]:
        for pop_name, pop in (("All eligible", E), ("Active base (bought in Feb/Mar-24)", active)):
            tb = rate_table(pop, col)
            tb.insert(0, "segment_variable", nm); tb.insert(1, "population", pop_name)
            tb = tb.rename(columns={col: "level"}); tb["chi2_p"] = chi2_p(pop, col)
            rows.append(tb)
    segtab = pd.concat(rows); segtab.to_csv(TAB / "02_churn_by_segment.csv", index=False)

    def seg_fig(items, fname, ncols=3, pop="All eligible", figsize=None):
        n = len(items); nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize or (4.2 * ncols, 3.3 * nrows)); axs = np.atleast_1d(axs).ravel()
        for ax, (nm, rot) in zip(axs, items):
            tb = segtab[(segtab.segment_variable == nm) & (segtab.population == pop)]
            bar_rate(ax, tb.rename(columns={"level": "lvl"}), "lvl", f"{nm}  (chi2 p={tb.chi2_p.iloc[0]:.3f})", rot=rot)
        for ax in axs[n:]: ax.axis("off")
        fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)

    seg_fig([("Membership tier", 0), ("Signup cohort (half-year)", 30), ("Support tickets (last 6m)", 0)], "eda_03_segments_all.png")
    seg_fig([("Membership tier", 0), ("Support tickets (last 6m)", 0), ("Email opens (last 6m, quartile)", 0),
             ("App sessions (last 6m, quartile)", 0), ("Coupons redeemed (last 6m)", 0), ("Marketing opt-in", 0)],
            "eda_04_active_base_drivers.png", pop="Active base (bought in Feb/Mar-24)")
    seg_fig([("City", 45)], "eda_05_city.png", ncols=1, figsize=(8, 3.4))

    # 4. activity trajectories: monthly mean transactions for churned vs retained (members present since Jan-23)
    full = E[(E.signup_date < "2023-01-01")].index
    t = a[a.CUSTOMER_ID.isin(full)].merge(E[["y"]], left_on="CUSTOMER_ID", right_index=True)
    t = t[t.MONTH <= "2024-03-01"]
    traj = t.groupby(["MONTH", "y"]).agg(txn=("TRANSACTIONS", "mean"), app=("APP_SESSIONS", "mean"),
                                         tickets=("SUPPORT_TICKETS", "mean")).reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, col, ttl in zip(axs, ["txn", "app", "tickets"], ["Transactions / member-month", "App sessions / member-month", "Support tickets / member-month"]):
        for yv, c, lb in ((0, C_RET, "Retained"), (1, C_CHURN, "Churned")):
            d = traj[traj.y == yv]; ax.plot(d.MONTH, d[col], color=c, label=lb, lw=1.8)
        ax.set_title(ttl, fontsize=9, loc="left"); ax.tick_params(axis="x", rotation=40)
    axs[0].legend(frameon=False)
    fig.suptitle("Pre-label-window trajectories (members who joined before 2023; rows only up to Mar-2024)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "eda_06_trajectories.png"); plt.close(fig)

    # 5. correlation of numeric features with churn + redundancy
    num = E.select_dtypes("number").drop(columns=["y", "CHURNED", "eligible", "churn_derived", "active_base", "avg_monthly_spend_l6"], errors="ignore")
    corr = num.corrwith(E.y, method="spearman").sort_values()
    corr.rename("spearman_with_churn").to_csv(TAB / "02_feature_correlation_with_churn.csv")
    fig, ax = plt.subplots(figsize=(6, 7.5))
    top = pd.concat([corr.head(12), corr.tail(12)])
    ax.barh(top.index, top.values, color=[C_RET if v < 0 else C_CHURN for v in top.values])
    ax.set_title("Spearman correlation with churn (top +/- 12 features)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "eda_07_feature_correlation.png"); plt.close(fig)

    # 6. distributions: spend before/after cleaning is described in audit; class balance
    summary = dict(eligible=len(E), churn_rate=E.y.mean(), active_base=len(active), active_base_churn=active.y.mean(),
                   lapsed2plus=int((E.recency_months >= 2).sum()), lapsed2plus_churn=E.y[E.recency_months >= 2].mean())
    pd.Series(summary).to_csv(TAB / "02_population_summary.csv", header=["value"])
    return segtab, summary