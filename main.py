"""Builds + executes notebooks/01_data_quality_and_eda.ipynb (run from project root)."""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)
nb.cells = [
md("# FreshBasket Loyalty - Data quality & EDA\nThis notebook audits the three source tables, builds the leakage-safe modelling dataset and explores the churn signal. All numbers come from the same `src/` modules used by `run_pipeline.py`."),
code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))
import pandas as pd, numpy as np
from IPython.display import Image, display
from config import *
from data_prep import load_raw, clean
from features import build_features
pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_colwidth', 90)
raw = load_raw()
for k, v in raw.items(): print(k, v.shape)"""),
md("## 1. Data-quality audit\nIssues are detected and fixed in `data_prep.clean`; every fix is counted."),
code("""p, a, l, audit = clean(raw, save_audit=False)
audit"""),
md("**Key judgement calls**\n* Rows only exist for months a member was *on file*; months after a member's last row are treated as **zero activity** (not missing) while gaps in the middle of a history stay `NaN` and are counted as a feature.\n* `TOTAL_SPEND` errors (negative, zero-with-transactions, 13 extreme outliers) are rebuilt from `AVG_BASKET_VALUE x TRANSACTIONS`.\n* `LAST_PURCHASE_DATE`, `MONTHS_OBSERVED`, `INSUFFICIENT_HISTORY_FLAG` in the label table are computed with the label window, so they are **excluded** (leakage)."),
code("""F = build_features(p, a, FEATURE_CUTOFF).join(l.set_index('CUSTOMER_ID').CHURNED)
print('members', len(p), '| eligible (bought on/before Mar-24):', int(F.eligible.sum()))
E = F[F.eligible == 1].copy(); E['y'] = E.CHURNED.astype(int)
print('churn rate (eligible): %.1f%%' % (100 * E.y.mean()))
print('official label vs label re-derived from activity rows - disagreements:', int((E.CHURNED != E.churn_derived).sum()))
E.head()"""),
md("## 2. Leakage-safe design\nFeatures use activity up to **Mar-2024** only; the label window is Apr-Jun 2024. Members who never purchased before April cannot 'churn' by definition and are excluded."),
code("""from eda import run_eda
segtab, summ = run_eda(E, a, audit)
summ"""),
md("### 2.1 Data-quality summary"),
code("display(Image(str(FIG / 'eda_01_data_quality.png')))"),
md("### 2.2 Recency dominates\nA member who has not bought for 3+ months is almost certainly churned; the real early-warning problem is the **active base** (bought in Feb/Mar-24)."),
code("""display(Image(str(FIG / 'eda_02_recency.png')))
pd.read_csv(TAB / '02_churn_by_recency.csv').round(3)"""),
md("### 2.3 Segments (tier, signup cohort, support tickets)\nAcross *all* eligible members, tickets look 'protective' only because lapsed members generate no tickets - the active-base view below removes that artefact."),
code("display(Image(str(FIG / 'eda_03_segments_all.png')))"),
code("display(Image(str(FIG / 'eda_04_active_base_drivers.png')))"),
code("display(Image(str(FIG / 'eda_05_city.png')))"),
md("### 2.4 Behavioural trajectories before the label window"),
code("display(Image(str(FIG / 'eda_06_trajectories.png')))"),
md("### 2.5 Feature correlation / redundancy"),
code("""display(Image(str(FIG / 'eda_07_feature_correlation.png')))
num = E.select_dtypes('number').drop(columns=['y','CHURNED','eligible','churn_derived'])
c = num.corr(method='spearman').abs().where(np.triu(np.ones((num.shape[1],)*2), 1).astype(bool)).stack()
c[c > 0.9].sort_values(ascending=False).head(12).rename('|spearman|')"""),
md("Highly collinear window features (e.g. `txn_l6`/`txn_l12`) are kept: tree models are robust to them and the logistic baseline is L2-regularised."),
]
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": "notebooks"}})
nbf.write(nb, "notebooks/01_data_quality_and_eda.ipynb")
print("notebook written")
