# FreshBasket Loyalty - Churn Prediction & Retention Simulation

Classification-based churn prediction and diagnosis for the FreshBasket loyalty programme
(2,600 members, Jan-2023 to Jun-2024). Python only; everything is reproducible from one command.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
python src/run_pipeline.py          # ~45 s; writes outputs/ and data/processed/
python src/validate_forward.py      # validates the Jul-Sep 2024 forward scores (replay + drift)
python build_notebook.py            # optional: re-executes the EDA notebook
```
Tested in clean virtual environments on Python 3.12 with both the newest libraries (pandas 3.0, scikit-learn 1.8) and older pinned ones
(pandas 2.2.2, numpy 1.26.4, scipy 1.11.4, scikit-learn 1.4.2, LightGBM 4.3.0, SHAP 0.45.1, statsmodels 0.14.2, matplotlib 3.8.4): results are identical.
Run all commands from the project root.
Input: `data/raw/FreshBasket_Loyalty_Churn_Dataset.xlsx` (the three tables; row 1 of each sheet is a title banner, so `header=1`).

## Project layout
| Path | Purpose |
|---|---|
| `src/config.py` | Paths, cut-off dates, split date, scenario assumptions (all editable) |
| `src/data_prep.py` | Load, audit and clean the three tables (logs every fix) |
| `src/features.py` | Leakage-safe, cut-off-relative feature engineering (5 feature groups) |
| `src/modeling.py` | Rule baseline, logistic regression, random forest, LightGBM; forward-chaining CV; metrics; Platt calibration; paired bootstrap |
| `src/eda.py`, `src/explain.py`, `src/scenario.py` | EDA figures, SHAP drivers, retention scenarios |
| `src/run_pipeline.py` | Orchestrates everything |
| `notebooks/01_data_quality_and_eda.ipynb` | Executed DQ + EDA notebook |
| `data/processed/modelling_dataset.csv` | **Final customer-level modelling dataset** (features + label + split) |
| `outputs/churn_predictions_validation_cohort.csv` | **Churn prediction file** (held-out cohort): probability, predicted label, risk band, top-3 SHAP drivers |
| `outputs/forward_risk_scores_jul_sep_2024.csv` | Forward-looking scores for Jul-Sep 2024 (model refit on all labelled members). No outcome exists yet; validated indirectly by `src/validate_forward.py` (replay one period earlier + drift check) |
| `outputs/tables/` | All result tables (DQ audit, model comparison, ablation, SHAP, odds ratios, scenarios, `run_summary.json`) |
| `outputs/figures/` | All charts |
| `reports/FreshBasket_Churn_Final_Report.docx` (+ `report.md`) | **Final report** (rebuild: `pandoc report.md -o FreshBasket_Churn_Final_Report.docx --toc`, run inside `reports/`) |
| `reports/FreshBasket_Churn_Presentation.pptx` | **Presentation**, 11 slides for 10-15 minutes (rebuild: `node build_deck.js` inside `reports/`, needs `pptxgenjs`) |
| `outputs/tables/09_*.csv` | Forward-score validation: protocol replay, band hit rates, drift check, clipping sensitivity |

## Design decisions (assumptions)
1. **Population** - members with at least one purchase on/before Mar-2024 (2,368 of 2,600). The 232 others signed up in Apr-Jun 2024 or never bought; churn is undefined for them.
2. **Leakage control** - features use activity up to **Mar-2024** only. `LAST_PURCHASE_DATE`, `MONTHS_OBSERVED` and `INSUFFICIENT_HISTORY_FLAG` are computed using the label window and are never used.
3. **Missing months** - months after a member's last data row = zero activity; gaps in the middle = missing (NaN, counted).
4. **Spend repairs** - negative / zero-with-transactions / extreme spend rebuilt as basket value x transactions.
5. **Validation** - cohort-based: members who signed up before 2023-04-01 train (n=1,657); later signups are the held-out validation cohort (n=711). Hyper-parameters tuned by forward-chaining over signup quarters. A second, out-of-time backtest trains on a Dec-2023 snapshot and tests on the Mar-2024 snapshot. No random split is used for evaluation.
6. **Imbalance** - 34% churn overall (39% train, 22% validation); handled with class weighting + threshold chosen on out-of-fold F1, compared with an unweighted model. Probabilities are Platt-calibrated on out-of-fold scores.
7. **Reporting** - every metric is reported both for *all* validation members and for the **active base** (bought in Feb/Mar-2024). The all-member score is inflated by members who had already lapsed (97.8% of them churn).
8. **Scenarios** - assumption-driven simulations (cost, margin, uplift in `config.py`), not causal estimates.

## Known limitations
* One label window only; no repeated outcomes, so seasonality of churn cannot be learned. The Jul-Sep 2024 scores can only be checked once that quarter closes.
* Associations (e.g. support tickets) are not causal; the retention what-if numbers are model sensitivities, not expected campaign effects.
* The validation cohort has shorter tenure/history than the training cohort; the logistic baseline extrapolates badly on those features and is therefore trained without them (documented in the report).
* 5 members have `CHURNED=1` but no purchase in the data window (history before Jan-2023 is not provided).
