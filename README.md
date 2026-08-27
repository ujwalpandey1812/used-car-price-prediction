# 🚗 Used Car Price Prediction

## Student
**Name:** Ujwal Pandey
**Course:** B.Sc. Artificial Intelligence & Machine Learning
**Class:** SYAIML
**Academic Year:** 2026–27

---

## 1. Problem Statement

Used-car pricing is often inconsistent — sellers overprice, buyers underestimate
value, and dealers rely on manual judgement. This project builds a machine
learning system that estimates the **fair market selling price of a used car**
from its listing attributes (brand, age, mileage, condition, etc.).

**Target users:**
- Used-car buyers, who want to know if a listed price is fair.
- Used-car sellers, who want a data-driven starting price.
- Used-car dealers, who need quick price estimates across many listings.

**Why Machine Learning:** Price depends on many interacting factors (age,
mileage, brand, condition, city) that don't follow a simple manual formula.
A regression model can learn these interactions directly from historical
sale data.

**Prediction target:** `sale_price` (INR), a continuous numeric value —
this is a **regression** problem.

**Measurable success criteria:** see the [KPI Framework](#7-kpi-framework)
below — in particular, MAE, R², and the percentage of predictions within
±10% of the actual price.

---

## 2. Dataset

- **File used:** `Used_Car_Price_Prediction.csv` (provided in the uploaded
  zip), saved here as `data/dataset.csv`.
- **Raw shape:** 7,400 rows × 29 columns.
- **Target variable:** `sale_price` — chosen because it is the only column
  representing the actual transaction price of the car; all other
  price-like columns (`broker_quote`, `original_price`, `emi_starts_from`,
  `booking_down_pymnt`) were found to be mathematically derived from it
  (see [Data Leakage](#data-leakage) below) and were therefore excluded
  as *features*, not used as the target.
- **Raw missing-value rate (all cells):** 2.20%.

---

## 3. Data Preprocessing

### Data Cleaning
- Removed 1 exact duplicate row.
- Removed 4 rows with an obviously invalid `sale_price` (≤ ₹35 — data
  entry errors, not real transactions).
- Final cleaned shape: **7,395 rows**.
- Missing categorical values were filled with an explicit `"missing"`
  category (not dropped) since missingness itself can carry information
  (e.g. a listing with no `transmission` specified).
- Missing boolean values were filled with the majority class.
- Numeric features (`car_age`, `kms_run`, `total_owners`) were
  median-imputed and standard-scaled inside the pipeline.

### Data Leakage

The following columns were **removed** after inspection because they leak
information about the target that would not be available *before* a sale
is finalized, or are computed directly from `sale_price`:

| Column | Correlation with `sale_price` | Reason removed |
|---|---|---|
| `emi_starts_from` | 0.9999999 | Mathematically derived (EMI formula) from sale price |
| `booking_down_pymnt` | 0.9999999 | Mathematically derived (down-payment %) from sale price |
| `original_price` | 0.986 | Too tightly coupled with sale price, 44% missing, leak-like |
| `broker_quote` | 0.963 | A price quote generated around the sale price |
| `car_rating` | — | A "great/good/fair/overpriced" label computed by comparing sale price to market value |
| `times_viewed`, `is_hot`, `reserved` | — | Only known **after** the ad has been live — unavailable at prediction time |

High-cardinality / redundant identifier columns were also dropped to keep
the pipeline simple and deployable: `car_name`, `variant` (943 unique
values), `model` (185 unique values), `rto` (261 unique values),
`registered_city` (243 unique values), `ad_created_on`.

### Final Feature Set (14 features)

| Type | Features |
|---|---|
| Numeric | `car_age`, `kms_run`, `total_owners` |
| Categorical | `fuel_type`, `body_type`, `transmission`, `city`, `registered_state`, `make`, `source`, `car_availability` |
| Boolean | `assured_buy`, `warranty_avail`, `fitness_certificate` |

## 4. Feature Engineering

- **`car_age`** = 2021 − `yr_mfr` (2021 chosen as the reference year since
  listing dates in the dataset run from Feb-2019 to May-2021). `yr_mfr`
  itself was then dropped in favor of the derived age.
- **Target transform:** `sale_price` is right-skewed (skew ≈ 2.77). All
  models are trained on `log1p(sale_price)` via
  `sklearn.compose.TransformedTargetRegressor`, and predictions are
  converted back to rupees with `expm1` before evaluation — this is
  handled transparently inside the saved pipeline, so the Streamlit app
  never has to think about it.
- Categorical text was lower-cased and stripped for consistency, then
  one-hot encoded (`handle_unknown="ignore"` so the app never crashes on
  an unseen category).

---

## 5. Train / Validation / Test Split

- **60% train / 20% validation / 20% test**, `random_state=42`.
- The preprocessing `ColumnTransformer` is fit **only** on the training
  data inside each model's pipeline; validation and test data are only
  ever `.transform()`-ed, never used to fit the imputers/scalers/encoders.
- The final held-out **test set was only touched once**, at the very end,
  to produce the reported comparison table.

---

## 6. Models

| Model | Role |
|---|---|
| Linear Regression | Baseline |
| Random Forest Regressor | Candidate 1 |
| XGBoost Regressor | Candidate 2 |
| Tuned XGBoost (RandomizedSearchCV) | Final candidate |

**Hyperparameter tuning:** `RandomizedSearchCV` (5-fold CV, 20 iterations,
scoring = R²) was run on the stronger of Random Forest / XGBoost after the
initial validation comparison (XGBoost had the higher validation R²).
Best parameters found:

```
subsample: 0.9
n_estimators: 600
max_depth: 5
learning_rate: 0.05
colsample_bytree: 0.8
```

---

## 7. Evaluation (Actual Test-Set Results)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **Tuned XGBoost (final)** | **₹63,107.92** | **₹94,228.05** | **0.8739** |
| XGBoost | ₹64,916.64 | ₹98,972.27 | 0.8609 |
| Random Forest | ₹68,342.69 | ₹103,243.21 | 0.8486 |
| Linear Regression (baseline) | ₹75,030.30 | ₹115,712.63 | 0.8098 |

**Final model selected: Tuned XGBoost.**

- It outperforms the Linear Regression baseline by a wide margin (R² 0.874
  vs 0.810) and beats the untuned Random Forest and XGBoost models,
  confirming that (a) the relationship between features and price is
  non-linear, and (b) tuning meaningfully improved XGBoost.
- **Strengths:** captures non-linear interactions (e.g. brand × age × body
  type), handles the mixed categorical/numeric feature set well, robust to
  outliers in `kms_run`.
- **Limitations:** the model is *not* perfect — a MAE of ~₹63k on cars
  averaging ~₹450k means real listings can still miss by a meaningful
  margin, particularly for rare makes/models with few training examples,
  or unusual combinations of features. It should be used as a **starting
  estimate**, not a final valuation.

## KPI Framework

| KPI | Category | Definition | Formula | Target | Actual | Interpretation |
|---|---|---|---|---:|---:|---|
| Price Estimation Accuracy | Business | % of test predictions within ±10% of actual price | count(\|pred−actual\|/actual ≤ 0.10)/n × 100 | ≥ 80% | **44.83%** | Below target — the model is directionally useful but not yet precise enough for ±10% pricing guarantees |
| MAE | ML | Average absolute prediction error | mean(\|pred − actual\|) | Minimize | **₹63,107.92** | Typical prediction error is ~₹63k |
| R² | ML | Variance in price explained by the model | 1 − SS_res/SS_tot | ≥ 0.85 | **0.8739** | Meets target |
| Missing Value Rate | Data Quality | Missing cells / total cells in raw data | — | < 5% | **2.20%** | Within target |
| Prediction Latency | Product/Engineering | Time for one `.predict()` call | wall-clock, avg of 50 runs | < 2 sec | **6.69 ms** | Meets target, well within real-time requirements |

*(All "Actual" values above are computed directly by `train.py` from the
uploaded dataset and are re-generated in `results/model_results.csv` and
`results/kpi_report.csv` every time the script is run.)*

---

## 8. Streamlit Application

`app.py` loads the saved pipeline (`models/final_model.pkl`) — the exact
same `ColumnTransformer` + `TransformedTargetRegressor(XGBRegressor)` used
during training — so there is **zero train/serve skew**.

- **Predict tab:** collects the 14 model features via dropdowns, sliders
  and number inputs (options are pulled from the actual training data via
  `models/metadata.json`, so the UI can never offer an option the model
  didn't see), validates input, and displays the predicted price along
  with a ±10% "fair price range" and the measured prediction latency.
- **About tab:** shows the actual model comparison table, the KPI report,
  the feature list, and student details — handy for a viva demo.

---

## 9. Project Structure

```text
used-car-price-prediction/
│
├── app.py                     # Streamlit application
├── train.py                   # Full training pipeline (run this first)
├── requirements.txt
├── README.md
│
├── data/
│   └── dataset.csv            # the uploaded dataset
│
├── models/
│   ├── final_model.pkl        # saved pipeline (preprocessing + tuned XGBoost)
│   └── metadata.json          # feature list, dropdown options, saved test metrics
│
├── src/
│   ├── preprocessing.py       # shared cleaning / feature engineering / pipeline builder
│   └── evaluation.py          # metric + KPI helper functions
│
└── results/
    ├── model_results.csv      # actual MAE/RMSE/R2 comparison table
    └── kpi_report.csv         # actual 5-KPI report
```

---

## 10. How to Run Locally

```bash
pip install -r requirements.txt
python train.py        # re-trains everything and regenerates models/ and results/
streamlit run app.py   # launches the web app at http://localhost:8501
```

(`models/final_model.pkl` is already included, so you can skip
`python train.py` and go straight to `streamlit run app.py` if you just
want to demo the app.)

---

## 11. Deployment on Streamlit Community Cloud

1. Push this entire `used-car-price-prediction/` folder to a public GitHub
   repository (keep the folder structure as-is).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select the repository, branch, and set:
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch the app.
5. No secrets, API keys, or external services are required — the model
   file is loaded directly from the repository.

---

## 12. Final Requirement Checklist

- [x] Dataset loaded and inspected directly (no assumptions)
- [x] Target variable identified from actual data (`sale_price`)
- [x] Missing values, duplicates, data types, outliers all checked
- [x] Data leakage explicitly identified and removed
- [x] Feature engineering (`car_age`, log-target transform)
- [x] Train / validation / test split with fixed seed, no leakage into val/test
- [x] Baseline (Linear Regression), Random Forest, XGBoost all trained
- [x] Hyperparameter tuning via RandomizedSearchCV with cross-validation
- [x] MAE, RMSE, R² computed and compared across all models on the test set
- [x] Final model selected based on actual test performance (Tuned XGBoost)
- [x] 5 KPIs defined, measured, and interpreted (Business, 2×ML, Data Quality, Product)
- [x] Single shared preprocessing pipeline used by both training and the app
- [x] Streamlit app with real input fields matching the actual model features
- [x] Input validation and error handling in the app
- [x] `requirements.txt`, `README.md`, saved model, and dataset all included
- [x] No fabricated numbers anywhere — every metric above is generated by `train.py`
