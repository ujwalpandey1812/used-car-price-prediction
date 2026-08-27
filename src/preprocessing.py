"""
preprocessing.py
-----------------
Shared data-cleaning and feature-engineering logic for the
Used Car Price Prediction project.

IMPORTANT: This module is imported by BOTH train.py and app.py so that
the exact same preprocessing is used at training time and at prediction
time. This avoids train/serve skew.

Reference year used for computing car age. The dataset's listing dates
(ad_created_on) run from Feb-2019 to May-2021, so 2021 is used as the
fixed "current year" for this dataset snapshot.
"""

import numpy as np
import pandas as pd

REFERENCE_YEAR = 2021

# Columns dropped permanently and the reason why.
# These were identified by inspecting the actual dataset
# (see notebook / EDA section in README for the full analysis).
LEAKAGE_COLUMNS = [
    "broker_quote",        # corr with sale_price = 0.96  -> a price quote, leaks target
    "original_price",      # corr with sale_price = 0.99  -> too tightly coupled, leak-like, 44% missing
    "emi_starts_from",      # corr with sale_price = 0.9999999 -> mathematically derived from sale_price
    "booking_down_pymnt",  # corr with sale_price = 0.9999999 -> mathematically derived from sale_price
    "car_rating",           # "great/good/fair/overpriced" label computed by comparing sale price to market value
]

POST_LISTING_COLUMNS = [
    "times_viewed",   # accumulates only AFTER the ad is live, unknown at prediction time
    "is_hot",          # platform flag set after listing performance is observed
    "reserved",         # becomes true only after a buyer books the car
]

ID_LIKE_COLUMNS = [
    "car_name",          # redundant: same info as make + model + variant combined
    "variant",             # 943 unique values -> too sparse for a simple academic pipeline
    "model",                # 185 unique values -> dropped in favour of 'make' to keep the
                             # pipeline lightweight and deployable; make + body_type + age +
                             # kms capture most of the price signal without an unwieldy
                             # one-hot / target-encoding scheme.
    "rto",                   # 261 unique RTO codes, redundant with registered_state
    "registered_city",       # 243 unique values, redundant with registered_state / city
    "ad_created_on",         # listing timestamp, not a property of the car itself
]

TARGET = "sale_price"

NUMERIC_FEATURES = ["car_age", "kms_run", "total_owners"]

CATEGORICAL_FEATURES = [
    "fuel_type",
    "body_type",
    "transmission",
    "city",
    "registered_state",
    "make",
    "source",
    "car_availability",
]

BOOLEAN_FEATURES = [
    "assured_buy",
    "warranty_avail",
    "fitness_certificate",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


def load_raw_dataset(path: str) -> pd.DataFrame:
    """Load the raw CSV exactly as provided."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data-cleaning steps identified during EDA:
      - remove exact duplicate rows
      - remove invalid target rows (sale_price <= 1000, i.e. data entry errors)
      - drop leakage / post-listing / id-like columns
      - engineer car_age
      - standardise missing categorical values
    Returns a cleaned dataframe that still contains the target column.
    """
    df = df.copy()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 2. Remove clearly invalid target values (data entry errors: price ~ 0)
    #    Found during EDA: 4 rows with sale_price of 0 or 35 (impossible for a real car)
    df = df[df["sale_price"] > 1000]

    # 3. Feature engineering: car_age from year of manufacture
    df["car_age"] = REFERENCE_YEAR - df["yr_mfr"]
    df.loc[df["car_age"] < 0, "car_age"] = 0

    # 4. Drop leakage / post-listing / id-like columns (if present)
    drop_cols = [
        c
        for c in (LEAKAGE_COLUMNS + POST_LISTING_COLUMNS + ID_LIKE_COLUMNS + ["yr_mfr"])
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    # 5. Standardise categorical missing values (keep as explicit "missing" category
    #    rather than dropping rows, since several categorical columns have <10% missing)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str).str.lower().str.strip()

    # 6. Boolean features: fill any missing with the majority class (False)
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    # 7. Numeric sanity: kms_run must be non-negative
    df = df[df["kms_run"] >= 0]

    return df


def get_feature_target_split(df: pd.DataFrame):
    """Split a cleaned dataframe into X (features used by the model) and y (target)."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


def _bool_to_float(x):
    """Module-level (picklable) helper: cast boolean columns to float before imputing."""
    return x.astype(float)


def build_full_pipeline():
    """
    Build the scikit-learn ColumnTransformer used to preprocess raw feature
    columns into a model-ready numeric matrix. Used identically by every
    model (Linear Regression, Random Forest, XGBoost).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    from sklearn.preprocessing import FunctionTransformer

    boolean_transformer = Pipeline(steps=[
        ("to_float", FunctionTransformer(_bool_to_float)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("bool", boolean_transformer, BOOLEAN_FEATURES),
    ])

    return preprocessor


def get_category_options(df: pd.DataFrame):
    """Return the valid dropdown options for each categorical feature, derived
    from the actual cleaned training data. Used by the Streamlit app so the
    UI never lets a user pick a category the model has never seen well
    represented."""
    options = {}
    for col in CATEGORICAL_FEATURES:
        options[col] = sorted(df[col].dropna().unique().tolist())
    return options
