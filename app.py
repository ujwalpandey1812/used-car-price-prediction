"""
app.py
------
Streamlit web application for the Used Car Price Prediction project.

Loads the exact same pipeline (preprocessing + tuned model) that was
saved by train.py, so predictions made here are produced with IDENTICAL
preprocessing logic as during training/testing.

Run locally with:
    streamlit run app.py
"""

import json
import time
import os

import joblib
import pandas as pd
import streamlit as st

# src.preprocessing must be importable so joblib can unpickle the
# TransformedTargetRegressor / ColumnTransformer correctly.
from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES

MODEL_PATH = os.path.join("models", "final_model.pkl")
METADATA_PATH = os.path.join("models", "metadata.json")
RESULTS_PATH = os.path.join("results", "model_results.csv")
KPI_PATH = os.path.join("results", "kpi_report.csv")

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def pretty(label: str) -> str:
    return label.replace("_", " ").title()


def main():
    model = load_model()
    meta = load_metadata()

    st.title("🚗 Used Car Price Predictor")
    st.write(
        "Estimate the fair resale price of a used car based on its make, age, "
        "condition and listing details. Built as an academic ML project "
        "(B.Sc. AI & ML, SYAIML, 2026–27)."
    )

    tab_predict, tab_about = st.tabs(["🔮 Predict Price", "ℹ️ About the Project"])

    # ------------------------------------------------------------------
    # PREDICTION TAB
    # ------------------------------------------------------------------
    with tab_predict:
        st.subheader("Enter Car Details")

        options = meta["category_options"]
        ranges = meta["numeric_ranges"]

        col1, col2 = st.columns(2)

        with col1:
            make = st.selectbox("Make (Brand)", options["make"])
            body_type = st.selectbox("Body Type", [o for o in options["body_type"] if o != "missing"])
            fuel_type = st.selectbox("Fuel Type", options["fuel_type"])
            transmission = st.selectbox(
                "Transmission", [o for o in options["transmission"] if o != "missing"]
            )
            city = st.selectbox("Listing City", options["city"])

        with col2:
            car_age = st.slider(
                "Car Age (years)",
                min_value=0,
                max_value=int(ranges["car_age"]["max"]),
                value=int(ranges["car_age"]["median"]),
            )
            kms_run = st.number_input(
                "Kilometers Driven",
                min_value=0,
                max_value=int(ranges["kms_run"]["max"]),
                value=int(ranges["kms_run"]["median"]),
                step=1000,
            )
            total_owners = st.slider(
                "Total Previous Owners",
                min_value=1,
                max_value=int(ranges["total_owners"]["max"]),
                value=1,
            )
            registered_state = st.selectbox("Registered State", options["registered_state"])
            car_availability = st.selectbox("Availability", options["car_availability"])

        st.markdown("**Additional Details**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            assured_buy = st.checkbox("Assured Buy", value=True)
        with col4:
            warranty_avail = st.checkbox("Warranty Available", value=False)
        with col5:
            fitness_certificate = st.checkbox("Fitness Certificate", value=True)
        with col6:
            source = st.selectbox("Sale Source", options["source"])

        st.markdown("---")

        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            input_dict = {
                "car_age": car_age,
                "kms_run": kms_run,
                "total_owners": total_owners,
                "fuel_type": fuel_type,
                "body_type": body_type,
                "transmission": transmission,
                "city": city,
                "registered_state": registered_state,
                "make": make,
                "source": source,
                "car_availability": car_availability,
                "assured_buy": assured_buy,
                "warranty_avail": warranty_avail,
                "fitness_certificate": fitness_certificate,
            }

            input_df = pd.DataFrame([input_dict])

            # basic input validation
            errors = []
            if kms_run < 0:
                errors.append("Kilometers driven cannot be negative.")
            if car_age < 0:
                errors.append("Car age cannot be negative.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                try:
                    start = time.perf_counter()
                    prediction = model.predict(input_df)[0]
                    latency = time.perf_counter() - start

                    st.success("Prediction generated successfully.")
                    st.markdown("### Estimated Used Car Price")
                    st.markdown(f"## ₹ {prediction:,.0f}")
                    st.caption(f"Prediction latency: {latency*1000:.1f} ms")

                    low = prediction * 0.90
                    high = prediction * 1.10
                    st.write(f"Likely fair-price range (±10%): ₹{low:,.0f} — ₹{high:,.0f}")
                except Exception as e:
                    st.error(f"Something went wrong while generating the prediction: {e}")

    # ------------------------------------------------------------------
    # ABOUT TAB
    # ------------------------------------------------------------------
    with tab_about:
        st.subheader("Project Overview")
        st.write(
            "This application predicts the resale price of a used car using a "
            f"**{meta['final_model_name']}** model, selected after comparing "
            "Linear Regression, Random Forest and XGBoost on a held-out test set."
        )

        st.markdown("#### Model Performance (Test Set)")
        if os.path.exists(RESULTS_PATH):
            results_df = pd.read_csv(RESULTS_PATH)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("#### Project KPIs")
        if os.path.exists(KPI_PATH):
            kpi_df = pd.read_csv(KPI_PATH)
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)

        st.markdown("#### Features Used by the Model")
        st.write(
            f"- **Numeric:** {', '.join(pretty(c) for c in meta['features']['numeric'])}\n"
            f"- **Categorical:** {', '.join(pretty(c) for c in meta['features']['categorical'])}\n"
            f"- **Boolean:** {', '.join(pretty(c) for c in meta['features']['boolean'])}"
        )

        st.markdown("#### Student Details")
        st.write(
            "**Name:** Ujwal Pandey  \n"
            "**Course:** B.Sc. Artificial Intelligence & Machine Learning  \n"
            "**Class:** SYAIML  \n"
            "**Academic Year:** 2026–27"
        )


if __name__ == "__main__":
    main()
