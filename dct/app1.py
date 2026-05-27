import streamlit as st
import numpy as np
import pickle

# =========================
# Load Model
# =========================

model = pickle.load(
    open("models/dt_regressor.pkl", "rb")
)

metrics = pickle.load(
    open("models/metrics.pkl", "rb")
)

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Decision Tree Regressor",
    layout="centered"
)

# =========================
# Title
# =========================

st.title("🏠 House Price Prediction")

st.write(
    "Enter house details below:"
)

# =========================
# Input Fields
# =========================

transaction_date = st.number_input(
    "Transaction Date",
    value=2013.0
)

house_age = st.number_input(
    "House Age",
    value=10.0
)

distance_to_mrt = st.number_input(
    "Distance to MRT Station",
    value=300.0
)

num_convenience_stores = st.number_input(
    "Number of Convenience Stores",
    value=5
)

latitude = st.number_input(
    "Latitude",
    value=24.97
)

longitude = st.number_input(
    "Longitude",
    value=121.54
)

# =========================
# Prediction
# =========================

if st.button("Predict Price"):

    input_data = np.array([[
        transaction_date,
        house_age,
        distance_to_mrt,
        num_convenience_stores,
        latitude,
        longitude
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price: {prediction[0]:.2f}"
    )

# =========================
# Model Information
# =========================

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"""
        Max Depth:
        {model.max_depth}
        """
    )

    st.info(
        f"""
        Min Samples Split:
        {model.min_samples_split}
        """
    )

with col2:

    st.info(
        f"""
        Min Samples Leaf:
        {model.min_samples_leaf}
        """
    )

    st.info(
        f"""
        Features Expected:
        {model.n_features_in_}
        """
    )

# =========================
# Metrics
# =========================

st.subheader("📈 Model Metrics")

col3, col4 = st.columns(2)

with col3:

    st.metric(
        "MAE",
        f"{metrics['MAE']:.2f}"
    )

    st.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}"
    )

with col4:

    st.metric(
        "MSE",
        f"{metrics['MSE']:.2f}"
    )

    st.metric(
        "R2 Score",
        f"{metrics['R2']:.2f}"
    )

# =========================
# Footer
# =========================

st.caption(
    "Built using Streamlit & Scikit-Learn"
)