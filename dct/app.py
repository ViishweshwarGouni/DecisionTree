import streamlit as st
import numpy as np
import pickle

# =========================
# Load Model
# =========================

model = pickle.load(
    open("models/dt_classifier.pkl", "rb")
)

metrics = pickle.load(
    open("models/metrics.pkl", "rb")
)

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Decision Tree Classification",
    layout="centered"
)

# =========================
# Title
# =========================

st.title("🌸 Iris Flower Classification")

st.write(
    "Enter flower measurements below:"
)

# =========================
# Input Fields
# =========================

sepal_length = st.number_input(
    "Sepal Length",
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width",
    value=3.5
)

petal_length = st.number_input(
    "Petal Length",
    value=1.4
)

petal_width = st.number_input(
    "Petal Width",
    value=0.2
)

# =========================
# Prediction
# =========================

if st.button("Predict Flower"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted Flower: {flower_names[prediction[0]]}"
    )

# =========================
# Model Information
# =========================

st.subheader("📊 Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        f"""
        Criterion:
        {model.criterion}
        """
    )

    st.info(
        f"""
        Max Depth:
        {model.max_depth}
        """
    )

with col2:

    st.info(
        f"""
        Min Samples Split:
        {model.min_samples_split}
        """
    )

    st.info(
        f"""
        Min Samples Leaf:
        {model.min_samples_leaf}
        """
    )

# =========================
# Metrics
# =========================

st.subheader("📈 Model Metrics")

col3, col4 = st.columns(2)

with col3:

    st.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.2f}"
    )

    st.metric(
        "Precision",
        f"{metrics['Precision']:.2f}"
    )

with col4:

    st.metric(
        "Recall",
        f"{metrics['Recall']:.2f}"
    )

    st.metric(
        "F1 Score",
        f"{metrics['F1 Score']:.2f}"
    )

# =========================
# Footer
# =========================

st.caption(
    "Built using Streamlit & Scikit-Learn"
)