# app.py – Streamlit Web App for Predicting Early‑Stage Diabetes
# --------------------------------------------------------------
#  • Loads a LightGBM pipeline saved with joblib
#  • Collects user inputs (age, gender, 14 symptoms)
#  • Outputs prediction and probability
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ❗ IMPORTANT: Import the function/class that lives inside the pickled pipeline
#    (We don't actually *use* it here; we just need it to exist at import time.)
from data_processing import build_preprocessor  # noqa: F401  ← keep for un‑pickling

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Early‑Stage Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# --------------------------------------------------
# Helper – load model once & cache
# --------------------------------------------------
MODEL_PATH = "lightgbm_symptom_full.pkl"

@st.cache_resource(show_spinner="Loading ML model…")
def load_model(path: str):
    if not Path(path).exists():
        st.error(f"❌ Model file not found: `{path}`")
        st.stop()
    return joblib.load(path)

model = load_model(MODEL_PATH)
st.success("✅ Model loaded – ready for prediction!")

# --------------------------------------------------
# UI – collect inputs
# --------------------------------------------------
st.title("🩺 Early‑Stage Diabetes Prediction")

st.markdown(
    "Fill in the details below, then click **Predict** "
    "to estimate the likelihood of early‑stage diabetes."
)

age    = st.slider("Age (years)", 1, 120, 40)
gender = st.radio("Gender", ["Male", "Female"])

symptom_questions = {
    "Polyuria (excessive urination)": "Polyuria",
    "Polydipsia (excessive thirst)": "Polydipsia",
    "Sudden weight loss": "sudden weight loss",
    "Weakness": "weakness",
    "Polyphagia (excessive hunger)": "Polyphagia",
    "Genital thrush": "Genital thrush",
    "Visual blurring": "visual blurring",
    "Itching": "Itching",
    "Irritability": "Irritability",
    "Delayed healing": "delayed healing",
    "Partial paresis": "partial paresis",
    "Muscle stiffness": "muscle stiffness",
    "Alopecia": "Alopecia",
    "Obesity": "Obesity",
}

st.markdown("### Symptom checklist")
user_symptoms = {}
for question, col_name in symptom_questions.items():
    user_symptoms[col_name] = st.radio(question, ["No", "Yes"], key=col_name)

# --------------------------------------------------
# Build input DataFrame
# --------------------------------------------------
def build_input_df():
    record = {"Age": age, "Gender": gender}
    record.update(user_symptoms)
    return pd.DataFrame([record])

# --------------------------------------------------
# Predict
# --------------------------------------------------
if st.button("🔮 Predict"):
    X_input = build_input_df()

    try:
        pred_class  = model.predict(X_input)[0]          # 0 / 1
        pred_proba  = model.predict_proba(X_input)[0][1] # probability of being Positive
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    result_text = "Positive (1)" if pred_class == 1 else "Negative (0)"
    st.subheader(f"**Prediction:** {result_text}")
    st.write(f"Probability of being Positive: **{pred_proba:.3f}**")

    with st.expander("See model input"):
        st.dataframe(X_input)

st.caption("ℹ️ This tool is for educational purposes only. "
           "Always consult a qualified healthcare professional for medical advice.")
