# app.py – Early‑Stage Diabetes Risk Predictor (all‑in‑one)
# ---------------------------------------------------------
# • Embeds build_preprocessor and registers a fake `data_processing` module
#   so joblib can un‑pickle the saved LightGBM pipeline.
# • Presents a Streamlit UI for inference.
# ---------------------------------------------------------

# ----------------- std‑lib + third‑party imports -----------------
import sys, types
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm.sklearn import LGBMClassifier  # noqa: F401  (needed for un‑pickle)
from sklearn.preprocessing import LabelEncoder

# ----------------- 1️⃣  clone of build_preprocessor -----------------
TARGET_COLUMN = "class"

def build_preprocessor(df: pd.DataFrame):
    """Return a ColumnTransformer that one‑hot‑encodes categorical cols."""
    cat_cols, num_cols = [], []
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if df[col].dtype == "object":
            cat_cols.append(col)
        else:
            num_cols.append(col)

    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer(
        [("categorical", cat_pipe, cat_cols),
         ("numeric", "passthrough", num_cols)]
    )

# ----------------- 2️⃣  register fake module for unpickling ----------
fake_mod = types.ModuleType("data_processing")
fake_mod.build_preprocessor = build_preprocessor
sys.modules["data_processing"] = fake_mod     # 🎯 un‑picker will find it here

# ----------------- 3️⃣  Streamlit page config -----------------------
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")

# ----------------- 4️⃣  load model (cached) -------------------------
MODEL_PATH = "lightgbm_symptom_full.pkl"

@st.cache_resource(show_spinner="Loading ML model…")
def load_model(path: str):
    if not Path(path).exists():
        st.error(f"Model file not found: {path}")
        st.stop()
    return joblib.load(path)

model = load_model(MODEL_PATH)
st.success("✅ Model ready for prediction!")

# ----------------- 5️⃣  UI – collect user inputs --------------------
st.title("🩺 Early‑Stage Diabetes Prediction")

age    = st.slider("Age (years)", 1, 120, 40)
gender = st.radio("Gender", ["Male", "Female"])

symptom_questions = {
    "Polyuria (excessive urination)": "Polyuria",
    "Polydipsia (excessive thirst)": "Polydipsia",
    "Sudden weight loss": "sudden weight loss",
    "Weakness": "weakness",
    "Polyphagia (excessive hunger)": "Polyphagia",
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
user_symptoms = {
    col: st.radio(q, ["No", "Yes"], key=col)
    for q, col in symptom_questions.items()
}

# ----------------- 6️⃣  assemble input & predict --------------------
def make_input():
    record = {"Age": age, "Gender": gender}
    record.update(user_symptoms)
    return pd.DataFrame([record])

if st.button("🔮 Predict"):
    X = make_input()
    try:
        pred       = model.predict(X)[0]          # 0 / 1
        pred_prob  = model.predict_proba(X)[0][1] # probability of Positive
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    result = "Positive (1)" if pred == 1 else "Negative (0)"
    st.subheader(f"**Prediction:** {result}")
    st.write(f"Probability of being Positive: **{pred_prob:.3f}**")

    with st.expander("See model input"):
        st.dataframe(X)

st.caption(
    "ℹ️ This app is for educational purposes only. "
    "Consult a qualified healthcare professional for medical advice."
)
