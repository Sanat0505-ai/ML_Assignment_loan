import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# Hide Streamlit default icons
st.markdown("""
<style>
header {visibility:hidden;}
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# Page background
st.markdown("""
<style>
.stApp {
    background-color: #f4f9f4;
}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1 style='text-align:center;color:#2e7d32;'>🏦 Smart Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-based system to estimate whether a loan will be approved</p>", unsafe_allow_html=True)

st.markdown("---")

# -------- FILE UPLOAD --------
uploaded_file = st.file_uploader("📂 Upload Loan Applicant CSV")

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    # -------- MODEL SELECT --------
    st.subheader("⚙ Select Prediction Model")
    model_name = st.selectbox("", ["logistic","tree","knn","bayes","forest","xgb"])

    # Clean column names
    data.columns = data.columns.str.strip()

    # Remove ID column if present
    data.drop(columns=["Loan_ID"], errors="ignore", inplace=True)

    # -------- HANDLE ALL MISSING VALUES --------

    # Convert special text to NaN
    data.replace(["", " ", "NA", "N/A", "null", "?"], np.nan, inplace=True)

    # Numeric columns → median
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace=True)

    # Categorical columns → mode
    cat_cols = data.select_dtypes(include="object").columns
    for col in cat_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # -------- ENCODING --------
    data_encoded = pd.get_dummies(data)

    # Load training columns
    train_cols = joblib.load("model/columns.pkl")
    train_cols = [c for c in train_cols if c != "Loan_Status"]

    # Align columns
    data_encoded = data_encoded.reindex(columns=train_cols, fill_value=0)

    # FINAL SAFETY: remove any remaining NaN
    data_encoded = data_encoded.fillna(0)

    # -------- LOAD MODEL --------
    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_name}.pkl")

    X = scaler.transform(data_encoded)
    preds = model.predict(X)

    # Labels
    results = ["✅ Loan Approved" if p == 1 else "❌ Loan Rejected" for p in preds]

    st.markdown("---")

    # -------- SUMMARY --------
    st.subheader("📊 Approval Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applications", len(preds))
    c2.metric("Approved", sum(preds))
    c3.metric("Rejected", len(preds) - sum(preds))

    st.markdown("---")

    # -------- FINAL REPORT --------
    report_df = pd.DataFrame({
        "Applicant No": range(1, len(preds)+1),
        "Result": results
    })

    st.subheader("🧾 Loan Decision Report")
    st.dataframe(report_df)
