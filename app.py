# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import os
# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Fraud Detection System")
st.write("Deteksi fraud menggunakan Random Forest & Naive Bayes")

# =====================================
# FEATURE ENGINEERING
# =====================================
def feature_engineering(raw_df):
    df = raw_df.copy()

    df["time_diff"] = (
        pd.to_datetime(df["purchase_time"]) -
        pd.to_datetime(df["signup_time"])
    ).dt.total_seconds()

    df["device_freq"] = 1
    df["ip_freq"] = 1

    df["source_Direct"] = (df["source"] == "Direct").astype(int)
    df["source_SEO"] = (df["source"] == "SEO").astype(int)
    df["browser_IE"] = (df["browser"] == "IE").astype(int)

    df["value_group_medium"] = (
        (df["purchase_value"] >= 20) &
        (df["purchase_value"] < 100)
    ).astype(int)

    return df

# =====================================
# LOAD MODEL OTOMATIS
# =====================================
@st.cache_resource
def load_model(model_type, cv):
    model_map = {
        ("Random Forest", 3): "models/rf_tuned_cv3.pkl",
        ("Random Forest", 5): "models/rf_tuned_cv5.pkl",
        ("Naive Bayes", 3): "models/cnb_tuned_cv3.pkl",
        ("Naive Bayes", 5): "models/cnb_tuned_cv5.pkl",
    }

    path = model_map[(model_type, cv)]

    if not os.path.exists(path):
        st.error(f"Model tidak ditemukan: {path}")
        st.stop()

    return joblib.load(path)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙️ Konfigurasi Model")

model_type = st.sidebar.selectbox(
    "Pilih Algoritma",
    ["Random Forest", "Naive Bayes"]
)

cv = st.sidebar.selectbox(
    "Cross Validation",
    [3, 5]
)

bundle = load_model(model_type, cv)

model = bundle["model"]
features = bundle["features"]
scaler = bundle.get("scaler")
selector = bundle.get("feature_selector")

st.sidebar.success(f"{model_type} | CV {cv}")

# =====================================
# INPUT DATA MENTAH
# =====================================
st.subheader("🧾 Input Data Transaksi")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        purchase_value = st.number_input("Purchase Value", min_value=1.0, value=35.0)
        source = st.selectbox("Traffic Source", ["Direct", "SEO", "Ads"])
        browser = st.selectbox("Browser", ["Chrome", "Firefox", "IE", "Safari"])

    with col2:
        signup_time = st.datetime_input("Signup Time")
        purchase_time = st.datetime_input("Purchase Time")

    submit = st.form_submit_button("🔍 Prediksi")

# =====================================
# PREDIKSI
# =====================================
if submit:
    raw_input = pd.DataFrame([{
        "purchase_value": purchase_value,
        "source": source,
        "browser": browser,
        "signup_time": signup_time,
        "purchase_time": purchase_time
    }])

    X = feature_engineering(raw_input)
    X = X[features]

    if scaler is not None:
        X = scaler.transform(X)

    if selector is not None:
        X = selector.transform(X)

    y_pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
        st.metric("Probabilitas Fraud", f"{prob:.2%}")

    st.markdown("---")
    if y_pred == 1:
        st.error("🚨 TRANSAKSI TERINDIKASI FRAUD")
    else:
        st.success("✅ TRANSAKSI NORMAL")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("Fraud Detection System | RF & Naive Bayes")
