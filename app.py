# -*- coding: utf-8 -*-
# =====================================
# FRAUD DETECTION STREAMLIT APP
# =====================================

import streamlit as st
import pandas as pd
import joblib
import os

# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Fraud Detection System")
st.write(
    "Aplikasi deteksi fraud berbasis Machine Learning "
    "(Random Forest & Complement Naive Bayes)"
)

# =====================================
# FEATURE ENGINEERING (BACKEND)
# =====================================
def feature_engineering(raw_df):
    df = raw_df.copy()

    # Time difference (detik)
    df["time_diff"] = (
        pd.to_datetime(df["purchase_time"]) -
        pd.to_datetime(df["signup_time"])
    ).dt.total_seconds()

    # Frequency (dummy / placeholder, sesuai training)
    df["device_freq"] = 1
    df["ip_freq"] = 1

    # Encoding source
    df["source_Direct"] = (df["source"] == "Direct").astype(int)
    df["source_SEO"] = (df["source"] == "SEO").astype(int)

    # Encoding browser
    df["browser_IE"] = (df["browser"] == "IE").astype(int)

    # Value group
    df["value_group_medium"] = (
        (df["purchase_value"] >= 20) &
        (df["purchase_value"] < 100)
    ).astype(int)

    return df

# =====================================
# LOAD MODELS
# =====================================
@st.cache_resource
def load_models():
    models = {
        "RF Baseline CV 3": "models/rf_baseline_cv3.pkl",
        "RF Baseline CV 5": "models/rf_baseline_cv5.pkl",
        "RF Tuned CV 3": "models/rf_tuned_cv3.pkl",
        "RF Tuned CV 5": "models/rf_tuned_cv5.pkl",
        "CNB Baseline CV 3": "models/cnb_baseline_cv3.pkl",
        "CNB Baseline CV 5": "models/cnb_baseline_cv5.pkl",
        "CNB Tuned CV 3": "models/cnb_tuned_cv3.pkl",
        "CNB Tuned CV 5": "models/cnb_tuned_cv5.pkl",
    }

    loaded = {}
    for name, path in models.items():
        if not os.path.exists(path):
            st.error(f"Model tidak ditemukan: {path}")
            st.stop()

        bundle = joblib.load(path)

        for key in ["model", "features"]:
            if key not in bundle:
                st.error(f"Model {name} tidak valid (missing '{key}')")
                st.stop()

        loaded[name] = bundle

    return loaded

MODELS = load_models()

# =====================================
# PILIH MODEL
# =====================================
st.sidebar.header("⚙️ Konfigurasi Model")

model_name = st.sidebar.selectbox(
    "Pilih Model",
    list(MODELS.keys())
)

bundle = MODELS[model_name]
model = bundle["model"]
features = bundle["features"]
scaler = bundle.get("scaler", None)
selector = bundle.get("feature_selector", None)

st.sidebar.success(bundle.get("model_type", model_name))

# =====================================
# INPUT DATA MENTAH
# =====================================
st.subheader("🧾 Input Data Transaksi (Data Mentah)")

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
    # RAW INPUT
    raw_input = pd.DataFrame([{
        "purchase_value": purchase_value,
        "source": source,
        "browser": browser,
        "signup_time": signup_time,
        "purchase_time": purchase_time
    }])

    # FEATURE ENGINEERING
    X = feature_engineering(raw_input)

    # Samakan fitur dengan training
    try:
        X = X[features]
    except KeyError as e:
        st.error(f"Fitur tidak cocok dengan model: {e}")
        st.stop()

    # Scaling
    if scaler is not None:
        X = scaler.transform(X)

    # Feature Selection
    if selector is not None:
        X = selector.transform(X)

    # Prediksi
    y_pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
        st.metric("Probabilitas Fraud", f"{prob:.2%}")

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")

    if y_pred == 1:
        st.error("🚨 TRANSAKSI TERINDIKASI FRAUD")
    else:
        st.success("✅ TRANSAKSI NORMAL (LEGIT)")

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("Fraud Detection System | Random Forest & Complement Naive Bayes")
