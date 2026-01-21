# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import os
import datetime as dt

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
# RANGE WAKTU (INI KUNCI)
# =====================================
MIN_DATE = dt.datetime(1990, 1, 1)
MAX_DATE = dt.datetime(2090, 12, 31)

# =====================================
# UTIL
# =====================================
def normalize_year(dtime, min_year=1990):
    if dtime.year < min_year:
        return dtime.replace(year=min_year)
    return dtime

# =====================================
# FEATURE ENGINEERING
# =====================================
def feature_engineering(raw_df):
    df = raw_df.copy()

    # Normalisasi waktu (backend safety)
    df["signup_time"] = df["signup_time"].apply(normalize_year)
    df["purchase_time"] = df["purchase_time"].apply(normalize_year)

    # Selisih waktu
    df["time_diff"] = (
        pd.to_datetime(df["purchase_time"]) -
        pd.to_datetime(df["signup_time"])
    ).dt.total_seconds()

    df["time_diff"] = df["time_diff"].clip(lower=0)

    # Frequency features (deploy = 1)
    df["device_freq"] = 1
    df["ip_freq"] = 1

    # Encoding kategori
    df["source_Direct"] = (df["source"] == "Direct").astype(int)
    df["source_SEO"] = (df["source"] == "SEO").astype(int)
    df["browser_IE"] = (df["browser"] == "IE").astype(int)

    # Value group
    df["value_group_medium"] = (
        (df["purchase_value"] >= 20) &
        (df["purchase_value"] < 100)
    ).astype(int)

    return df

# =====================================
# LOAD MODEL
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

    bundle = joblib.load(path)

    if "model" not in bundle or "features" not in bundle:
        st.error("Format model .pkl tidak valid")
        st.stop()

    return bundle

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
# INPUT DATA
# =====================================
st.subheader("🧾 Input Data Transaksi")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        purchase_value = st.number_input(
            "Purchase Value", min_value=1.0, value=35.0
        )
        device_id = st.text_input(
            "Device ID", value="QVPSPJUOCKZAR"
        )
        ip_address = st.text_input(
            "IP Address", value="192.168.1.1"
        )
        source = st.selectbox(
            "Traffic Source", ["Direct", "SEO", "Ads"]
        )

    with col2:
        browser = st.selectbox(
            "Browser", ["Chrome", "Firefox", "IE", "Safari"]
        )

        signup_time = st.datetime_input(
            "Signup Time",
            value=MIN_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE
        )

        purchase_time = st.datetime_input(
            "Purchase Time",
            value=MIN_DATE,
            min_value=MIN_DATE,
            max_value=MAX_DATE
        )

    submit = st.form_submit_button("🔍 Prediksi")

# =====================================
# PREDICTION
# =====================================
if submit:
    raw_input = pd.DataFrame([{
        "purchase_value": purchase_value,
        "device_id": device_id,
        "ip_address": ip_address,
        "source": source,
        "browser": browser,
        "signup_time": signup_time,
        "purchase_time": purchase_time
    }])

    X = feature_engineering(raw_input)

    # Paksa fitur sesuai model
    for col in features:
        if col not in X.columns:
            X[col] = 0

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
st.caption("Fraud Detection System | Random Forest & Naive Bayes")
