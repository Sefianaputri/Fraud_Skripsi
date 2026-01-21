# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import os
from collections import defaultdict

# =====================================
# CONFIG
# =====================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Fraud Detection System")
st.write("Deteksi fraud menggunakan Random Forest dan Naive Bayes")

# =====================================
# STORAGE FREKUENSI (SIMULASI BACKEND)
# di production → database / redis
# =====================================
if "device_counter" not in st.session_state:
    st.session_state.device_counter = defaultdict(int)

if "ip_counter" not in st.session_state:
    st.session_state.ip_counter = defaultdict(int)

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

    # =====================
    # DEVICE FREQUENCY
    # =====================
    device_id = str(df.loc[0, "device_id"])
    st.session_state.device_counter[device_id] += 1
    df["device_freq"] = st.session_state.device_counter[device_id]

    # =====================
    # IP FREQUENCY
    # =====================
    ip_address = str(df.loc[0, "ip_address"])
    st.session_state.ip_counter[ip_address] += 1
    df["ip_freq"] = st.session_state.ip_counter[ip_address]

    # =====================
    # ENCODING
    # =====================
    df["source_Direct"] = (df["source"] == "Direct").astype(int)
    df["source_SEO"] = (df["source"] == "SEO").astype(int)
    df["browser_IE"] = (df["browser"] == "IE").astype(int)

    # =====================
    # VALUE GROUP
    # =====================
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

    path = model_map.get((model_type, cv))

    if path is None or not os.path.exists(path):
        st.error("Model tidak ditemukan. Pastikan file .pkl tersedia.")
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
st.subheader("🧾 Input Data Transaksi (Raw Data)")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        purchase_value = st.number_input(
            "Purchase Value", min_value=1.0, value=35.0
        )
        source = st.selectbox(
            "Traffic Source", ["Direct", "SEO", "Ads"]
        )
        browser = st.selectbox(
            "Browser", ["Chrome", "Firefox", "IE", "Safari"]
        )
        device_id = st.text_input(
            "Device ID", value="QVPSPJUOCKZAR"
        )

    with col2:
        ip_address = st.text_input(
            "IP Address", value="103.25.61.88"
        )
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
        "device_id": device_id,
        "ip_address": ip_address,
        "signup_time": signup_time,
        "purchase_time": purchase_time
    }])

    # Feature engineering
    X = feature_engineering(raw_input)

    # Samakan fitur dengan training
    try:
        X = X[features]
    except KeyError as e:
        st.error(f"Fitur tidak sesuai dengan model: {e}")
        st.stop()
