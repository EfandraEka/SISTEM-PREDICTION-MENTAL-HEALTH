# =============================================
# STREAMLIT APP — Mental Health Prediction
# =============================================

import streamlit as st
import pandas as pd
import joblib
import os

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="Mental Health",
    layout="centered"
)

# =============================================
# LOAD PIPELINE
# =============================================
PIPELINE_PATH = "mental_health_pipeline.pkl"

if not os.path.exists(PIPELINE_PATH):
    st.error(
        "File model tidak ditemukan.\n\n"
        "Pastikan `mental_health_pipeline.pkl` berada "
        "di folder yang sama dengan app.py."
    )
    st.stop()

pipeline = joblib.load(PIPELINE_PATH)

# =============================================
# HEADER
# =============================================
st.title("Mental Health Prediction App")
st.markdown(
    """
Aplikasi ini memprediksi potensi masalah kesehatan mental
berdasarkan data yang dimasukkan pengguna.
"""
)

# =============================================
# INPUT USER
# =============================================
st.subheader("Input Data")

age = st.slider("Usia", 10, 80, 25)
sleep_hours = st.slider("Jam tidur per hari", 0, 12, 7)
screen_time = st.slider("Waktu layar per hari (jam)", 0, 16, 6)

stress_level = st.selectbox(
    "Frekuensi stres",
    ["Jarang", "Kadang-kadang", "Sering", "Sangat sering"]
)

exercise_freq = st.selectbox(
    "Frekuensi olahraga per minggu",
    ["Tidak pernah", "1-2 kali", "3-5 kali", "Setiap hari"]
)

social_support = st.selectbox(
    "Dukungan sosial",
    ["Tidak sama sekali", "Sedikit", "Cukup", "Sangat kuat"]
)

diet_quality = st.selectbox(
    "Kualitas pola makan",
    ["Tidak sehat", "Cukup sehat", "Sehat"]
)

work_pressure = st.selectbox(
    "Tekanan pekerjaan atau studi",
    ["Tidak", "Kadang-kadang", "Sering", "Sangat sering"]
)

# =============================================
# DATAFRAME INPUT (MENTAH, TANPA ENCODING)
# =============================================
user_data = pd.DataFrame(
    {
        "age": [age],
        "sleep_hours": [sleep_hours],
        "screen_time": [screen_time],
        "stress_level": [stress_level],
        "exercise_freq": [exercise_freq],
        "social_support": [social_support],
        "diet_quality": [diet_quality],
        "work_pressure": [work_pressure],
    }
)

# =============================================
# PREDIKSI
# =============================================
st.markdown("---")
st.subheader("Hasil Prediksi")

if st.button("Prediksi"):
    try:
        with st.spinner("Memproses data"):
            prediction = pipeline.predict(user_data)[0]

            if hasattr(pipeline.named_steps["model"], "predict_proba"):
                proba = pipeline.predict_proba(user_data)[0][1]
            else:
                proba = None

        if prediction == 1:
            st.error("Berisiko mengalami masalah kesehatan mental")
        else:
            st.success("Tidak berisiko mengalami masalah kesehatan mental")

        if proba is not None:
            st.write(f"Tingkat keyakinan model: {proba * 100:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# =============================================
# FOOTER
# =============================================
st.markdown(
    """
---
Dibuat oleh Efandra Eka  
Model menggunakan pipeline preprocessing dan machine learning
"""
)
