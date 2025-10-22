# =============================================
# STREAMLIT APP — Mental Health Prediction
# =============================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# =============================================
# MUAT MODEL DAN SCALER (CEK OTOMATIS)
# =============================================

model_path = None
scaler_path = None

# Coba deteksi lokasi file model dan scaler
if os.path.exists("best_model.pkl") and os.path.exists("standard_scaler.pkl"):
    model_path = "best_model.pkl"
    scaler_path = "standard_scaler.pkl"
elif os.path.exists("saved_models/best_model.pkl") and os.path.exists("saved_models/standard_scaler.pkl"):
    model_path = "saved_models/best_model.pkl"
    scaler_path = "saved_models/standard_scaler.pkl"

# Jika file tidak ditemukan, hentikan aplikasi
if model_path is None or not os.path.exists(model_path):
    st.error("❌ File model tidak ditemukan.\n\nPastikan `best_model.pkl` dan `standard_scaler.pkl` berada di folder yang sama dengan `app.py` atau di dalam folder `saved_models/`.")
    st.stop()

# Muat model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

# =============================================
# HEADER
# =============================================
st.title("🧠 Mental Health Prediction App")
st.markdown("""
Selamat datang di aplikasi **Prediksi Kesehatan Mental**!  
Silakan isi beberapa pertanyaan berikut untuk memperkirakan apakah seseorang berpotensi mengalami **masalah kesehatan mental** berdasarkan data yang dimasukkan.
""")

# =============================================
# INPUT PERTANYAAN
# =============================================
st.subheader("📋 Jawab Pertanyaan Berikut:")

age = st.slider("Berapa usia Anda?", 10, 80, 25)
sleep_hours = st.slider("Berapa jam tidur rata-rata Anda per hari?", 0, 12, 7)

stress_level = st.selectbox(
    "Seberapa sering Anda merasa stres berat?",
    ["Jarang", "Kadang-kadang", "Sering", "Sangat sering"]
)

exercise_freq = st.selectbox(
    "Seberapa sering Anda berolahraga dalam seminggu?",
    ["Tidak pernah", "1-2 kali", "3-5 kali", "Setiap hari"]
)

social_support = st.selectbox(
    "Apakah Anda memiliki dukungan sosial (teman/keluarga)?",
    ["Tidak sama sekali", "Sedikit", "Cukup", "Sangat kuat"]
)

diet_quality = st.selectbox(
    "Bagaimana pola makan Anda?",
    ["Tidak sehat", "Cukup sehat", "Sehat"]
)

screen_time = st.slider(
    "Berapa jam rata-rata Anda menghabiskan waktu di depan layar per hari?",
    0, 16, 6
)

work_pressure = st.selectbox(
    "Apakah pekerjaan atau studi Anda membuat tekanan berlebih?",
    ["Tidak", "Kadang-kadang", "Sering", "Sangat sering"]
)

# =============================================
# KONVERSI INPUT KE DATAFRAME
# =============================================
user_data = pd.DataFrame({
    "age": [age],
    "sleep_hours": [sleep_hours],
    "stress_level": [stress_level],
    "exercise_freq": [exercise_freq],
    "social_support": [social_support],
    "diet_quality": [diet_quality],
    "screen_time": [screen_time],
    "work_pressure": [work_pressure]
})

# Encoding manual sesuai model training
encoding_maps = {
    "stress_level": {"Jarang": 0, "Kadang-kadang": 1, "Sering": 2, "Sangat sering": 3},
    "exercise_freq": {"Tidak pernah": 0, "1-2 kali": 1, "3-5 kali": 2, "Setiap hari": 3},
    "social_support": {"Tidak sama sekali": 0, "Sedikit": 1, "Cukup": 2, "Sangat kuat": 3},
    "diet_quality": {"Tidak sehat": 0, "Cukup sehat": 1, "Sehat": 2},
    "work_pressure": {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Sangat sering": 3}
}

for col, mapping in encoding_maps.items():
    user_data[col] = user_data[col].map(mapping)

# =============================================
# SESUAIKAN URUTAN KOLOM DENGAN SCALER
# =============================================
try:
    if hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
        for col in expected_features:
            if col not in user_data.columns:
                user_data[col] = 0
        user_data = user_data[expected_features]
    else:
        st.warning("Scaler tidak memiliki atribut 'feature_names_in_'. Pastikan urutan kolom sesuai saat training.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat menyelaraskan fitur dengan scaler: {e}")
    st.stop()

# =============================================
# SKALING & PREDIKSI
# =============================================
try:
    scaled_data = scaler.transform(user_data)
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1] if hasattr(model, "predict_proba") else None
except Exception as e:
    st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
    st.stop()

# =============================================
# TAMPILKAN HASIL
# =============================================
st.markdown("---")
st.subheader("Hasil Prediksi:")

if st.button("Prediksi Sekarang"):
    if prediction == 1:
        st.error("Hasil Prediksi: **Berisiko Mengalami Masalah Kesehatan Mental**")
    else:
        st.success("Hasil Prediksi: **Sehat / Tidak Berisiko**")

    if proba is not None:
        st.write(f"**Tingkat Keyakinan Model:** {proba*100:.2f}%")

    st.markdown("---")
    st.caption("Model ini menggunakan algoritma Machine Learning (Naive Bayes / SVM) yang dilatih dari dataset gabungan dua sumber data ZIP.")

# =============================================
# FOOTER
# =============================================
st.markdown(f"""
---
*Dibuat oleh Efandra Eka*  
Model: **{model.__class__.__name__}**  
Scaler: StandardScaler  
""")
