import streamlit as st
import numpy as np
import os

# --- Coba import joblib dengan penanganan error ---
try:
    import joblib
except ModuleNotFoundError:
    st.error(" Modul 'joblib' belum terinstal. Jalankan perintah berikut di terminal:")
    st.code("pip install joblib")
    st.stop()

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Sistem Prediksi Kesehatan Mental")

# --- Cek File Model ---
if not os.path.exists("best_model.pkl") or not os.path.exists("standard_scaler.pkl"):
    st.error("File model tidak ditemukan. Pastikan `best_model.pkl` dan `standard_scaler.pkl` ada di root folder.")
    st.stop()

# --- Load Model ---
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    st.success(" Model berhasil dimuat.")
except Exception as e:
    st.error(f" Gagal memuat model: {e}")
    st.stop()

# --- Input Data ---
st.subheader("Masukkan Data Pengguna")
stress = st.slider("Tingkat Stres", 0.0, 10.0, 5.0)
anxiety = st.slider("Tingkat Kecemasan", 0.0, 10.0, 5.0)
sleep = st.slider("Kualitas Tidur", 0.0, 10.0, 5.0)
energy = st.slider("Tingkat Energi", 0.0, 10.0, 5.0)
focus = st.slider("Konsentrasi", 0.0, 10.0, 5.0)

if st.button("Prediksi"):
    data = np.array([[stress, anxiety, sleep, energy, focus]])
    try:
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]

        st.subheader("🩺 Hasil Prediksi")
        if pred == 1:
            st.error(" Potensi gangguan mental terdeteksi.")
        else:
            st.success(" Kondisi mental sehat.")
    except Exception as e:
        st.error(f" Error saat melakukan prediksi: {e}")
