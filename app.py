# =============================================
#  Streamlit App - Prediksi Penyakit Mental
# =============================================
import streamlit as st
import joblib
import numpy as np

# --- Judul Utama ---
st.title(" Sistem Prediksi Penyakit Mental")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Naive Bayes & SVM)** untuk memprediksi kemungkinan gangguan kesehatan mental berdasarkan data input pengguna.
""")

# --- Muat Model & Scaler ---
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    st.success(" Model dan Scaler berhasil dimuat.")
except Exception as e:
    st.error(" Gagal memuat model atau scaler. Pastikan file .pkl tersedia di folder yang sama.")
    st.stop()

# --- Input Pengguna ---
st.header(" Input Data")

col1, col2 = st.columns(2)
with col1:
    feature1 = st.number_input("Fitur 1 (misal: tingkat stres)", min_value=0.0, step=0.1)
    feature2 = st.number_input("Fitur 2 (misal: tingkat kecemasan)", min_value=0.0, step=0.1)
    feature3 = st.number_input("Fitur 3 (misal: kualitas tidur)", min_value=0.0, step=0.1)
with col2:
    feature4 = st.number_input("Fitur 4 (misal: tingkat energi)", min_value=0.0, step=0.1)
    feature5 = st.number_input("Fitur 5 (misal: fokus/konsentrasi)", min_value=0.0, step=0.1)

# --- Tombol Prediksi ---
if st.button(" Prediksi"):
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error(" Hasil Prediksi: Potensi gangguan mental **terdeteksi**.")
    else:
        st.success(" Hasil Prediksi: Kondisi mental **sehat**.")

# --- Catatan ---
st.markdown("---")
st.caption("Model Machine Learning © 2025 | Dikembangkan untuk analisis mental health awareness.")
