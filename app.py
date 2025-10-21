# =============================================
# 🧠 Streamlit App - Sistem Prediksi Kesehatan Mental
# =============================================
import streamlit as st
import joblib
import numpy as np

# --- Judul Aplikasi ---
st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")
st.title("Sistem Prediksi Kesehatan Mental")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (SVM atau Naive Bayes)** untuk memprediksi kemungkinan seseorang mengalami gangguan kesehatan mental.
""")

# --- Muat Model dan Scaler ---
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    st.success("Model dan Scaler berhasil dimuat.")
except Exception as e:
    st.error("Gagal memuat model atau scaler. Pastikan file `best_model.pkl` dan `standard_scaler.pkl` ada di folder yang sama.")
    st.stop()

# --- Input Pengguna ---
st.header("Input Data")
st.markdown("Masukkan nilai-nilai berdasarkan parameter yang digunakan oleh model:")

col1, col2 = st.columns(2)
with col1:
    feature1 = st.number_input("Tingkat Stres", min_value=0.0, step=0.1)
    feature2 = st.number_input("Tingkat Kecemasan", min_value=0.0, step=0.1)
    feature3 = st.number_input("Kualitas Tidur", min_value=0.0, step=0.1)
with col2:
    feature4 = st.number_input("Tingkat Energi", min_value=0.0, step=0.1)
    feature5 = st.number_input("Fokus & Konsentrasi", min_value=0.0, step=0.1)

# --- Tombol Prediksi ---
if st.button(" Prediksi Sekarang"):
    try:
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5]])
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.subheader(" Hasil Prediksi")
        if prediction == 1:
            st.error(" **Potensi gangguan mental terdeteksi.**")
        else:
            st.success(" **Kondisi mental dalam keadaan sehat.**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

# --- Footer ---
st.markdown("---")
st.caption("Model Machine Learning © 2025 | Dibuat untuk keperluan penelitian dan edukasi.")
