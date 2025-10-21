import streamlit as st
import joblib
import numpy as np
import os
import traceback

st.set_page_config(page_title="Mental Health Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Sistem Prediksi Kesehatan Mental (Debug Mode)")
st.markdown("Jika terjadi error, detailnya akan muncul di bawah.")

try:
    st.subheader(" Cek File di Folder Sekarang")
    st.write(os.listdir("."))  # menampilkan file yang ada di server

    # Muat model
    model_path = "best_model.pkl"
    scaler_path = "standard_scaler.pkl"

    if not os.path.exists(model_path):
        st.error(f" File tidak ditemukan: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f" File tidak ditemukan: {scaler_path}")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success(" Model dan Scaler berhasil dimuat.")

    st.header("Input Data")
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.number_input("Tingkat Stres", min_value=0.0, step=0.1)
        f2 = st.number_input("Tingkat Kecemasan", min_value=0.0, step=0.1)
        f3 = st.number_input("Kualitas Tidur", min_value=0.0, step=0.1)
    with c2:
        f4 = st.number_input("Tingkat Energi", min_value=0.0, step=0.1)
        f5 = st.number_input("Konsentrasi", min_value=0.0, step=0.1)

    if st.button(" Prediksi"):
        data = np.array([[f1, f2, f3, f4, f5]])
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]
        st.subheader(" Hasil Prediksi")
        if pred == 1:
            st.error(" Potensi gangguan mental terdeteksi.")
        else:
            st.success(" Kondisi mental sehat.")

except Exception as e:
    st.error(" Terjadi error saat menjalankan aplikasi:")
    st.code(traceback.format_exc())
