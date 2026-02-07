# =============================================
# STREAMLIT APP — Mental Health Prediction
# =============================================

import streamlit as st
import pandas as pd
import joblib
import os

# =============================================
# KONFIGURASI HALAMAN (HARUS PALING ATAS)
# =============================================
st.set_page_config(
    page_title="Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

# =============================================
# MUAT MODEL DAN SCALER (AUTO-DETECT)
# =============================================

model_path = None
scaler_path = None

if os.path.exists("best_model.pkl") and os.path.exists("standard_scaler.pkl"):
    model_path = "best_model.pkl"
    scaler_path = "standard_scaler.pkl"
elif os.path.exists("saved_models/best_model.pkl") and os.path.exists("saved_models/standard_scaler.pkl"):
    model_path = "saved_models/best_model.pkl"
    scaler_path = "saved_models/standard_scaler.pkl"

if model_path is None:
    st.error(
        "File model tidak ditemukan.\n\n"
        "Pastikan `best_model.pkl` dan `standard_scaler.pkl` berada di:\n"
        "- Folder yang sama dengan `app.py`, atau\n"
        "- Folder `saved_models/`"
    )
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# =============================================
# HEADER
# =============================================
st.title("Mental Health Prediction App")
st.markdown("""
Selamat datang di aplikasi Prediksi Kesehatan Mental.  
Isi pertanyaan berikut untuk memperkirakan apakah seseorang berpotensi mengalami  
masalah kesehatan mental berdasarkan data yang dimasukkan.
""")

# =============================================
# INPUT USER
# =============================================
st.subheader("Jawab Pertanyaan Berikut")

age = st.slider("Berapa usia Anda?", 10, 80, 25)
sleep_hours = st.slider("Berapa jam tidur rata-rata per hari?", 0, 12, 7)

stress_level = st.selectbox(
    "Seberapa sering Anda merasa stres berat?",
    ["Jarang", "Kadang-kadang", "Sering", "Sangat sering"]
)

exercise_freq = st.selectbox(
    "Seberapa sering Anda berolahraga dalam seminggu?",
    ["Tidak pernah", "1-2 kali", "3-5 kali", "Setiap hari"]
)

social_support = st.selectbox(
    "Apakah Anda memiliki dukungan sosial?",
    ["Tidak sama sekali", "Sedikit", "Cukup", "Sangat kuat"]
)

diet_quality = st.selectbox(
    "Bagaimana pola makan Anda?",
    ["Tidak sehat", "Cukup sehat", "Sehat"]
)

screen_time = st.slider(
    "Berapa jam waktu layar per hari?",
    0, 16, 6
)

work_pressure = st.selectbox(
    "Apakah pekerjaan atau studi memberi tekanan berlebih?",
    ["Tidak", "Kadang-kadang", "Sering", "Sangat sering"]
)

# =============================================
# VALIDASI RINGAN
# =============================================
if sleep_hours < 3:
    st.warning("Jam tidur sangat rendah dan dapat memengaruhi akurasi prediksi.")

# =============================================
# DATAFRAME INPUT
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

# =============================================
# ENCODING SESUAI TRAINING
# =============================================
encoding_maps = {
    "stress_level": {"Jarang": 0, "Kadang-kadang": 1, "Sering": 2, "Sangat sering": 3},
    "exercise_freq": {"Tidak pernah": 0, "1-2 kali": 1, "3-5 kali": 2, "Setiap hari": 3},
    "social_support": {"Tidak sama sekali": 0, "Sedikit": 1, "Cukup": 2, "Sangat kuat": 3},
    "diet_quality": {"Tidak sehat": 0, "Cukup sehat": 1, "Sehat": 2},
    "work_pressure": {"Tidak": 0, "Kadang-kadang": 1, "Sering": 2, "Sangat sering": 3}
}

for col, mapping in encoding_maps.items():
    user_data[col] = user_data[col].map(mapping).fillna(0)

# =============================================
# PENYESUAIAN URUTAN KOLOM
# =============================================
try:
    if hasattr(scaler, "feature_names_in_"):
        expected_features = list(scaler.feature_names_in_)
        for col in expected_features:
            if col not in user_data.columns:
                user_data[col] = 0
        user_data = user_data[expected_features]
    else:
        st.warning(
            "Scaler tidak memiliki atribut feature_names_in_. "
            "Pastikan urutan fitur sesuai saat training."
        )
except Exception as e:
    st.error(f"Gagal menyelaraskan fitur: {e}")
    st.stop()

# =============================================
# PREDIKSI
# =============================================
st.markdown("---")
st.subheader("Hasil Prediksi")

if st.button("Prediksi Sekarang"):
    try:
        with st.spinner("Sedang memproses data"):
            scaled_data = scaler.transform(user_data)
            prediction = model.predict(scaled_data)[0]
            proba = (
                model.predict_proba(scaled_data)[0][1]
                if hasattr(model, "predict_proba")
                else None
            )

        if prediction == 1:
            st.error("Hasil Prediksi: Berisiko mengalami masalah kesehatan mental")
        else:
            st.success("Hasil Prediksi: Sehat atau tidak berisiko")

        if proba is not None:
            st.write(f"Tingkat keyakinan model: {proba * 100:.2f}%")

            if proba > 0.8:
                st.caption("Keyakinan model sangat tinggi")
            elif proba > 0.6:
                st.caption("Keyakinan model sedang")
            else:
                st.caption("Keyakinan model rendah")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# =============================================
# FOOTER
# =============================================
st.markdown(
    f"""
---
Dibuat oleh Efandra Eka  
Model: {model.__class__.__name__}  
Scaler: StandardScaler
"""
)
