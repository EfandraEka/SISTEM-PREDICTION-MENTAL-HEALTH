import streamlit as st
import pandas as pd
import joblib
import os

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Mental Health Predictor",
    layout="centered"
)

# =============================
# LOAD FILE
# =============================
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "standard_scaler.pkl"
FEATURE_PATH = "feature_columns.pkl"

for path in [MODEL_PATH, SCALER_PATH, FEATURE_PATH]:
    if not os.path.exists(path):
        st.error(f"File {path} tidak ditemukan")
        st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# =============================
# MAPPING
# =============================
stress_map = {
    "Jarang": 0,
    "Kadang-kadang": 1,
    "Sering": 2,
    "Sangat sering": 3
}

exercise_map = {
    "Tidak pernah": 0,
    "1-2 kali": 1,
    "3-5 kali": 2,
    "Setiap hari": 3
}

support_map = {
    "Tidak sama sekali": 0,
    "Sedikit": 1,
    "Cukup": 2,
    "Sangat kuat": 3
}

diet_map = {
    "Tidak sehat": 0,
    "Cukup sehat": 1,
    "Sehat": 2
}

work_map = {
    "Tidak": 0,
    "Kadang-kadang": 1,
    "Sering": 2,
    "Sangat sering": 3
}

# =============================
# UI
# =============================
st.title("Mental Health Prediction")

age = st.slider("Usia", 10, 80, 25)
sleep_hours = st.slider("Jam tidur", 0, 12, 7)
screen_time = st.slider("Waktu layar", 0, 16, 6)

stress = st.selectbox("Frekuensi stres", list(stress_map))
exercise = st.selectbox("Frekuensi olahraga", list(exercise_map))
support = st.selectbox("Dukungan sosial", list(support_map))
diet = st.selectbox("Pola makan", list(diet_map))
work = st.selectbox("Tekanan kerja/studi", list(work_map))

# =============================
# PREDICTION
# =============================
if st.button("Prediksi"):
    try:
        input_dict = {
            "age": age,
            "sleep_hours": sleep_hours,
            "screen_time": screen_time,
            "stress_level": stress_map[stress],
            "exercise_freq": exercise_map[exercise],
            "social_support": support_map[support],
            "diet_quality": diet_map[diet],
            "work_pressure": work_map[work],
        }

        df = pd.DataFrame([input_dict])

        # pastikan urutan fitur
        df = df[feature_columns]

        X_scaled = scaler.transform(df)
        pred = model.predict(X_scaled)[0]

        if pred == 1:
            st.error("Berisiko mengalami gangguan kesehatan mental")
        else:
            st.success("Tidak berisiko mengalami gangguan kesehatan mental")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
