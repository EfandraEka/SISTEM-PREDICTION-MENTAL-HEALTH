import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# =========================================================
# ⚙️ Load Model & Encoder
# =========================================================
@st.cache_resource
def load_model():
    try:
        # Pastikan nama file sesuai
        model = pickle.load(open("best_model.pkl", "rb"))
        encoder = pickle.load(open("label_encoder.pkl", "rb"))  # ubah sesuai isi sebenarnya
        return model, encoder
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan. Pastikan `best_model.pkl` dan `label_encoder.pkl` ada di folder utama.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau encoder: {e}")
        st.stop()

model, le = load_model()

# =========================================================
# 🧠 Judul & Deskripsi
# =========================================================
st.set_page_config(page_title="Mental Health Detector", layout="centered")
st.title("🧠 Mental Health Depression Detector")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning (Naive Bayes)* untuk memprediksi kemungkinan seseorang mengalami **depresi**  
berdasarkan data kesehatan mental.  

_Model ini dilatih menggunakan dataset nasional kesehatan mental._
""")

st.divider()

# =========================================================
# Input Data Responden
# =========================================================
st.subheader("Masukkan Data Responden")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    occupation = st.selectbox("Pekerjaan", ["Pelajar", "Mahasiswa", "Karyawan", "Pengangguran", "Lainnya"])

# =========================================================
# Penilaian Gejala Depresi (PHQ-9)
# =========================================================
st.markdown("### Kuesioner PHQ-9 — 2 Minggu Terakhir")
st.write("Pilih seberapa sering Anda mengalami hal-hal berikut:")

phq_questions = [
    "1️⃣ Merasa sedih, murung, atau putus asa.",
    "2️⃣ Kehilangan minat atau kesenangan dalam melakukan kegiatan sehari-hari.",
    "3️⃣ Kesulitan tidur, atau tidur terlalu banyak.",
    "4️⃣ Merasa lelah atau kurang energi hampir setiap hari.",
    "5️⃣ Nafsu makan menurun atau makan berlebihan.",
    "6️⃣ Merasa buruk tentang diri sendiri, merasa gagal, atau mengecewakan orang lain.",
    "7️⃣ Kesulitan berkonsentrasi pada hal-hal seperti membaca atau menonton TV.",
    "8️⃣ Bergerak atau berbicara sangat lambat, atau terlalu gelisah dan tidak bisa diam.",
    "9️⃣ Memiliki pikiran bahwa Anda lebih baik mati atau menyakiti diri sendiri."
]

options = {"Tidak Pernah": 0, "Jarang": 1, "Sering": 2, "Sangat Sering": 3}

phq_score = sum(st.radio(q, list(options.keys()), horizontal=True, index=1, key=q).replace(q, "") or options[st.session_state[q]] for q in phq_questions)

# =========================================================
# Interpretasi Skor PHQ-9
# =========================================================
if phq_score <= 4:
    level, color = "Tidak ada / Minimal", "🟢"
elif phq_score <= 9:
    level, color = "Depresi ringan", "🟡"
elif phq_score <= 14:
    level, color = "Depresi sedang", "🟠"
elif phq_score <= 19:
    level, color = "Depresi cukup berat", "🔴"
else:
    level, color = "Depresi berat", "⚫"

color_map = {
    "🟢": "#b7efc5",
    "🟡": "#fff6a5",
    "🟠": "#ffd6a5",
    "🔴": "#ffadad",
    "⚫": "#c0c0c0"
}

st.markdown(
    f"""
    <div style='background-color:{color_map[color]}; padding:10px; border-radius:10px; text-align:center;'>
        <strong>Skor PHQ-9 Anda: {phq_score}</strong><br>
        {color} <b>{level}</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("PHQ-9 adalah alat ukur standar untuk menilai tingkat keparahan depresi.")

st.divider()

# =========================================================
# Fitur Tambahan (Tidur & Olahraga)
# =========================================================
col3, col4 = st.columns(2)
with col3:
    sleep_hours = st.slider("Rata-rata Jam Tidur per Hari", 0, 12, 7)
with col4:
    exercise_freq = st.slider("Frekuensi Olahraga per Minggu", 0, 7, 2)

# =========================================================
# Encoding Input
# =========================================================
gender_encoded = 1 if gender == "Perempuan" else 0

input_data = pd.DataFrame([{
    "age": age,
    "gender": gender_encoded,
    "occupation": occupation,
    "phq_score": phq_score,
    "sleep_hours": sleep_hours,
    "exercise_freq": exercise_freq
}])

input_data = pd.get_dummies(input_data)

# =========================================================
# Prediksi
# =========================================================
st.subheader("Hasil Prediksi")

if st.button("Deteksi Tingkat Depresi"):
    try:
        prediction = model.predict(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]

        if "depresi" in result_label.lower():
            st.error(f"Hasil Deteksi: **{result_label.upper()}** 😔")
            st.markdown("> Disarankan untuk berkonsultasi dengan profesional kesehatan mental.")
        else:
            st.success(f"Hasil Deteksi: **{result_label.upper()}** 😄")
            st.markdown("> Tetap jaga kesehatan mental dan fisikmu!")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

st.divider()

# =========================================================
# Tentang Aplikasi
# =========================================================
st.subheader("Tentang Aplikasi")
st.markdown("""
- Algoritma utama: **Naive Bayes (GaussianNB)**
- Data latih: Dataset nasional kesehatan mental (versi praproses)
- Fitur utama: PHQ-9 score, usia, jam tidur, kebiasaan olahraga, dan jenis kelamin
- Tujuan: Edukasi dan penelitian untuk meningkatkan kesadaran kesehatan mental
""")

st.caption("© 2025 Mental Health Detection App — powered by Streamlit & scikit-learn")
