import streamlit as st
import joblib
import numpy as np

# === Konfigurasi halaman ===
st.set_page_config(page_title="Prediksi Diabetes", page_icon="💉", layout="centered")

# === Load model & scaler ===
model = joblib.load("model_diabetes_clean.pkl")
scaler = joblib.load("scaler_clean.pkl")

# === Header modern ===
st.markdown("""
    <style>
        .title {
            font-size: 2.5em;
            font-weight: bold;
            text-align: center;
            color: #3b82f6;
        }
        .subtitle {
            font-size: 1.1em;
            text-align: center;
            color: #6b7280;
            margin-bottom: 30px;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #9ca3af;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Prediksi Risiko Diabetes</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Masukkan data pasien untuk memprediksi kemungkinan diabetes berdasarkan model AI</div>', unsafe_allow_html=True)

# === Input Form ===
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Jumlah Kehamilan", 0, 20, 1)
        glucose = st.number_input("Glukosa (mg/dL)", 1, 300, 120)
        bp = st.number_input("Tekanan Darah (mm Hg)", 1, 180, 70)
        skin = st.number_input("Ketebalan Kulit (mm)", 1, 100, 20)

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", 1, 900, 85)
        bmi = st.number_input("BMI", 1.0, 70.0, 24.0)
        dpf = st.number_input("Riwayat Keluarga Diabetes (DPF)", 0.0, 2.5, 0.5)
        age = st.number_input("Umur", 10, 100, 33)

    submitted = st.form_submit_button("🔍 Prediksi")

# === Prediksi & Output ===
if submitted:
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ Hasil: Berisiko Diabetes.\n\nProbabilitas: **{proba:.2f}%**")
    else:
        st.success(f"✅ Hasil: Tidak Berisiko Diabetes.\n\nProbabilitas: **{proba:.2f}%**")

# === Footer ===
st.markdown('<div class="footer">© 2025 Aplikasi Prediksi Diabetes </div>', unsafe_allow_html=True)
