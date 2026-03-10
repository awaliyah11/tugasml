import streamlit as st
import numpy as np
import joblib

# ======================
# LOAD MODEL, SCALER & ENCODER
# ======================
model = joblib.load("best_model.pkl")
scaler = joblib.load("best_scaler.pkl")

le_cut = joblib.load("le_cut.pkl")
le_color = joblib.load("le_color.pkl")
le_clarity = joblib.load("le_clarity.pkl")

st.title("Prediksi Harga Diamond - DEBUG READY")
st.write("Masukkan karakteristik diamond")

# ======================
# INPUT USER
# ======================
carat = st.number_input("Carat", min_value=0.0)
cut = st.selectbox("Cut", le_cut.classes_)
color = st.selectbox("Color", le_color.classes_)
clarity = st.selectbox("Clarity", le_clarity.classes_)
depth = st.number_input("Depth", min_value=0.0)
table = st.number_input("Table", min_value=0.0)
x = st.number_input("Panjang (x)", min_value=0.0)
y = st.number_input("Lebar (y)", min_value=0.0)
z = st.number_input("Tinggi (z)", min_value=0.0)

# ======================
# TRANSFORM KATEGORI
# ======================
cut_encoded = le_cut.transform([cut])[0]
color_encoded = le_color.transform([color])[0]
clarity_encoded = le_clarity.transform([clarity])[0]

# ======================
# PREDIKSI & DEBUG
# ======================
if st.button("Prediksi Harga"):
    # Array input sesuai urutan kolom training
    data = np.array([[carat, cut_encoded, color_encoded, clarity_encoded,
                  depth, table, x, y, z]])
    
    st.write("=== Data Input (sebelum scaling) ===")
    st.write(data)
    
    # Scaling
    data_scaled = scaler.transform(data)
    st.write("=== Data Input (setelah scaling) ===")
    st.write(data_scaled)
    
    # Prediksi
    prediction = model.predict(data_scaled)
    st.success(f"Perkiraan Harga Diamond: ${prediction[0]:,.2f}")
