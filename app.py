import os
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. Load Model dengan Path Relatif Aman
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_joblib.pkl')

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"File model tidak ditemukan di: {path}. Pastikan file 'model_joblib.pkl' berada di folder yang sama dengan app.py!")
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ---------------------------------------------------------
# 2. Judul & Sidebar App
# ---------------------------------------------------------
st.title('Prediksi Tagihan Listrik Jakarta')
st.write('Aplikasi untuk memprediksi jumlah tagihan listrik berdasarkan parameter yang diberikan.')

st.sidebar.header('Input Parameter')

def user_input_features():
    kwh = st.sidebar.slider('Konsumsi KWH (kWh)', 150.0, 600.0, 350.0)
    ac_units = st.sidebar.slider('Jumlah AC', 0, 3, 1)
    ac_hours_per_day = st.sidebar.slider('Jam AC per Hari', 0.0, 10.0, 5.0)
    family_size = st.sidebar.slider('Jumlah Anggota Keluarga', 2, 6, 4)
    month_name = st.sidebar.selectbox('Bulan', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    tariff_class = st.sidebar.selectbox('Kelas Tarif', ['R1', 'R2', 'R3'])

    data = {
        'kwh': [kwh],
        'ac_units': [ac_units],
        'ac_hours_per_day': [ac_hours_per_day],
        'family_size': [family_size],
        'month_name': [month_name],
        'tariff_class': [tariff_class]
    }
    return pd.DataFrame(data)

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# ---------------------------------------------------------
# 3. Preprocessing Feature Encoding & Alignment
# ---------------------------------------------------------
# Urutan kolom & tipe data persis seperti yang digunakan saat pelatihan
training_columns_and_dtypes = {
    'kwh': 'float64',
    'ac_units': 'int64',
    'ac_hours_per_day': 'float64',
    'family_size': 'int64',
    'month_name_Aug': 'bool',
    'month_name_Dec': 'bool',
    'month_name_Feb': 'bool',
    'month_name_Jan': 'bool',
    'month_name_Jul': 'bool',
    'month_name_Jun': 'bool',
    'month_name_Mar': 'bool',
    'month_name_May': 'bool',
    'month_name_Nov': 'bool',
    'month_name_Oct': 'bool',
    'month_name_Sep': 'bool',
    'tariff_class_R2': 'bool',
    'tariff_class_R3': 'bool'
}

# Terapkan One-Hot Encoding pada input pengguna
encoded_input = pd.get_dummies(df_input, columns=['month_name', 'tariff_class'], drop_first=False)

# Reindex agar urutan & keberadaan kolom persis sama dengan training_columns, lalu isi nilai kosong dengan False/0
final_input_df = encoded_input.reindex(columns=training_columns_and_dtypes.keys(), fill_value=False)

# Konversi tipe data sesuai tipe data training
for col, dtype in training_columns_and_dtypes.items():
    final_input_df[col] = final_input_df[col].astype(dtype)

# ---------------------------------------------------------
# 4. Melakukan Prediksi
# ---------------------------------------------------------
if st.sidebar.button('Prediksi Tagihan'):
    if model is None:
        st.error("Model gagal di-load. Periksa ketersediaan file model_joblib.pkl di server/repository.")
    else:
        try:
            prediction = model.predict(final_input_df)
            st.subheader('Hasil Prediksi Tagihan Listrik:')
            st.success(f"Tagihan Diprediksi: **Rp {prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.exception(e)
