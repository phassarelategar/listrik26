import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the trained model
model = joblib.load('model_joblib.pkl') # Changed model filename and loading method

# Streamlit app title
st.title('Prediksi Tagihan Listrik Jakarta')
st.write('Aplikasi untuk memprediksi jumlah tagihan listrik berdasarkan parameter yang diberikan.')

# Sidebar for user inputs
st.sidebar.header('Input Parameter')

def user_input_features():
    kwh = st.sidebar.slider('Konsumsi KWH (kWh)', 150.0, 600.0, 350.0)
    ac_units = st.sidebar.slider('Jumlah AC', 0, 3, 1)
    ac_hours_per_day = st.sidebar.slider('Jam AC per Hari', 0.0, 10.0, 5.0)
    family_size = st.sidebar.slider('Jumlah Anggota Keluarga', 2, 6, 4)
    month_name = st.sidebar.selectbox('Bulan', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    tariff_class = st.sidebar.selectbox('Kelas Tarif', ['R1', 'R2', 'R3'])

    data = {
        'kwh': kwh,
        'ac_units': ac_units,
        'ac_hours_per_day': ac_hours_per_day,
        'family_size': family_size,
        'month_name': month_name,
        'tariff_class': tariff_class
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# Define the exact columns and their dtypes expected by the model during training
# This list ensures correct order and includes all dummy variables used during training
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

# Create an empty DataFrame with the correct columns and dtypes
final_input_df = pd.DataFrame(columns=training_columns_and_dtypes.keys())
for col, dtype in training_columns_and_dtypes.items():
    final_input_df[col] = final_input_df[col].astype(dtype)

# Add a single row of data, initially all zeros/False
final_input_df.loc[0] = 0
for col, dtype in training_columns_and_dtypes.items():
    if dtype == 'bool':
        final_input_df.loc[0, col] = False

# Populate numerical features
final_input_df.loc[0, 'kwh'] = df_input['kwh'][0]
final_input_df.loc[0, 'ac_units'] = df_input['ac_units'][0]
final_input_df.loc[0, 'ac_hours_per_day'] = df_input['ac_hours_per_day'][0]
final_input_df.loc[0, 'family_size'] = df_input['family_size'][0]

# Populate one-hot encoded categorical features
selected_month_col = f"month_name_{df_input['month_name'][0]}"
if selected_month_col in final_input_df.columns:
    final_input_df.loc[0, selected_month_col] = True

selected_tariff_col = f"tariff_class_{df_input['tariff_class'][0]}"
if selected_tariff_col in final_input_df.columns:
    final_input_df.loc[0, selected_tariff_col] = True

# Make prediction
if st.sidebar.button('Prediksi Tagihan'):
    try:
        prediction = model.predict(final_input_df) # Use the new DataFrame name
        st.subheader('Hasil Prediksi Tagihan Listrik:')
        st.write(f"Tagihan Diprediksi: Rp {prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e) # Show full traceback
