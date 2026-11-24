import streamlit as st
import pandas as pd
import joblib
import json

model = joblib.load('deploy/model_mobil.pkl')
with open('info.json') as f:
    info = json.load(f)

pilihan = info['pilihan']
mm = info['min_max']

st.title('Prediksi Harga Mobil Bekas')
st.write('Model:', info['model'], '| R2 test:', round(info['r2_test'], 3), '| MAE test: $', round(info['mae_test']))

brand = st.selectbox('Merek', pilihan['brand'])
fuel_type = st.selectbox('Bahan bakar', pilihan['fuel_type'])
transmission = st.selectbox('Transmisi', pilihan['transmission'])
color = st.selectbox('Warna', pilihan['color'])
service_history = st.selectbox('Riwayat servis', pilihan['service_history'])
insurance_valid = st.selectbox('Asuransi masih berlaku', pilihan['insurance_valid'])

make_year = st.number_input('Tahun pembuatan', 1990, 2026, 2015)
engine_cc = st.number_input('Kapasitas mesin (cc)', int(mm['engine_cc'][0]), int(mm['engine_cc'][1]), 1500)
mileage_kmpl = st.number_input('Efisiensi bahan bakar (km/l)', float(mm['mileage_kmpl'][0]), float(mm['mileage_kmpl'][1]), 18.0)
owner_count = st.number_input('Jumlah pemilik sebelumnya', 1, 10, 2)
accidents_reported = st.number_input('Jumlah kecelakaan', 0, 10, 0)

if st.button('Prediksi'):
    data = pd.DataFrame([{
        'mileage_kmpl': mileage_kmpl,
        'engine_cc': engine_cc,
        'fuel_type': fuel_type,
        'owner_count': owner_count,
        'brand': brand,
        'transmission': transmission,
        'color': color,
        'service_history': service_history,
        'accidents_reported': accidents_reported,
        'insurance_valid': insurance_valid,
        'car_age': 2026 - make_year
    }])
    data = data[info['urutan_kolom']]
    harga = model.predict(data)[0]
    st.success(f'Perkiraan harga: ${harga:,.0f}')
    st.write(f"Kira-kira antara ${harga - info['mae_test']:,.0f} dan ${harga + info['mae_test']:,.0f}")
