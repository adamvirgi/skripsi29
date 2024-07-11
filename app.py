import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the index labels
index_labels = {
    1: 'tinggi',
    2: 'normal',
    3: 'stunted',
    4: 'severely stunted'
}

# Set the title of the app
st.title('Stunting Prediction App')

# Get the input features from the user
tinggi_badan = st.number_input('Tinggi Badan (cm)')
berat_badan = st.number_input('Berat Badan (kg)')
umur_bulan = st.number_input('Umur (bulan)')
jenis_kelamin = st.selectbox('Jenis Kelamin', ['laki-laki', 'perempuan'])

# Preprocess the input features
input_data = np.array([[tinggi_badan, berat_badan, umur_bulan, 1 if jenis_kelamin == 'laki-laki' else 0]])
input_data_scaled = scaler.transform(input_data)

# Make the prediction
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)[0]
    st.write(f'Predicted Stunting Category: {index_labels[prediction]}')
