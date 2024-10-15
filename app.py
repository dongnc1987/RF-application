import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import pickle

st.title("Predicting electrical characteristics of tandem solar cells!")
st.markdown("Using an IoT system to predict electrical characteristics of tandem solar cell in real time")

# Load the min and max values from the file
with open('min_max.pkl', 'rb') as f:
    min_max = pickle.load(f)

# Extract the min and max values
time_min, time_max = min_max['time_min'], min_max['time_max']
irrad_min, irrad_max = min_max['irrad_min'], min_max['irrad_max']
cell_temp_min, cell_temp_max = min_max['cell_temp_min'], min_max['cell_temp_max']
amb_temp_min, amb_temp_max = min_max['amb_temp_min'], min_max['amb_temp_max']
humid_min, humid_max = min_max['humid_min'], min_max['humid_max']


st.header("Input predictors")

variable1 = st.slider("Time", time_min, time_max, 0.5)
col1, col2 = st.columns(2)
with col1:
    st.text("Column 1")
    variable2 = st.slider("Irradiance", irrad_min, irrad_max, 0.5)
    variable3 = st.slider("Cell temperature", cell_temp_min, cell_temp_max, 0.5)

with col2:
    st.text("Column 2")
    variable4 = st.slider("Ambient temperature", amb_temp_min, amb_temp_max, 0.5)
    variable5 = st.slider("Humidity", humid_min, humid_max, 0.5)

if st.button("Predict electrical characteristics"):
    input_array = np.array([[variable1, variable2, variable3, variable4, variable5]])
    result = predict(input_array)
    st.text("  Jsc        Voc            FF       Pmax")
    st.text(result[0])