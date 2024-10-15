import os
import joblib
import requests
import streamlit as st

def download_model():
    # Direct download link from Google Drive
    url = "https://drive.google.com/uc?export=download&id=1Ph1DAiX2GCZ3kuCUbj9e1u09w4-8ilTo"
    model_path = "RF_model.sav"
    return model_path

def predict(data):
    # Ensure the model is downloaded
    model_path = download_model()

    RF = joblib.load(model_path)
    # Make predictions
    result = RF.predict(data)
    return result
