import os
import joblib
import requests
import streamlit as st

def download_model():
    # Direct download link from Google Drive
    url = "https://drive.google.com/uc?export=download&id=1Ph1DAiX2GCZ3kuCUbj9e1u09w4-8ilTo"  # Replace with your direct download link
    model_path = "RF_model.sav"
    
    # Check if the model file already exists
    if not os.path.exists(model_path):
        # Download the model from Google Drive
        st.write("Downloading the model...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for request errors
            
            # Save the model to the local file system
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    if chunk:
                        f.write(chunk)
            st.write(f"Model downloaded to {model_path} with size {os.path.getsize(model_path)} bytes")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading the model: {e}")
            return None

    return model_path


def predict(data):
    # Ensure the model is downloaded
    model_path = download_model()

    if model_path is None:
        st.error("Model could not be downloaded.")
        return None
    
    try:
        # Load the model
        st.write("Loading the model...")
        RF = joblib.load(model_path)
        st.write("Model loaded successfully.")

        # Make predictions
        return RF.predict(data)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None
