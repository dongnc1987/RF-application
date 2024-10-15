import os
import requests
import joblib

def download_model():
    # Direct download link from Google Drive
    url = "https://drive.google.com/uc?export=download&id=1Ph1DAiX2GCZ3kuCUbj9e1u09w4-8ilTo"
    model_path = "RF_model.sav"
    
    # Check if the file already exists
    if not os.path.exists(model_path):
        # Download the file from Google Drive
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded to {model_path}")

def predict(data):
    # Ensure the model is downloaded
    download_model()
    
    # Load the model
    RF = joblib.load('RF_model.sav')
    return RF.predict(data)
