import joblib

def predict(data):
    RF = joblib.load("RF_model.sav")
    return RF.predict(data)