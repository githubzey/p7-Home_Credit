
# Library imports
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import uvicorn
import shap
import json

# Loading the model and data
#path = "data_api/"
model = pickle.load(open('lgbm_imp.pkl', 'rb'))
explainer = shap.TreeExplainer(model)

# Create a FastAPI instance
app = FastAPI()

# Functions
@app.get('/')
def home():

    return 'Welcome to the API'

class ClientData(BaseModel):
    data: str  # JSON-encoded DataFrame

@app.post("/predict")
def predict_home_credit(client_data: ClientData):

    # Convert the JSON-encoded DataFrame back to a DataFrame
    client_df = pd.read_json(client_data.data)
    # Make a prediction using your model
    prediction = model.predict_proba(client_df)[0][1].round(2)
    return {"proba": prediction}


@app.post('/shaplocal/')
def shap_values_local(client_data: ClientData):
   
    client_df = pd.read_json(client_data.data)
    shap_val = explainer(client_df)[0][:, 1]

    return {'shap_values': shap_val.values.tolist(),
            'base_values': shap_val.base_values}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn api:app --reload