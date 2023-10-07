# Library imports
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import uvicorn
import shap
import json
from sklearn.pipeline import Pipeline

class Item(BaseModel):
    EXT_SOURCE_2: float = 0.0304315707
    EXT_SOURCE_3:float = 0.1107962886
    CODE_GENDER:float = 0.7209239877
    PAYMENT_RATE:float = -0.2418253653
    DAYS_EMPLOYED:float = 0.2833922874
    PREV_CNT_PAYMENT_MEAN:float = -0.2903476431
    NAME_EDUCATION_TYPE_Highereducation:float = -0.052079095
    INSTAL_DPD_MEAN:float = -0.7625702765
    DAYS_BIRTH:float = -0.9914858035
    AMT_ANNUITY:float = 0.6444642056


# Loading the model and data
model = pickle.load(open('best_model.pkl', 'rb'))
pipeline_preprocess = pickle.load(open('pipeline_preprocess.pkl', 'rb'))
explainer = shap.TreeExplainer(model)

# Create a FastAPI instance
app = FastAPI()

# Functions
@app.get('/')
def home():

    return 'Welcome to API Home Credit test'

class ClientData(BaseModel):
    data: str

@app.post('/predicts_items')
async def categorize_trx(item: Item):
    # Check if the trx label is in the url
    df = pd.DataFrame(item.dict(), index=[0])
    print(df)
    try:
        results = model.predict_proba(df)
        return {'item': results[0][1]}
    except:
        return 'Error: No id field provided. Please specify a label.'
    


@app.post("/predict")
def predict_home_credit(client_data: ClientData):
    
    client_null = pd.read_json(client_data.data)
    client_df = pipeline_preprocess.transform(client_null)
    prediction = model.predict_proba(client_df)[0][1].round(2)
    return {"proba": prediction}


@app.post('/shaplocal/')
def shap_values_local(client_data: ClientData):
   
    client_null = pd.read_json(client_data.data)
    client_df = pipeline_preprocess.transform(client_null)
    shap_val = explainer(client_df)[0][:, 1]

    return {'shap_values': shap_val.values.tolist(),
            'base_value': shap_val.base_values}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn api:app --reload