



# Library imports
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle
import uvicorn
import shap
from sklearn.preprocessing import StandardScaler
# Create a FastAPI instance
app = FastAPI()

# Loading the model and data
path = "data_api/"
model = pickle.load(open(path + 'model.pkl', 'rb'))
data_test = pd.read_csv(path + 'df_test_api.csv')
data_train = pd.read_csv(path + 'df_train_api.csv')
data_app = pd.read_csv(path + 'application_train.csv')
model = model
data_train['SK_ID_CURR'] = data_train['SK_ID_CURR'].astype(int)
data_test['SK_ID_CURR'] = data_test['SK_ID_CURR'].astype(int)
data_train = data_train.set_index('SK_ID_CURR')
data_test = data_test.set_index('SK_ID_CURR')
scaler =  StandardScaler()
X_train_scaled = scaler.fit_transform(data_train)
X_test_scaled = scaler.transform(data_test)
cols_train = data_train.columns
cols_test = data_test.columns

X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols_train, index=data_train.index).reset_index()
X_test_scaled = pd.DataFrame(X_test_scaled, columns=cols_test, index=data_test.index).reset_index()

data_app['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
# On va remplacer les valeurs infinies par NaN
data_app.replace([np.inf, -np.inf], np.nan, inplace=True)

def serialize_nan(value):
    if pd.isna(value):
        return "None"
    return value

# Apply custom serialization to all columns
data_app = data_app.applymap(serialize_nan)

explainer = shap.TreeExplainer(model)

# Functions
@app.get('/')
def home():

    return 'Welcome to the API'

@app.get('/information/{client_id}')
def information(client_id: int):
   
  
    client_data = data_app[data_app['SK_ID_CURR'] == client_id]
    info_client = client_data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    info_client_dict = info_client.to_dict(orient="records")
    return info_client_dict
    #return info_client

@app.get('/prediction/{client_id}')
def prediction(client_id: int):
    
    client_data = X_test_scaled[X_test_scaled['SK_ID_CURR'] == client_id]
    info_client = client_data.drop('SK_ID_CURR', axis=1)
    prediction = model.predict_proba(info_client)[0][1].round(2)
    return prediction




if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn api:app --reload