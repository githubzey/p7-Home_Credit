from fastapi import status
import requests
import json

#API_URL = "http://127.0.0.1:8000/"
API_URL = "https://apihomecredit-861d00eaed91.herokuapp.com/"


def test_homepage():
    
    response = requests.get(API_URL)
    assert response.status_code == 200
    assert json.loads(response.content) == 'Welcome to the API'






def test_predict_home_credit():
    url_get_pred = API_URL + "predict" 
    #client_data = data_test_imp[data_test_imp['SK_ID_CURR']==int(client_id)].drop('SK_ID_CURR', axis=1)
    #client_data_json = client_data.to_json(orient="records")
    client_data_json ="[{\"EXT_SOURCE_2\":0.0304315707,\"EXT_SOURCE_3\":0.1107962886,\"CODE_GENDER\":0.7209239877,\"PAYMENT_RATE\":-0.2418253653,\"DAYS_EMPLOYED\":0.2833922874,\"PREV_CNT_PAYMENT_MEAN\":-0.2903476431,\"NAME_EDUCATION_TYPE_Highereducation\":1.7625702765,\"INSTAL_DPD_MEAN\":-0.052079095,\"DAYS_BIRTH\":-0.9914858035,\"AMT_ANNUITY\":0.6444642056}]"
    #client_data_json = '[{"EXT_SOURCE_2":0.0304315707,"EXT_SOURCE_3":0.1107962886,"CODE_GENDER":0.7209239877,"PAYMENT_RATE":-0.2418253653,"DAYS_EMPLOYED":-0.2827170179,"PREV_CNT_PAYMENT_MEAN":-0.2903476431,"INSTAL_DPD_MEAN":-0.052079095,"NAME_EDUCATION_TYPE_Highereducation":1.7625702765,"DAYS_BIRTH":0.9917147317,"AMT_ANNUITY":0.6444642056}]'
    response = requests.post(url_get_pred , json={"data": client_data_json})
    prediction = response.json()["proba"]                   
    assert response.status_code == 200
    assert prediction == 0.38



