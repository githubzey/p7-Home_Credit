# Import des librairies
import streamlit as st
from PIL import Image
import shap
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# local
#API_URL = "http://localhost:8000/"
# deployment cloud
API_URL = "https://apihomecredit-861d00eaed91.herokuapp.com/"

data_dash = pd.read_csv("filtered_data.csv")
data_test_imp = pd.read_csv("data_test_imp.csv")


def prediction(client_id):
    url_get_pred = API_URL + "predict" 
    client_data = data_test_imp[data_test_imp['SK_ID_CURR']==int(client_id)].drop('SK_ID_CURR', axis=1)
    client_data_json = client_data.to_json(orient="records")
    response = requests.post(url_get_pred , json={"data": client_data_json})
    prediction = response.json()["proba"]
    return prediction


# To set a webpage title, header and subtitle
st.set_page_config(page_title="Home Credit", layout='wide')
st.title("Décision Home Credit")

def info_client(client_id):
    data_dash = pd.read_csv("filtered_data.csv")
    df_info = data_dash[data_dash['SK_ID_CURR']==client_id]
    st.dataframe(data=df_info)
    #st.write(df_info.drop(columns=['TARGET'],axis=1))
    st.markdown("<h5 style='font-weight: bold;'>Informations de client :</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.write("ID client : " + str(client_id))
    col1.write("Genre : " + df_info['CODE_GENDER'].item())
    col1.write("Age : " + str(int((df_info['DAYS_BIRTH'] / -365))))
    col1.write("Type d'éducation : " + df_info['NAME_EDUCATION_TYPE'].item())
    col1.write("Statut familial : " + df_info['NAME_FAMILY_STATUS'].item())
    col1.write("Nombre d'enfant : " + str(df_info['CNT_CHILDREN'].item()))
    col1.write("Type de contrat : " + df_info['NAME_CONTRACT_TYPE'].item())

    col2.write("Type de revenu : " + df_info['NAME_INCOME_TYPE'].item()) 
    col2.write("Revenu total : " + str(df_info['AMT_INCOME_TOTAL'].item()))
    col2.write("Prêt total : " + str(df_info['AMT_INCOME_TOTAL'].item()))
    col2.write("Durée travail(année) : " + str(int((df_info['DAYS_EMPLOYED']/ -365))))
    col2.write("Propriétaire maison/appartement : " + df_info['FLAG_OWN_REALTY'].item())
    col2.write("Type de logement : " + df_info['NAME_HOUSING_TYPE'].item())

def score_risque(proba_fail):
   
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = proba_fail * 100,
        mode = "gauge+number+delta",
        title = {'text': "Score risque"},
        delta = {'reference': 50},
        gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
               'bar': {'color': "MidnightBlue"},
               'steps': [
                   {'range': [0, 20], 'color': "Green"},
                   {'range': [20, 45], 'color': "LimeGreen"},
                   {'range': [45, 50], 'color': "Orange"},
                   {'range': [50, 100], 'color': "Red"}],
               'threshold': {'line': {'color': "black", 'width': 6}, 'thickness': 1, 'value': 50}}))
    fig.update_layout(width=400, height=300)
    st.plotly_chart(fig)


def shap_val_local():
  
    url_get_shap_local = API_URL + "shaplocal/"
    client_data = data_test_imp[data_test_imp['SK_ID_CURR']==int(client_id)].drop('SK_ID_CURR', axis=1)
    client_data_json = client_data.to_json(orient="records")
    response = requests.post(url_get_shap_local, json={"data": client_data_json})
    res = json.loads(response.content)
    shap_val_local = res['shap_values']
    base_values = res['base_values']

    explanation = shap.Explanation(np.reshape(np.array(shap_val_local, dtype='float'), (1, -1)),
                                   base_values, 
                                   data=np.reshape(np.array(client_data.values.tolist(), dtype='float'),
                                                    (1, -1)),
                                   feature_names=client_data.columns)

    return explanation[0]


Selections = ["Home",
         "Information client",
		 "Score et décision", "Comparaison"]

st.sidebar.image('image/logo.png')
st.sidebar.title('Menu')
selection = st.sidebar.radio("Choissez votre page.", Selections)


if selection == "Home":
    st.subheader("Home Page")
    st.markdown("Cette application permet d'expliquer aux clients les motifs"
                " d'approbation ou de refus de leurs demandes de crédit.\n"
                
                "\nEn se basant sur l'historique de prêt du client et ses informations personnelles,"
                "il prédit s'il est susceptible de rembourser un crédit."
                "Les prédictions sont calculées à partir d'un algorithme d'apprentissage automatique, "
                "préalablement entraîné." 
                "Il s'agit d'un modèle *Light GBM* (Light Gradient Boosting Machine).\n "
                
                
                "\nLe dashboard est composé de plusieurs pages :\n"

                "\n- **Information du client**: Vous y trouverez toutes les informations relatives au client sélectionné .\n"
                "- **Score et décision**: Vous y trouverez quelles caractéristiques du client ont le plus"
                "influencé le choix d'approbation ou de refus de la demande de crédit et"
                "la décision de l'algorithme.\n")
                

if selection == "Information client":
    client_id = st.selectbox("Sélectionnez le numéro du client", data_test_imp['SK_ID_CURR'])
    st.subheader("Les informations des clients")
    info_client(client_id)

    
if selection == "Score et décision":
    client_id = st.selectbox("Sélectionnez le numéro du client", data_test_imp['SK_ID_CURR'])
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Visualisation des scores de prédiction")
        proba_fail = prediction(client_id)
        st.write('La probabilité de faillite du client : '+str(int(proba_fail * 100))+ "%")
        threshold = 0.5
        if proba_fail <= threshold:
                    st.success("Crédit accepté", icon="✅")
        else:
                    st.error("Crédit refusé", icon="🚨")

    with col2:
        score_risque(proba_fail)

    
    col1, col2 = st.columns([7,5])
    with col1:
        shap_val = shap_val_local()
        # Affichage du waterfall plot : shap local
        fig, ax = plt.subplots()
        #fig = plt.figure()
        shap.waterfall_plot(shap_val, show=False)
        st.pyplot(fig)     
    with col2:
         st.image('image/shap_global.png')
         
if selection == "Comparaison":
    client_id = st.selectbox("Sélectionnez le numéro du client", data_test_imp['SK_ID_CURR'])
   



# streamlit run dashboard.py









