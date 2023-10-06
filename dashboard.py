# Import des librairies
import streamlit as st
from PIL import Image
import shap
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
# local
#API_URL = "http://localhost:8000/"
# deployment cloud
API_URL = "https://apihomecredit-861d00eaed91.herokuapp.com/"

pipeline_preprocess = pickle.load(open('pipeline_preprocess.pkl', 'rb'))

data_dash = pd.read_csv("data/filtered_data_sample.csv")
data_test_sample = pd.read_csv("data/data_test_sample.csv")
data_knn = pd.read_csv('data/data_knn_sample.csv')
data_indexed = data_test_sample.set_index("SK_ID_CURR")
data_process = pipeline_preprocess.transform(data_indexed)
data_process = pd.DataFrame(data_process, columns=data_indexed.columns, index=data_indexed.index)
data_scaled = data_process.reset_index()



def prediction(client_id):
    url_get_pred = API_URL + "predict" 
    client_data = data_test_sample[data_test_sample['SK_ID_CURR']==int(client_id)].drop('SK_ID_CURR', axis=1)
    client_data_json = client_data.to_json(orient="records", default_handler=str)
    response = requests.post(url_get_pred , json={"data": client_data_json})
    prediction = response.json()["proba"]
    return prediction


# To set a webpage title, header and subtitle
st.set_page_config(page_title="Home Credit", layout='wide')
st.title("Décision Home Credit")

# Function to handle NaN values 

# Function to handle NaN values
def handle_nan(value):
    if isinstance(value, pd.Series):
        return value.apply(lambda x: "None" if pd.isna(x) else str(x))
    else:
        return "None" if pd.isna(value) else str(value)


def info_client(client_id):
    data_dash = pd.read_csv("data/filtered_data_sample.csv")
    df_info = data_dash[data_dash['SK_ID_CURR']==client_id]
    st.dataframe(data=df_info)
    st.markdown("<h5 style='font-weight: bold;'>Informations de client :</h5>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.write("ID client : " + str(client_id))
    col1.write("Genre : " + handle_nan(df_info['CODE_GENDER'].item()))
    col1.write("Age : " + handle_nan(df_info['DAYS_BIRTH'].item()))
    col1.write("Type d'éducation : " + handle_nan(df_info['NAME_EDUCATION_TYPE'].item()))
    col1.write("Statut familial : " + handle_nan(df_info['NAME_FAMILY_STATUS'].item()))
    col1.write("Nombre d'enfant : " + handle_nan(df_info['CNT_CHILDREN'].item()))
    col1.write("Type de contrat : " + handle_nan(df_info['NAME_CONTRACT_TYPE'].item()))

    col2.write("Type de revenu : " + handle_nan(df_info['NAME_INCOME_TYPE'].item()))
    col2.write("Revenu total : " + handle_nan(df_info['AMT_INCOME_TOTAL'].item()))
    col2.write("Prêt total : " + handle_nan(df_info['AMT_CREDIT'].item()))
    col2.write("Durée travail(année) : " + handle_nan(df_info['DAYS_EMPLOYED'].item()))
    col2.write("Propriétaire maison/appartement : " + handle_nan(df_info['FLAG_OWN_REALTY'].item()))
    col2.write("Type de logement : " + handle_nan(df_info['NAME_HOUSING_TYPE'].item()))

def score_risque(proba_fail):
   
    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = proba_fail * 100,
        mode = "gauge+number+delta",
        title = {'text': "Score risque"},
        delta = {'reference': 54},
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
    client_data = data_test_sample[data_test_sample['SK_ID_CURR']==int(client_id)].drop('SK_ID_CURR', axis=1)
    client_data_json = client_data.to_json(orient="records", default_handler=str)
    response = requests.post(url_get_shap_local, json={"data": client_data_json})
    res = json.loads(response.content)
    shap_val_local = res['shap_values']
    base_values = res['base_value']

    explanation = shap.Explanation(np.reshape(np.array(shap_val_local, dtype='float'), (1, -1)),
                                   base_values, 
                                   data=np.reshape(np.array(client_data.values.tolist(), dtype='float'),
                                                    (1, -1)),
                                   feature_names=client_data.columns)

    return explanation[0]

def comparaison(data, client_id, num_neighbors):
     features_of_interest = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'CODE_GENDER',
       'PAYMENT_RATE', 'DAYS_EMPLOYED', 'INSTAL_DPD_MEAN',
       'PREV_CNT_PAYMENT_MEAN', 'NAME_EDUCATION_TYPE_Highereducation',
       'DAYS_BIRTH', 'AMT_ANNUITY']
     selected_client = data[data['SK_ID_CURR'] == client_id]
     features_of_interest = features_of_interest 
     selected_client_features = selected_client[features_of_interest]
     knn = NearestNeighbors(n_neighbors=num_neighbors) 
     knn.fit(data_knn[features_of_interest])
     # Find similar clients
     similar_clients_indices = knn.kneighbors(selected_client_features)[1]
     similar_clients = data_knn.iloc[similar_clients_indices[0]]
     return selected_client, similar_clients 




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
    client_id = st.selectbox("Sélectionnez le numéro du client", data_test_sample['SK_ID_CURR'])
    st.subheader("Les informations des clients")
    info_client(client_id)

    
if selection == "Score et décision":
    client_id = st.selectbox("Sélectionnez le numéro du client", data_test_sample['SK_ID_CURR'])
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Visualisation des scores de prédiction")
        proba_fail = prediction(client_id)
        st.write('La probabilité de faillite du client : '+str(int(proba_fail * 100))+ "%")
        threshold = 0.54
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
         st.image('image/shap_global2.png')
         
if selection == "Comparaison":
    features_of_interest = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'CODE_GENDER',
       'PAYMENT_RATE', 'DAYS_EMPLOYED', 'INSTAL_DPD_MEAN',
       'PREV_CNT_PAYMENT_MEAN', 'NAME_EDUCATION_TYPE_Highereducation',
       'DAYS_BIRTH', 'AMT_ANNUITY']
    client_id = st.selectbox("Sélectionnez le numéro du client", data_test_sample['SK_ID_CURR'])
    # Sidebar to choose the number of neighbors for KNN
    num_neighbors = st.slider('Number of Neighbors', min_value=100, max_value=10000, value=500)
    # Sidebar to select a feature
    selected_feature = st.selectbox('Select a Feature', features_of_interest)

    selected_client, similar_clients = comparaison(data_scaled, client_id, num_neighbors)
    # Create a DataFrame for selected clients (including the chosen client)
    
    filtered_graph = similar_clients[[selected_feature, 'TARGET']]
    
    if selected_feature in ['CODE_GENDER', 'NAME_EDUCATION_TYPE_Highereducation']:
        #fig = px.bar(filtered_graph, x=selected_feature, color='TARGET', barmode='group')
        fig = px.histogram(filtered_graph, x=selected_feature, color='TARGET',
                           color_discrete_map={0: 'green', 1: 'red'}, barmode='group')

        # Add a marker for the selected client
        selected_client_value = selected_client[selected_feature].values[0]
        fig.add_shape(
            type='line',
            x0=selected_client_value,
            x1=selected_client_value,
            y0=0,
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color='black', width=2)
        )

        # Add a text label near the marker
        fig.add_annotation(
            text='Selected Client',
            x=selected_client_value,
            y=1.05,  # Adjust the Y position for the label
            showarrow=False,
            font=dict(color='black', size=12)
        )



        fig.update_layout(title=f'Count Plot for {selected_feature}')
        st.plotly_chart(fig)
    else:  # Show Box Plot
        
        selected_client_value = selected_client[selected_feature].values[0]
        # Create a box plot using Plotly
        fig = px.box(filtered_graph, x='TARGET', y=selected_feature, color='TARGET',
                     color_discrete_map={0: 'green', 1: 'red'})

        # Add a marker line for the selected client
        fig.add_shape(type='line',
                    x0=-0.5, x1=1.5, y0=selected_client_value, y1=selected_client_value,
                    line=dict(color='black', width=2))

        # Add a text label near the marker
        fig.add_annotation(
            text='Selected Client',
            x=selected_client_value,
            y=1.05,  # Adjust the Y position for the label
            showarrow=False,
            font=dict(color='black', size=12)
        )


        # Customize the layout
        fig.update_layout(title=f'Box Plot for {selected_feature}',
                        xaxis_title='Credit Status',
                        yaxis_title=selected_feature,
                        legend_title='TARGET')

        # Show the plot
        st.plotly_chart(fig)



    st.write("Legend:")
    st.write("0: Credit Accepted")
    st.write("1: Credit Refused")


# streamlit run dashboard.py