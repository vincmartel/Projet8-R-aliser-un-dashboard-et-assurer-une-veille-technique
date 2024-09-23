## URL de l'application Streamlit: https://scoringcredit.azurewebsites.net

import streamlit as st
import requests
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Chargement des données (echantillons de 1000 prets)
sample = pd.read_csv('data_after_feat_eng.csv') #307507 rows × 802 columns
sample=sample[:1000] #1000 premières lignes

# Initialisation de predict_button_state
if "predict_button_state" not in st.session_state:
    st.session_state.predict_button_state = False

# Affichage des informations clients à gauche
st.sidebar.image('logo.jpg')
prets_test = ["",100002, 100003, 100004, 100006, 100007, 100008, 100009, 100010, 100011, 100012]# Ajoutez d'autres champs ici (par exemple, revenu, état civil, etc.)
#prets_test = ["",100014, 100015, 100016, 100017, 100018, 100019, 100020, 100021, 100022, 100023, 100024, 100025, 100026, 100027 ]# Ajoutez d'autres champs ici (par exemple, revenu, état civil, etc.)
SK_ID_CURR = st.sidebar.selectbox("Sélectionner un prêt", prets_test)

@st.cache_data
def predict(SK_ID_CURR):
    attrib1="test"
    api_url = "http://127.0.0.1:5000/prediction"
    #api_url = "https://scoringcredit.azurewebsites.net/prediction"
    payload = {"Identifiant du prêt": SK_ID_CURR}
    try:
        # Envoi de la requête à l'API et récupération de la réponse
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            # Traitement de la réponse de l'API
            return response.json()        
        else:
            st.error("Erreur lors de l'appel à l'API")
    except requests.exceptions.RequestException as e:
            st.error(f"Erreur : {e}")
    #return None  # Retourne None en cas d'erreur

@st.cache_data
def build_df(result):
    df = pd.DataFrame({
        'features': result['features'],
        'shap_local': result['shap_local'],
        'values': result['values']
                    })
    df_sorted = df.sort_values(by="shap_local", ascending=False).head(10)
    important_feat=df_sorted['features'][:10]                     
    return df, df_sorted, important_feat

# Affichage de l'identifiant sélectionné
if SK_ID_CURR:
    st.sidebar.header("Profil du client")
    age=int(sample.loc[sample['SK_ID_CURR']==SK_ID_CURR,'DAYS_BIRTH']/365*(-1))
    sexe=sample.loc[sample['SK_ID_CURR']==SK_ID_CURR,'CODE_GENDER']
    enfants=int(sample.loc[sample['SK_ID_CURR']==SK_ID_CURR,'CNT_CHILDREN'])
    sexe=['Homme' if int(sample.loc[sample['SK_ID_CURR']==SK_ID_CURR,'CODE_GENDER'])==0 else 'Femme']
    st.sidebar.write(sexe[0])
    st.sidebar.write('Âge', age, 'ans')
    st.sidebar.write(enfants, 'enfants')

# 1 - RISQUE DE NON REMBOURSEMENT
st.subheader("Risque de non-remboursement")

@st.cache_data
def display_decision(result):
    # Affichage de la jauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(100*result['pred']),
        domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'blue'},
                'steps': [
                    {'range': [0, 37], 'color': 'green'},
                    {'range': [37, 47], 'color': 'orange'},
                    {'range': [47, 100], 'color': 'red'}
                ]}))
    st.plotly_chart(fig)
   # Affichage décision prêt
    if result['decision'] == "prêt refusé":
                st.error(f"Décision :  {result['decision']}")
    elif result['decision'] == "prêt accordé":
                st.success(f"Décision : {result['decision']}")

def callback():
# Permet de garder l'affichage du haut de la page quand la page est rafraichie (au moment de la selection feature impactante).
# fonction sera appelee dans st.selectbox 
    st.session_state.predict_button_state = True

def get_bin_client(df_sorted, selected_feature, df_sample):
        #Déterminer le bin auquel appartient le pret selectionné. Retoune l'indice du bin auquel apparitent le client
        valeur=df_sorted.loc[df_sorted['features']==selected_feature,'values']
        _, bins = np.histogram(df_sample[selected_feature], bins=10) # calcul des bins du sample  
        indice=int(np.digitize(valeur, bins)) -1
        return indice

@st.cache_data
def display_scatter(selected_feature1, selected_feature2, SK_ID_CURR):
# Création d'un graphique de dispersion avec Plotly
    fig2 = go.Figure(data=go.Scatter(
    x=sample[selected_feature1],
    y=sample[selected_feature2],
    mode='markers',
    marker=dict(size=[20 if val == SK_ID_CURR else 10 for val in sample['SK_ID_CURR']],
                color=['yellow' if val == SK_ID_CURR else 'blue' for val in sample['SK_ID_CURR']]),   
    text=sample.index  # Utilisation des indices comme étiquettes pour chaque point
))
# Personnalisation du graphique
    fig2.update_layout(
        title=selected_feature1 + ' et ' + selected_feature2,
        title_x=0.2,  # Centrage du titre,
        title_font=dict(size=30), 
        xaxis_title=selected_feature1,
        yaxis_title=selected_feature2,
        showlegend=False, 
        width=800,  # Largeur de la figure
        height=800  # Hauteur de la figure
)
# Affichage du graphique dans Streamlit
    st.plotly_chart(fig2)

if st.button("Prédire") or st.session_state.predict_button_state == True:
    result=predict(SK_ID_CURR)
    df, df_sorted, important_feat=build_df(result)
        
    display_decision(result)
    # Réinitialisez l'option sélectionnée en utilisant la session state
    st.session_state.predict_button_state = True


# 2 - CONTRIBUTION DES FEATURES AU SCORE
    st.subheader("Contribution des features au score")
    chart = alt.Chart(df_sorted).mark_bar().encode(x='shap_local', y=alt.Y('features').sort('-x') #features classées par shap_value (ordre decroissant)
                                                       ).properties(height=700)

    st.altair_chart(chart, theme="streamlit", use_container_width=True)

# 3 - COMPARAISON AVEC LES AUTRES CLIENTS
if SK_ID_CURR and st.session_state.predict_button_state:
    st.subheader("Comparaison avec les autres clients")
    important_feat=pd.concat([pd.Series(""), important_feat], ignore_index=True)
    # Liste déroulante - sélection d'une feature parmi les 10 ayant le plus impacté la décision
    selected_feature=st.selectbox("Choisir une feature impactante", important_feat, on_change=callback())

    if selected_feature:
        
    # séparation des données target=0 et target=1
        sample_0=sample[sample['TARGET']==0]
        sample_1=sample[sample['TARGET']==1]    

        valeur=df_sorted.loc[df_sorted['features']==selected_feature,'values']
        #valeur
        _, bins1 = np.histogram(sample_0[selected_feature], bins=10)
        _, bins2 = np.histogram(sample_1[selected_feature], bins=10)
        data=sample_0[selected_feature]
        data2=sample_1[selected_feature]
# Création d'une figure
        fig = go.Figure()
# Ajout du 1er histogramme en jaune
        fig.add_trace(go.Histogram(
        x=data,
        xbins=dict(start=bins1[0], end=bins1[-1], size=bins1[1] - bins1[0]),          
        marker_color=['green'] * get_bin_client(df_sorted, selected_feature, sample_0) + ['yellow'] + ['green'] * (11 - get_bin_client(df_sorted, selected_feature, sample_0)), 
        name='Classe 0: Prêts remboursés'
        ))

        fig.add_trace(go.Histogram(
        x=data2,
        xbins=dict(start=bins2[0], end=bins2[-1], size=bins2[1] - bins2[0]), #nbinsx=10,
        marker_color=['red'] * get_bin_client(df_sorted, selected_feature, sample_1) + ['yellow'] + ['red'] * (11 - get_bin_client(df_sorted, selected_feature, sample_1)),  # colorer en jaune le bin du client, les autres en rouge
        name='Classe 1: Prêts non-remboursés',
        xaxis='x2',  # Utilisation d'un deuxième axe x
        yaxis='y2'
        ))
        fig.update_yaxes(title_text="Nombre de clients")
# Mise en forme du graphique
        fig.update_layout(
        title='Distribution de ' + selected_feature,
        title_x=0.2,  # Centrage du titre,
        title_font=dict(size=30),  # Augmentation de la taille du titre
        yaxis2=dict(overlaying='y', side='right'),
        bargroupgap=0.1,  # Espacement entre les barres du même groupe
        xaxis1=dict(domain=[0, 0.45]),
        xaxis2=dict(domain=[0.50, 1]),  # Positionnement de l'axe x2
        width=1000,  # Largeur de la figure
        height=800  # Hauteur de la figure
        )   

# Affichage du graphique
        st.plotly_chart(fig)

# 4 - ANALYSE BI-VARIEE ENTRE 2 FEATURES

if SK_ID_CURR and st.session_state.predict_button_state and selected_feature:  
    st.subheader("Analyse bi-variée")
    col1, col2 = st.columns(2)
    with col1:
        selected_feature1=st.selectbox("Choisir la feature 1", important_feat)

    with col2:
        selected_feature2=st.selectbox("Choisir la feature 2", important_feat)
    
    if selected_feature1 and selected_feature1:
    # Affichage du scatter si les deux features sont sélectionnées
        display_scatter(selected_feature1, selected_feature2, SK_ID_CURR)
    else:
        st.info("Sélectionnez les deux fonctionnalités pour afficher l'analyse bi-variée.")