import streamlit as st
import joblib

st.title('Wine Clustering')
Wine_Model = joblib.load("wine_model.pkl")

Alcohol= st.number_input('Enter the alcohol percentage: ', min_value=11., max_value=14.8, value=11., step=1., format="%.2f")
Malic_Acid = st.number_input('Enter the Malic_Acid percentage: ', min_value=0.74, max_value=5.8, value=0.74, step=1., format="%.2f")
Ash = st.number_input('Enter the Ash percentage: ', min_value=1.36, max_value=3.23, value=1.36, step=1., format="%.2f")
Ash_Alcanity= st.number_input('Enter the Ash_Alcanity percentage: ', min_value=10.6,max_value=30., value=10.6, step=1., format="%.2f")
Total_Phenols = st.number_input('Enter the Total_Phenols: ', min_value=0.98 ,max_value=3.88, value=0.98 , step=1., format="%.2f")
Flavanoids= st.number_input('Enter the Flavanoids: ', min_value=0.74, max_value=5.8, value=0.74, step=1., format="%.2f")
Nonflavanoid_Phenols = st.number_input('Enter the Nonflavanoid_Phenols: ', min_value=0.13, max_value=0.66, value=0.13, step=1., format="%.2f")
Proanthocyanins = st.number_input('Enter the Proanthocyanins percentage: ', min_value=0.41
, max_value=3.58, value=0.41, step=1., format="%.2f")
Color_Intensity	 = st.number_input('Enter the Color_Intensity	: ', min_value=1.28
, max_value=13., value=1.28, step=1., format="%.2f")
Hue	= st.number_input('Enter the Hue	: ', min_value=0.48
, max_value=1.71, value=0.48, step=1., format="%.2f")
OD280_percentage= st.number_input('Enter the OD280: ', min_value=1.27
, max_value=4., value=1.27, step=1., format="%.2f")

button_clicked = st.button("Predict")
    
if button_clicked:
    data = [Alcohol,Malic_Acid,Ash,Ash_Alcanity,Total_Phenols,Flavanoids,Nonflavanoid_Phenols,Proanthocyanins,Color_Intensity,Hue,OD280_percentage
    ]

    cluster_assignment = Wine_Model.predict([data])

    cluster_labels = {
        0: 'Group 1',
        1: 'Group 2',
        2: 'Group 3'
    }

    # Display the predicted cluster
    st.success(f'Based on the provided features, your wine cluster is: {cluster_labels[cluster_assignment[0]]}')