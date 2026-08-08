import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.write(''' # Predicción de Diabetes ''')

st.image("diabetes.jpg", caption="Predicción de Diabetes mediante Machine Learning.")

st.header('Datos del paciente')

def user_input_features():

    # Entrada

    Pregnancies = st.number_input('Número de embarazos:', min_value=0, max_value=20, value=0, step=1)

    Glucose = st.number_input('Nivel de glucosa:', min_value=0, max_value=300, value=120, step=1)

    BloodPressure = st.number_input('Presión arterial:', min_value=0, max_value=200, value=70, step=1)

    SkinThickness = st.number_input('Espesor de la piel:', min_value=0, max_value=100, value=20, step=1)

    Insulin = st.number_input('Nivel de insulina:', min_value=0, max_value=900, value=80, step=1)

    BMI = st.number_input('Índice de masa corporal (BMI):', min_value=0.0, max_value=70.0, value=25.0)

    DiabetesPedigreeFunction = st.number_input(
        'Factor hereditario:',
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    Age = st.number_input('Edad:', min_value=1, max_value=120, value=30, step=1)

    user_input_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    features = pd.DataFrame(user_input_data, index=[0])

    return features


df = user_input_features()

diabetes = pd.read_csv('diabetes_limpio.csv')

X = diabetes.drop(columns='Outcome')
Y = diabetes['Outcome']

classifier = DecisionTreeClassifier(
    max_depth=8,
    criterion='entropy',
    min_samples_leaf=10,
    random_state=0
)

classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')

if prediction[0] == 0:
    st.write('El paciente NO presenta diabetes')
elif prediction[0] == 1:
    st.write('El paciente PRESENTA diabetes')
else:
    st.write('Sin predicción')
