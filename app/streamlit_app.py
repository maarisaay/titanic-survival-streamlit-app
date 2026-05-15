import streamlit as st
import pandas as pd
import pickle

st.title("Titanic Survival Prediction")

st.write(
    "Aplikacja przewiduje, czy pasażer Titanica przeżyłby katastrofę "
    "na podstawie danych takich jak wiek, płeć, klasa biletu i opłata."
)

with open("../models/titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("../models/model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

st.header("Dane pasażera")

pclass = st.radio("Klasa biletu", [1, 2, 3])
sex = st.radio("Płeć", ["male", "female"])
age = st.slider("Wiek", 0, 100, 30)
sibsp = st.slider("Liczba rodzeństwa/małżonków na pokładzie", 0, 8, 0)
parch = st.slider("Liczba rodziców/dzieci na pokładzie", 0, 6, 0)
fare = st.number_input("Opłata za bilet", min_value=0.0, value=30.0)
embarked = st.selectbox("Port wejścia na pokład", ["C", "Q", "S"])

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

input_data = pd.get_dummies(input_data, columns=["Sex", "Embarked"], drop_first=True)

input_data = input_data.reindex(columns=model_columns, fill_value=0)

if st.button("Przewidź"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Pasażer prawdopodobnie przeżyłby katastrofę.")
    else:
        st.error("Pasażer prawdopodobnie nie przeżyłby katastrofy.")

    st.write(f"Prawdopodobieństwo przeżycia: {probability:.2%}")