import streamlit as st
import pandas as pd
import joblib as jb

model = jb.load("model.pkl")
scaler = jb.load("scaler.pkl")
expected_columns = jb.load("columns.pkl")

st.title("Titanic Survival Prediction")
st.markdown("Enter the details of the passenger to predict survival")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.2, format="%.2f")
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
alone = st.selectbox("Traveling Alone", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict"):
    input_data = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked],
        "alone": [alone],
    })

    input_data = input_data.reindex(columns=expected_columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")

