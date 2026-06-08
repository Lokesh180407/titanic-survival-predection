import streamlit as st
import pandas as pd
import joblib as jb

# Load model files
model = jb.load("model.pkl")
scaler = jb.load("scaler.pkl")
expected_columns = jb.load("columns.pkl")

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details and predict survival chances.")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

sibsp = st.number_input(
    "Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.20,
    format="%.2f"
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["C", "Q", "S"]
)

alone = st.selectbox(
    "Traveling Alone",
    [True, False]
)

# Prediction
if st.button("Predict Survival"):

    try:

        # Match training data
        input_data = pd.DataFrame({
            "pclass": [pclass],
            "sex": [sex],
            "age": [age],
            "sibsp": [sibsp],
            "parch": [parch],
            "fare": [fare],
            "embarked": [embarked],
            "alone": [alone]
        })

        # Same preprocessing used during training
        input_data["sex"] = input_data["sex"].map({
            "female": 0,
            "male": 1
        })

        input_data["alone"] = input_data["alone"].astype(int)

        input_data["embarked"] = input_data["embarked"].map({
            "C": 0,
            "Q": 1,
            "S": 2
        })

        # Same as training
        input_data = pd.get_dummies(input_data)

        input_data = input_data.reindex(
            columns=expected_columns,
            fill_value=0
        )

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success("✅ Passenger is likely to survive.")
        else:
            st.error("❌ Passenger is unlikely to survive.")

        # Show processed data
        with st.expander("Processed Input"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
