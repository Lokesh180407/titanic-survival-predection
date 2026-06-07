import streamlit as st
import pandas as pd
import joblib as jb

# Load saved files
model = jb.load("model.pkl")
scaler = jb.load("scaler.pkl")

st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict survival")

# User Inputs
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
    "Siblings/Spouses aboard",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Parents/Children aboard",
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
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

if st.button("Predict"):

    # Same encoding used during training

    sex_encoded = 1 if sex == "male" else 0

    embarked_map = {
        "C": 0,
        "Q": 1,
        "S": 2
    }

    embarked_encoded = embarked_map[embarked]

    input_data = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex_encoded],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "embarked": [embarked_encoded],
        "alone": [alone]
    })

    try:
        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)

        survival_prob = probability[0][1] * 100

        if prediction[0] == 1:
            st.success(
                f"✅ Passenger is likely to survive ({survival_prob:.2f}% chance)"
            )
        else:
            st.error(
                f"❌ Passenger is unlikely to survive ({100-survival_prob:.2f}% chance)"
            )

        st.write("### Input Data")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
