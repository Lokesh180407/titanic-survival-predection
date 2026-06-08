# titanic-survival-predection
# 🚢 Titanic Survival Prediction

accuracy=80%
f1_score=81%
made with svm algorithm
A Machine Learning web application built using Streamlit that predicts whether a passenger would survive the Titanic disaster based on passenger information.
see project:https://titanic-survival-predection-1.onrender.com
## 📌 Project Overview

This project uses a trained Machine Learning classification model to predict passenger survival on the Titanic dataset. Users can enter passenger details through a simple web interface and receive an instant prediction.

## 🎯 Features

* Interactive Streamlit Web Application
* Real-time Survival Prediction
* User-friendly Interface
* Machine Learning Classification Model
* Data Preprocessing and Feature Scaling
* Probability Score Display

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib

## 📂 Project Structure

```text
Titanic-Survival-Prediction/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
└── dataset.csv
```

## 📊 Input Features

The model uses the following passenger information:

* Passenger Class (Pclass)
* Gender (Sex)
* Age
* Number of Siblings/Spouses Aboard (SibSp)
* Number of Parents/Children Aboard (Parch)
* Fare
* Port of Embarkation
* Traveling Alone

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Encoding
6. Feature Scaling
7. Model Training
8. Model Evaluation
9. Deployment using Streamlit

## 📈 Model Evaluation Metrics

The model was evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix

## 🚀 Running Locally

Clone the repository:

```bash
git clone https://github.com/your-username/Titanic-Survival-Prediction.git
```

Navigate to the project folder:

```bash
cd Titanic-Survival-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🌐 Deployment

The application can be deployed using:

* Streamlit Community Cloud
* Hugging Face Spaces
* Render

## 📸 Application Preview

The user enters passenger information and clicks the Predict button. The application displays whether the passenger is likely to survive along with the prediction probability.

## 📚 Dataset

This project uses the famous Titanic dataset containing passenger information from the Titanic disaster.

## 👨‍💻 Author

**Lokesh Reddy**

B.Tech CSE (AI & ML)

Machine Learning Enthusiast

## ⭐ Future Improvements

* Compare multiple classification algorithms
* Hyperparameter tuning
* Advanced feature engineering
* Model explainability using SHAP
* Improved UI/UX
