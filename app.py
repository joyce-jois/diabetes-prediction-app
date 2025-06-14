import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import shap
import warnings
import os

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Diabetes Prediction App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Overview", "📊 Model Comparison", "🔍 Model Evaluation", "🧪 Live Prediction", "🚀 Deployment Guide"])

# Load data and models
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

def load_models():
    models = {
        "Logistic Regression": joblib.load("logreg.pkl"),
        "SVM": joblib.load("svm.pkl"),
        "Decision Tree": joblib.load("tree.pkl"),
        "Random Forest": joblib.load("rf.pkl"),
        "XGBoost": joblib.load("xgb.pkl")
    }
    scaler = joblib.load("scaler.pkl")
    return models, scaler

df = load_data()
models, scaler = load_models()

# Preprocess data
def preprocess_data(df):
    df['gender_encoded'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    df['smoking_encoded'] = LabelEncoder().fit_transform(df['smoking_history'])
    df['bmi_category'] = df['bmi'].apply(lambda x: 'Underweight' if x < 18.5 else 'Normal' if x < 25 else 'Overweight' if x < 30 else 'Obese')
    df['bmi_cat_encoded'] = LabelEncoder().fit_transform(df['bmi_category'])
    df['glucose_bucket'] = df['blood_glucose_level'].apply(lambda x: 'Normal' if x < 100 else 'Prediabetes' if x < 126 else 'High')
    df['glucose_encoded'] = LabelEncoder().fit_transform(df['glucose_bucket'])
    df['age_scaled'] = scaler.fit_transform(df[['age']])
    df['bmi_scaled'] = scaler.fit_transform(df[['bmi']])
    df['hba1c_scaled'] = scaler.fit_transform(df[['HbA1c_level']])
    df['glucose_scaled'] = scaler.fit_transform(df[['blood_glucose_level']])
    return df

df = preprocess_data(df)

features = ['gender_encoded', 'smoking_encoded', 'bmi_cat_encoded', 'glucose_encoded',
            'age_scaled', 'bmi_scaled', 'hba1c_scaled', 'glucose_scaled']
X = df[features]
y = df['diabetes']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if page == "🏠 Overview":
    st.title("🏠 Dataset Overview")
    st.dataframe(df.head())

elif page == "📊 Model Comparison":
    st.title("📊 Compare Model Performance")
    performance = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else [0]*len(y_pred)
        performance.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, y_proba)
        })
    perf_df = pd.DataFrame(performance)
    st.dataframe(perf_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    perf_df.set_index("Model").plot(kind='barh', ax=ax)
    plt.title("Horizontal Bar Chart of Model Scores")
    st.pyplot(fig)

elif page == "🔍 Model Evaluation":
    st.title("🔍 Evaluate Model + SHAP")
    model_choice = st.selectbox("Choose model to evaluate", list(models.keys()))
    model = models[model_choice]
    y_pred = model.predict(X_test)
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.subheader("Confusion Matrix")
    st.pyplot(fig)

    if model_choice in ["Random Forest", "XGBoost"]:
        st.subheader("SHAP Feature Importance")
        # Explain model predictions
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
         # Rename columns for clearer SHAP labels
        X_test_renamed = X_test.copy()
        X_test_renamed.columns = [
            "Gender",
            "Smoking History",
            "BMI Category",
            "Glucose Category",
            "Age (Standardized)",
            "BMI (Standardized)",
            "HbA1c (Standardized)",
            "Glucose (Standardized)"
        ]
        # Bar plot
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        fig_bar = plt.gcf()  # Get current figure
        st.pyplot(fig_bar)
        # Dot plot
        shap.summary_plot(shap_values, X_test, show=False)
        fig_dot = plt.gcf()
        st.pyplot(fig_dot)
   
elif page == "🧪 Live Prediction":
    st.title("🧪 Live Patient Prediction")
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])
        smoking = st.selectbox("Smoking History", df['smoking_history'].unique())
        age = st.slider("Age", 1, 120, 45)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        hba1c = st.slider("HbA1c Level", 3.5, 10.0, 5.5)
        glucose = st.slider("Blood Glucose Level", 50, 300, 120)
        model_choice = st.selectbox("Model", list(models.keys()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        gender_val = {'Female': 0, 'Male': 1, 'Other': 2}[gender]
        smoking_val = LabelEncoder().fit(df['smoking_history']).transform([smoking])[0]
        bmi_cat = 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'
        bmi_cat_val = LabelEncoder().fit(df['bmi_category']).transform([bmi_cat])[0]
        glucose_bucket = 'Normal' if glucose < 100 else 'Prediabetes' if glucose < 126 else 'High'
        glucose_val = LabelEncoder().fit(df['glucose_bucket']).transform([glucose_bucket])[0]
        age_scaled = scaler.transform([[age]])[0][0]
        bmi_scaled = scaler.transform([[bmi]])[0][0]
        hba1c_scaled = scaler.transform([[hba1c]])[0][0]
        glucose_scaled = scaler.transform([[glucose]])[0][0]

        features_input = [[gender_val, smoking_val, bmi_cat_val, glucose_val,
                           age_scaled, bmi_scaled, hba1c_scaled, glucose_scaled]]
        model = models[model_choice]
        pred = model.predict(features_input)[0]

        result_df = pd.DataFrame({
            'Gender': [gender], 'Smoking History': [smoking], 'Age': [age], 'BMI': [bmi],
            'HbA1c': [hba1c], 'Glucose': [glucose], 'Prediction': ["Diabetic" if pred==1 else "Not Diabetic"]
        })
        st.success(f"Prediction: {'Diabetic' if pred==1 else 'Not Diabetic'}")
        st.dataframe(result_df)

        # Export result
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result as CSV", data=csv, file_name="patient_prediction.csv", mime='text/csv')

elif page == "🚀 Deployment Guide":
    st.title("🚀 Streamlit Cloud Deployment Instructions")
    st.markdown("""
    1. **Create a GitHub repo** and upload `app.py`, `diabetes_prediction_dataset.csv`, all `.pkl` files.
    2. **Create `requirements.txt`** with:
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    shap
    xgboost
    streamlit
    joblib
    ```
    3. Go to [Streamlit Cloud](https://streamlit.io/cloud), link GitHub, and deploy your app.
    """)
