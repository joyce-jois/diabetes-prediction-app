import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

st.title("🔍 Diabetes Prediction App with ML and SHAP")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    return df

df = load_data()
st.write("### Raw Dataset Preview")
st.dataframe(df.head())

# Feature Engineering
def preprocess_data(df):
    df['gender_encoded'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    df['smoking_encoded'] = LabelEncoder().fit_transform(df['smoking_history'])
    df['bmi_category'] = df['bmi'].apply(lambda x: 'Underweight' if x < 18.5 else 'Normal' if x < 25 else 'Overweight' if x < 30 else 'Obese')
    df['bmi_cat_encoded'] = LabelEncoder().fit_transform(df['bmi_category'])
    df['glucose_bucket'] = df['blood_glucose_level'].apply(lambda x: 'Normal' if x < 100 else 'Prediabetes' if x < 126 else 'High')
    df['glucose_encoded'] = LabelEncoder().fit_transform(df['glucose_bucket'])
    
    scaler = StandardScaler()
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
def train_model(model_name, X_train, y_train):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "SVM":
        model = SVC(kernel='linear', probability=True)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "XGBoost":
        model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

model_names = ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "XGBoost"]
model_metrics = []

for name in model_names:
    m = train_model(name, X_train, y_train)
    pred = m.predict(X_test)
    proba = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else [0]*len(pred)
    model_metrics.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1 Score': f1_score(y_test, pred),
        'ROC AUC': roc_auc_score(y_test, proba)
    })

metrics_df = pd.DataFrame(model_metrics)
st.write("### 📊 Model Comparison")
st.dataframe(metrics_df)

fig, ax = plt.subplots(figsize=(10, 5))
metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']].plot(kind='line', marker='o', ax=ax)
plt.title("Model Performance Line Plot")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(True)
st.pyplot(fig)

# Model selector and SHAP
model_choice = st.selectbox("Select a model for analysis & prediction", model_names)
model = train_model(model_choice, X_train, y_train)
y_pred = model.predict(X_test)

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
st.pyplot(fig_cm)

if model_choice in ["Random Forest", "XGBoost"]:
    st.write("### SHAP Feature Importance")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')

# Prediction form
st.write("### 🧪 Make a Prediction")
gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])
smoking = st.selectbox("Smoking History", df['smoking_history'].unique())
age = st.slider("Age", 1, 120, 45)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
hba1c = st.slider("HbA1c Level", 3.5, 10.0, 5.5)
glucose = st.slider("Blood Glucose Level", 50, 300, 120)

# Process input
def process_input():
    gender_val = {'Female': 0, 'Male': 1, 'Other': 2}[gender]
    smoking_val = LabelEncoder().fit(df['smoking_history']).transform([smoking])[0]
    bmi_cat = 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'
    bmi_cat_val = LabelEncoder().fit(df['bmi_category']).transform([bmi_cat])[0]
    glucose_bucket = 'Normal' if glucose < 100 else 'Prediabetes' if glucose < 126 else 'High'
    glucose_val = LabelEncoder().fit(df['glucose_bucket']).transform([glucose_bucket])[0]
    age_scaled = StandardScaler().fit(df[['age']]).transform([[age]])[0][0]
    bmi_scaled = StandardScaler().fit(df[['bmi']]).transform([[bmi]])[0][0]
    hba1c_scaled = StandardScaler().fit(df[['HbA1c_level']]).transform([[hba1c]])[0][0]
    glucose_scaled = StandardScaler().fit(df[['blood_glucose_level']]).transform([[glucose]])[0][0]
    return [[gender_val, smoking_val, bmi_cat_val, glucose_val, age_scaled, bmi_scaled, hba1c_scaled, glucose_scaled]]

if st.button("Predict"):
    input_data = process_input()
    pred = model.predict(input_data)[0]
    st.success("Prediction: " + ("Diabetic" if pred == 1 else "Not Diabetic"))

# Deployment instructions
with st.expander("🚀 Deployment Instructions for Streamlit Cloud"):
    st.markdown("""
    1. Create a GitHub repo and upload `app.py` and `diabetes_prediction_dataset.csv`
    2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
    3. Connect your GitHub repo and deploy
    4. Add `requirements.txt` with necessary packages:

    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    shap
    xgboost
    streamlit
    ```
    """)
