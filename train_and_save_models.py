import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load and preprocess data
df = pd.read_csv("diabetes_prediction_dataset.csv")
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

features = ['gender_encoded', 'smoking_encoded', 'bmi_cat_encoded', 'glucose_encoded',
            'age_scaled', 'bmi_scaled', 'hba1c_scaled', 'glucose_scaled']
X = df[features]
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train models
models = {
    "logreg.pkl": LogisticRegression(max_iter=1000),
    "svm.pkl": SVC(kernel='linear', probability=True),
    "tree.pkl": DecisionTreeClassifier(),
    "rf.pkl": RandomForestClassifier(),
    "xgb.pkl": XGBClassifier(eval_metric='logloss')
}

for filename, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, filename)

# Save scalers too
joblib.dump(scaler, "scaler.pkl")
