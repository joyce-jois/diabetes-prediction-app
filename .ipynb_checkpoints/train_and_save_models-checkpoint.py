{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5905d0e2-d9f7-485d-85ac-fe24cf5aa323",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from xgboost import XGBClassifier\n",
    "\n",
    "# Load and preprocess data\n",
    "df = pd.read_csv(\"diabetes_prediction_dataset.csv\")\n",
    "df['gender_encoded'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})\n",
    "df['smoking_encoded'] = LabelEncoder().fit_transform(df['smoking_history'])\n",
    "df['bmi_category'] = df['bmi'].apply(lambda x: 'Underweight' if x < 18.5 else 'Normal' if x < 25 else 'Overweight' if x < 30 else 'Obese')\n",
    "df['bmi_cat_encoded'] = LabelEncoder().fit_transform(df['bmi_category'])\n",
    "df['glucose_bucket'] = df['blood_glucose_level'].apply(lambda x: 'Normal' if x < 100 else 'Prediabetes' if x < 126 else 'High')\n",
    "df['glucose_encoded'] = LabelEncoder().fit_transform(df['glucose_bucket'])\n",
    "\n",
    "scaler = StandardScaler()\n",
    "df['age_scaled'] = scaler.fit_transform(df[['age']])\n",
    "df['bmi_scaled'] = scaler.fit_transform(df[['bmi']])\n",
    "df['hba1c_scaled'] = scaler.fit_transform(df[['HbA1c_level']])\n",
    "df['glucose_scaled'] = scaler.fit_transform(df[['blood_glucose_level']])\n",
    "\n",
    "features = ['gender_encoded', 'smoking_encoded', 'bmi_cat_encoded', 'glucose_encoded',\n",
    "            'age_scaled', 'bmi_scaled', 'hba1c_scaled', 'glucose_scaled']\n",
    "X = df[features]\n",
    "y = df['diabetes']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Define and train models\n",
    "models = {\n",
    "    \"logreg.pkl\": LogisticRegression(max_iter=1000),\n",
    "    \"svm.pkl\": SVC(kernel='linear', probability=True),\n",
    "    \"tree.pkl\": DecisionTreeClassifier(),\n",
    "    \"rf.pkl\": RandomForestClassifier(),\n",
    "    \"xgb.pkl\": XGBClassifier(eval_metric='logloss')\n",
    "}\n",
    "\n",
    "for filename, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    joblib.dump(model, filename)\n",
    "\n",
    "# Save scalers too\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
