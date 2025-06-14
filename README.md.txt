# Diabetes Prediction with Machine Learning – Streamlit App

This Streamlit web application helps users analyze a diabetes prediction dataset, compare multiple ML models, and make live predictions based on patient inputs.

## Features

- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Model training and evaluation: 
  - Logistic Regression
  - SVM (Linear)
  - Decision Tree
  - Random Forest
  - XGBoost
- SHAP explainability (for XGBoost and Random Forest)
- Interactive model comparison charts
- Patient prediction form (real-time inference)

## Required Files

- `app.py` – Streamlit main script
- `diabetes_prediction_dataset.csv` – Input dataset
- `requirements.txt` – Python dependencies

## Deployment (Streamlit Cloud)

1. Fork or clone this repo.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and log in.
3. Click **“New App”** and link your GitHub repo.
4. Make sure the entry point is `app.py`.
5. Deploy!

## Try It Locally

```bash
pip install -r requirements.txt
streamlit run app.py