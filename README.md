# Loan Default Prediction System

## Overview
This project builds a machine learning system that predicts whether a borrower is likely to **default on a loan** based on financial and loan-related features.

The project follows a complete machine learning pipeline including **data analysis, preprocessing, model training, evaluation, and deployment through a Streamlit web application**.

The final deployed model uses **XGBoost**, selected after comparing multiple algorithms.

---
## Deployed Model Link

https://loan-default-ai-prediction.streamlit.app/

---
## Problem Statement
Financial institutions face significant risk when borrowers default on loans.  
The goal of this project is to develop a predictive model that can help identify **high-risk borrowers early**, allowing lenders to make better credit decisions.

---

## Dataset
The dataset contains borrower and loan information such as:

- Credit Score
- Income
- Loan Amount
- Loan Term
- Debt-to-Income Ratio
- Loan Purpose
- Loan Limit Type

Target variable:

**Status**
- `0` → Non-default  
- `1` → Default

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
Performed analysis to understand:

- Feature distributions
- Missing values
- Relationships between financial attributes and default risk

Key insights were discovered using statistical analysis and visualizations.

---

### 2. Data Preprocessing

Steps performed:

- Handling missing values using **SimpleImputer**
- Encoding categorical variables using **One-Hot Encoding**
- Feature scaling for numerical variables
- Train-test split to prevent data leakage

---

### 3. Model Training

Multiple models were trained and compared:

| Model | Purpose |
|------|------|
| Logistic Regression | Baseline model |
| Random Forest | Tree-based ensemble |
| XGBoost | Gradient boosting model |

---


### 4. Model Evaluation

Models were evaluated using several classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

#### Model Comparison

| Model | Accuracy | Precision (Default) | Recall (Default) | F1 Score |
|------|------|------|------|------|
| Logistic Regression | 0.57 | 0.30 | 0.56 | 0.39 |
| Random Forest | 0.84 | 0.72 | 0.61 | 0.66 |
| XGBoost | 0.78 | 0.53 | **0.71** | 0.61 |

**XGBoost was selected as the final model** because it achieved the highest recall for the default class, meaning it detects more risky borrowers.

---

## Feature Importance

The most influential factors affecting loan default prediction were:

- Debt-to-Income Ratio (DTI)
- Income
- Loan Amount
- Loan Purpose

This aligns with financial risk theory where borrowers with higher financial burden are more likely to default.

---

## Deployment

The trained model was deployed using **Streamlit** to create an interactive web application.

Users can enter borrower information and receive:

- Default prediction
- Default probability
- Risk assessment indicator

---

## Project Structure

```text
.
├── data/                 # Dataset files
│   └── Loan_Default.csv
├── models/               # Trained model artifacts and preprocessing files
│   ├── model_pipeline.pkl
│   ├── xgboost_model.pkl
│   ├── model_features.pkl
│   └── scaler.pkl
├── notebooks/            # EDA and model experimentation notebooks
│   ├── EDA_loan_dataset.ipynb
│   └── model_train.ipynb
├── feature_engineering.py
├── loan_app.py           # Streamlit application
├── train_pipeline.py     # Model training pipeline
├── requirements.txt
└── README.md
```

---

## Running the Application

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Streamlit

```bash
streamlit run loan_app.py
```

### Retrain the model

```bash
python train_pipeline.py
```

## Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

## Key Learnings

This project demonstrates:

- Building an end-to-end machine learning pipeline  
- Handling class imbalance in classification problems  
- Evaluating models using appropriate metrics  
- Deploying ML models with an interactive web interface  

---

## Future Improvements

Possible enhancements include:

- Hyperparameter tuning using GridSearchCV  
- Adding ROC-AUC analysis  
- Improving the Streamlit UI with advanced visualizations  
- Deploying the application to a cloud platform  

---

## Author

**Rishabh Devadiga**

Computer Science Engineering (AI & Data Science)

This project was built as part of my learning journey in machine learning and data science.
