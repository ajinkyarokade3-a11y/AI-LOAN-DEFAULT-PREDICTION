from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from feature_engineering import LoanFeatureEngineer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Loan_Default.csv"
MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"
TARGET_COL = "Status"


def build_pipeline() -> Pipeline:
    numeric_cols = [
        "Credit_Score",
        "income",
        "loan_amount",
        "term",
        "dtir1",
        "loan_to_income",
        "emi",
        "emi_to_income",
        "high_loan_flag",
    ]
    categorical_cols = ["loan_purpose", "loan_limit"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", LoanFeatureEngineer()),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    return pipeline


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    feature_cols = [
        "Credit_Score",
        "income",
        "loan_amount",
        "term",
        "dtir1",
        "loan_purpose",
        "loan_limit",
    ]

    df = df[feature_cols + [TARGET_COL]].copy()

    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print("Classification report:")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
