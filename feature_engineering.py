from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds engineered loan features to the input dataframe.
    This transformer is designed to be used inside an sklearn Pipeline.
    """

    income_col: str = "income"
    loan_amount_col: str = "loan_amount"
    term_col: str = "term"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        income = pd.to_numeric(X.get(self.income_col), errors="coerce")
        loan_amount = pd.to_numeric(X.get(self.loan_amount_col), errors="coerce")
        term = pd.to_numeric(X.get(self.term_col), errors="coerce")

        with np.errstate(divide="ignore", invalid="ignore"):
            loan_to_income = np.where(income > 0, loan_amount / income, np.nan)
            emi = np.where(term > 0, loan_amount / term, np.nan)
            emi_to_income = np.where(income > 0, emi / income, np.nan)

        X["loan_to_income"] = loan_to_income
        X["emi"] = emi
        X["emi_to_income"] = emi_to_income
        X["high_loan_flag"] = np.where(loan_to_income > 5, 1, 0)

        return X
