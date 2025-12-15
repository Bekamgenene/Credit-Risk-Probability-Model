from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xverse.transformer import WoE
except ImportError:
    WoE = None

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------
ID_COL = "CustomerId"
DATETIME_COL = "TransactionDate"
AMOUNT_COL = "Amount"
TARGET = "FraudResult"

AGG_SUFFIXES = ["_sum", "_mean", "_count", "_std"]


# ---------------------------------------------------------------------------
# Custom transformers
# ---------------------------------------------------------------------------
class Aggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_col=None, amount_col=None):
        self.id_col = id_col or ID_COL
        self.amount_col = amount_col or AMOUNT_COL

    def fit(self, X, y=None):
        self._aggs_ = (
            X.groupby(self.id_col)[self.amount_col]
            .agg(["sum", "mean", "count", "std"])
            .rename(columns=lambda c: f"{self.amount_col}_{c}")
        )
        return self

    def transform(self, X):
        return X.merge(
            self._aggs_, left_on=self.id_col, right_index=True, how="left"
        )


class DatetimeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col=None):
        self.datetime_col = datetime_col or DATETIME_COL

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce")
        X = X.copy()
        X[f"{self.datetime_col}_hour"] = dt.dt.hour
        X[f"{self.datetime_col}_day"] = dt.dt.day
        X[f"{self.datetime_col}_month"] = dt.dt.month
        X[f"{self.datetime_col}_year"] = dt.dt.year
        return X


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    numeric_features = [
        f"{AMOUNT_COL}_sum",
        f"{AMOUNT_COL}_mean",
        f"{AMOUNT_COL}_count",
        f"{AMOUNT_COL}_std",
        f"{DATETIME_COL}_hour",
        f"{DATETIME_COL}_day",
        f"{DATETIME_COL}_month",
        f"{DATETIME_COL}_year",
    ]

    categorical_features = [
        "TransactionType",
        "Channel",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("datetime", DatetimeExtractor()),
            ("aggregate", Aggregator()),
            ("prep", preprocessor),
        ]
    )


# ---------------------------------------------------------------------------
# WoE helper (USED OUTSIDE PIPELINE)
# ---------------------------------------------------------------------------
def apply_woe(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    if WoE is None:
        raise ImportError("xverse is not installed. Cannot apply WoE.")

    woe = WoE(binning_method="quantile", n_bins=10)
    return woe.fit_transform(X, y)
