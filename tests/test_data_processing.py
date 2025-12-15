from pathlib import Path

import pandas as pd
from src.data_processing import engineer_features
from src.utils.model_utils import compute_metrics


def test_engineer_features(tmp_path: Path):
    # Create sample dataframe
    df = pd.DataFrame({
        "FraudResult": [0, 1],
        "Amount": [100.0, 200.0],
    })

    X, y = engineer_features(df)

    assert "Amount" in X.columns
    assert len(X) == len(y) == 2


def test_compute_metrics():
    y_true = [0, 1, 1, 0]
    y_prob = [0.1, 0.8, 0.6, 0.4]
    metrics = compute_metrics(y_true, y_prob)
    # basic sanity checks
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert expected_keys.issubset(metrics.keys())
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
