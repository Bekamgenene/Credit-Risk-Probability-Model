from pathlib import Path

import pandas as pd
from src.data_processing import engineer_features
from src.utils.model_utils import compute_metrics


def test_engineer_features(tmp_path: Path):
    """Test feature engineering produces expected output."""
    # Create a sample dataframe
    df = pd.DataFrame(
        {
            "FraudResult": [0, 1],
            "Amount": [100.0, 200.0],
        }
    )

    X, y = engineer_features(df)

    # Check that the expected column exists
    assert "Amount" in X.columns, "Amount column should be in features"
    # Check lengths match
    assert len(X) == len(y) == 2, "Feature and target lengths should match"


def test_compute_metrics():
    """Test compute_metrics returns metrics in [0,1] with correct keys."""
    y_true = [0, 1, 1, 0]
    y_prob = [0.1, 0.8, 0.6, 0.4]

    metrics = compute_metrics(y_true, y_prob)

    # Check that expected keys exist
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert expected_keys.issubset(
        metrics.keys()
    ), "Missing expected metric keys"

    # Check all metric values are between 0 and 1
    for v in metrics.values():
        assert 0.0 <= v <= 1.0, "Metric values should be between 0 and 1"
