import pandas as pd
from src.features.rfm_target import create_rfm_features

def test_rfm_feature_creation():
    # sample input
    data = pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "InvoiceDate": pd.to_datetime(["2025-12-01", "2025-12-10"]),
            "Amount": [100, 200],
        }
    )

    rfm = create_rfm_features(data)

    # Check that expected columns exist
    expected_cols = ["CustomerID", "Recency", "Frequency", "Monetary"]
    for col in expected_cols:
        assert col in rfm.columns, f"Missing column {col}"

    # Basic sanity checks
    assert all(rfm["Recency"] >= 0), "Recency should be non-negative"
    assert all(rfm["Frequency"] > 0), "Frequency should be positive"
    assert all(rfm["Monetary"] >= 0), "Monetary should be non-negative"
