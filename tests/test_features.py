import pandas as pd
from src.features.rfm_target import add_rfm_target


def test_add_rfm_target_basic():
    # Sample input
    data = pd.DataFrame(
        {
            "CustomerId": [1, 2],
            "TransactionStartTime": pd.to_datetime(["2025-12-01", "2025-12-10"]),
            "Amount": [100, 200],
        }
    )

    # Run the RFM target function
    rfm_df = add_rfm_target(data)

    # Check that expected columns exist
    expected_cols = ["CustomerId", "Recency", "Frequency", "Monetary", "is_high_risk"]
    for col in expected_cols:
        assert col in rfm_df.columns, f"Missing column {col}"

    # Basic sanity checks
    assert all(rfm_df["Recency"] >= 0), "Recency should be non-negative"
    assert all(rfm_df["Frequency"] > 0), "Frequency should be positive"
    assert all(rfm_df["Monetary"] >= 0), "Monetary should be non-negative"
    assert rfm_df["is_high_risk"].isin([0, 1]).all(), "is_high_risk should be binary"
