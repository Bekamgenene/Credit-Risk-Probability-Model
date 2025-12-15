"""
Proxy target engineering using RFM + K-Means.

Adds a binary column ``is_high_risk`` that flags the customer cluster with the
lowest engagement (low frequency & low monetary spend, high recency).
"""

from __future__ import annotations

from typing import Hashable, Literal

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

__all__ = ["add_rfm_target"]


def _compute_rfm(
    df: pd.DataFrame,
    *,
    id_col: Hashable,
    amount_col: Hashable,
    datetime_col: Hashable,
    snapshot_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute RFM metrics per customer."""
    rfm = (
        df.groupby(id_col)
        .agg(
            Recency=(datetime_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(datetime_col, "count"),
            Monetary=(amount_col, "sum"),
        )
        .astype("float64")
    )
    return rfm


def add_rfm_target(
    df: pd.DataFrame,
    *,
    id_col: str = "CustomerId",
    amount_col: str = "Amount",
    datetime_col: str = "TransactionStartTime",
    snapshot_date: pd.Timestamp | str | None = None,
    n_clusters: Literal[3] | int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return a copy of *df* with a new ``is_high_risk`` column.

    High-risk customers are identified via K-Means clustering on standardized
    RFM features. The cluster with the lowest Frequency & Monetary values and
    highest Recency is labeled as high risk.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    out = df.copy()

    # ---- Parse datetime safely ----
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    if out[datetime_col].isna().all():
        raise ValueError("datetime_col could not be parsed to datetime")

    if out[datetime_col].dt.tz is not None:
        out[datetime_col] = out[datetime_col].dt.tz_convert(None)

    # ---- Define snapshot date ----
    if snapshot_date is None:
        snapshot_date = out[datetime_col].max() + pd.Timedelta(days=1)
    snapshot_date = pd.to_datetime(snapshot_date)

    # ---- Compute RFM ----
    rfm = _compute_rfm(
        out,
        id_col=id_col,
        amount_col=amount_col,
        datetime_col=datetime_col,
        snapshot_date=snapshot_date,
    )

    # ---- Scale features ----
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # ---- K-Means clustering ----
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    rfm["cluster"] = km.fit_predict(rfm_scaled)

    # ---- Identify high-risk cluster (in original RFM units) ----
    centers_scaled = km.cluster_centers_
    centers = pd.DataFrame(
        scaler.inverse_transform(centers_scaled),
        columns=rfm.columns[:-1],
    )

    centers["_risk_score"] = (
        centers["Frequency"].rank(ascending=True)
        + centers["Monetary"].rank(ascending=True)
        + centers["Recency"].rank(ascending=False)
    )

    risk_cluster = centers["_risk_score"].idxmin()

    # ---- Assign binary target ----
    rfm["is_high_risk"] = (rfm["cluster"] == risk_cluster).astype("int8")

    # ---- Merge back into main dataset ----
    out = out.merge(
        rfm[["is_high_risk"]],
        left_on=id_col,
        right_index=True,
        how="left",
    )

    return out
