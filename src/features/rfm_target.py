from __future__ import annotations
from typing import Hashable

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
    """Return an RFM frame indexed by ``id_col``."""
    rfm = df.groupby(id_col).agg(
        Recency=(datetime_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(datetime_col, "count"),
        Monetary=(amount_col, "sum"),
    )
    return rfm.astype("float64")


def add_rfm_target(
    df: pd.DataFrame,
    *,
    id_col: str = "CustomerId",
    amount_col: str = "Amount",
    datetime_col: str = "TransactionStartTime",
    snapshot_date: pd.Timestamp | str | None = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    out = df.copy()

    # parse datetimes safely
    out[datetime_col] = pd.to_datetime(out[datetime_col], errors="coerce")
    if out[datetime_col].dt.tz is not None:
        out[datetime_col] = out[datetime_col].dt.tz_convert(None)
    if out[datetime_col].isna().all():
        raise ValueError("datetime_col could not be parsed to datetime")

    # default snapshot date
    if snapshot_date is None:
        snapshot_date = out[datetime_col].max() + pd.Timedelta(days=1)
    snapshot_date = pd.to_datetime(snapshot_date)

    # compute RFM
    rfm = _compute_rfm(
        out,
        id_col=id_col,
        amount_col=amount_col,
        datetime_col=datetime_col,
        snapshot_date=snapshot_date,
    )

    # adjust clusters if fewer samples than requested clusters
    n_clusters = min(n_clusters, len(rfm))

    # scale & cluster
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    km = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init="auto"
    )
    rfm["cluster"] = km.fit_predict(rfm_scaled)

    # pick high-risk cluster: lowest Frequency + Monetary, highest Recency
    centers = pd.DataFrame(km.cluster_centers_, columns=rfm.columns[:-1])
    centers["_score"] = (
        centers["Frequency"].rank(method="average")
        + centers["Monetary"].rank(method="average")
        - centers["Recency"].rank(method="average")
    )
    risk_cluster = centers["_score"].idxmin()

    rfm["is_high_risk"] = (rfm["cluster"] == risk_cluster).astype("int8")

    # merge RFM metrics and risk flag back to original df
    out = out.merge(
        rfm[["Recency", "Frequency", "Monetary", "is_high_risk"]],
        left_on=id_col,
        right_index=True,
        how="left",
    )
    return out
