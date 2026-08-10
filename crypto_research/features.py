"""Causal feature and forward-label construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_20",
    "vol_60",
    "range_1",
    "volume_z_20",
]


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 4)).mean()
    std = series.rolling(window, min_periods=max(5, window // 4)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _asset_features(
    market: pd.DataFrame,
    onchain: pd.DataFrame,
    asset: str,
    horizon: int,
    round_trip_cost: float,
    onchain_ffill_limit: int,
) -> pd.DataFrame:
    price = market[market["asset"].eq(asset)].sort_values("date").copy()
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    price["log_close"] = np.log(price["close"])
    returns = price["log_close"].diff()
    price["ret_1"] = returns
    for days in [3, 5, 10, 20, 60]:
        price[f"ret_{days}"] = price["log_close"].diff(days)
    # 12-1 momentum: trailing twelve months, excluding most recent month.
    price["ret_252_skip_21"] = price["log_close"].shift(21) - price["log_close"].shift(252)
    price["vol_20"] = returns.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252.0)
    price["vol_60"] = returns.rolling(60, min_periods=20).std(ddof=0) * np.sqrt(252.0)
    price["range_1"] = np.log(price["high"] / price["low"])
    price["volume_z_20"] = _zscore(np.log1p(price["volume"]), 20)

    chain = onchain[onchain["asset"].eq(asset)].sort_values("date").copy()
    chain["date"] = pd.to_datetime(chain["date"]).dt.normalize()
    chain["available_date"] = chain["date"] + pd.Timedelta(days=1)
    metric_columns = [column for column in chain.columns if column not in {"date", "asset", "available_date"}]
    if metric_columns:
        right = chain[["available_date", *metric_columns]].rename(columns={"available_date": "date"})
        merged = pd.merge_asof(price.sort_values("date"), right.sort_values("date"), on="date", direction="backward")
        for metric in metric_columns:
            merged[metric] = merged[metric].ffill(limit=onchain_ffill_limit)
            transformed = np.log1p(merged[metric].clip(lower=0.0))
            if transformed.notna().sum() == 0:
                # Some metrics are asset-specific. Keep schema stable without
                # inventing observations for assets lacking that metric.
                merged[f"onchain_{metric.lower()}_z"] = 0.0
            else:
                merged[f"onchain_{metric.lower()}_z"] = _zscore(transformed, 60)
        price = merged

    price["label_end_date"] = price["date"].shift(-horizon)
    price["label"] = price["close"].shift(-horizon) / price["close"] - 1.0 - round_trip_cost
    price["asset"] = asset
    return price


def build_feature_table(
    market: pd.DataFrame,
    onchain: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Build tidy features; every feature row is measurable at its date close."""
    backtest = config["backtest"]
    evaluation = config["evaluation"]
    round_trip_cost = 2.0 * (float(backtest["fee_bps"]) + float(backtest["slippage_bps"])) / 10_000.0
    horizon = int(config.get("labels", {}).get("horizon", 5))
    frames = [
        _asset_features(
            market,
            onchain,
            asset,
            horizon,
            round_trip_cost,
            int(config["data"].get("onchain_ffill_limit", 3)),
        )
        for asset in config["data"]["assets"]
    ]
    features = pd.concat(frames, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    onchain_features = sorted(column for column in features.columns if column.startswith("onchain_") and column.endswith("_z"))
    feature_columns = BASE_FEATURES + onchain_features
    missing = [column for column in BASE_FEATURES if column not in features]
    if missing:
        raise ValueError(f"Feature construction missing columns: {missing}")
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["label_end_date"] = pd.to_datetime(features["label_end_date"])
    features.attrs["feature_columns"] = feature_columns
    features.attrs["label_horizon"] = horizon
    features.attrs["embargo_sessions"] = int(evaluation.get("embargo_sessions", 1))
    return features, feature_columns


def assert_causal_features(features: pd.DataFrame, feature_columns: list[str]) -> None:
    """Reject malformed rows that could make walk-forward checks meaningless."""
    if features.empty:
        raise ValueError("No feature rows")
    if features.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate feature date/asset rows")
    if not features["date"].is_monotonic_increasing:
        raise ValueError("Feature table is not date sorted")
    # NaNs are expected during rolling warm-up; infinities are not.
    if np.isinf(features[feature_columns].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature value")
