"""Small causal models used before foundation-model experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return coefficients, mean, scale


def _predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coefficients, mean, scale = model
    return np.column_stack([np.ones(len(x)), (x - mean) / scale]) @ coefficients


def walk_forward_ridge(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    alpha: float,
    use_onchain: bool,
) -> pd.DataFrame:
    """Predict each asset using only labels ending before current signal date."""
    selected_features = [c for c in feature_columns if use_onchain or not c.startswith("onchain_")]
    evaluation = config["evaluation"]
    train_start = pd.Timestamp(evaluation["train_start"])
    min_train_days = int(evaluation["min_train_days"])
    refit_sessions = int(evaluation["refit_sessions"])
    embargo = int(evaluation.get("embargo_sessions", 1))
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_dates = pd.DatetimeIndex(asset_rows["date"])
        model: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        last_fit_index = -refit_sessions
        for signal_index, signal_date in enumerate(signal_dates):
            current = asset_rows[asset_rows["date"].eq(signal_date)]
            if current.empty:
                continue
            if signal_index - last_fit_index >= refit_sessions or model is None:
                asset_position = asset_dates.get_loc(signal_date)
                cutoff_position = asset_position - embargo
                if cutoff_position <= 0:
                    continue
                label_cutoff = asset_dates[cutoff_position]
                train = asset_rows[
                    (asset_rows["date"] >= train_start)
                    & (asset_rows["date"] < signal_date)
                    & (asset_rows["label_end_date"] < label_cutoff)
                ].dropna(subset=[*selected_features, "label"])
                if train["date"].nunique() >= min_train_days:
                    x = train[selected_features].to_numpy(dtype=float)
                    y = train["label"].to_numpy(dtype=float)
                    model = _fit_ridge(x, y, alpha)
                    last_fit_index = signal_index
            if model is None:
                continue
            x_now = current[selected_features].to_numpy(dtype=float)
            if not np.isfinite(x_now).all():
                continue
            prediction = float(_predict(model, x_now)[0])
            rows.append({"date": signal_date, "asset": asset, "prediction": prediction, "positive": prediction > 0.0})
    return pd.DataFrame(rows)


def walk_forward_ar(
    features: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    lags: tuple[int, ...] = (1, 2, 3, 5, 10),
    horizon: int = 5,
) -> pd.DataFrame:
    """Causal autoregressive forecast of log returns, refit on expanding history."""
    evaluation = config["evaluation"]
    train_start = pd.Timestamp(evaluation["train_start"])
    min_train_days = int(evaluation["min_train_days"])
    refit_sessions = int(evaluation["refit_sessions"])
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date")
        returns = asset_rows.set_index("date")["ret_1"].dropna()
        model: tuple[np.ndarray, float] | None = None
        last_fit_index = -refit_sessions
        for signal_index, signal_date in enumerate(signal_dates):
            history = returns[returns.index <= signal_date]
            if signal_index - last_fit_index >= refit_sessions or model is None:
                history = history[history.index >= train_start]
                if len(history) >= min_train_days:
                    values = history.to_numpy(dtype=float)
                    max_lag = max(lags)
                    x = []
                    y = []
                    for i in range(max_lag, len(values)):
                        x.append([values[i - lag] for lag in lags])
                        y.append(values[i])
                    if len(y) >= 50:
                        design = np.column_stack([np.ones(len(x)), np.asarray(x)])
                        coef, *_ = np.linalg.lstsq(design, np.asarray(y), rcond=None)
                        model = (coef, float(values[-1]))
                        last_fit_index = signal_index
            if model is None:
                continue
            coef, last_value = model
            recent = list(history.to_numpy(dtype=float))
            if not recent:
                continue
            forecast: list[float] = []
            for _ in range(horizon):
                lag_values = [recent[-lag] if len(recent) >= lag else last_value for lag in lags]
                pred = float(np.r_[1.0, lag_values] @ coef)
                forecast.append(pred)
                recent.append(pred)
            rows.append({"date": signal_date, "asset": asset, "prediction": float(np.sum(forecast)), "positive": float(np.sum(forecast)) > 0.0})
    return pd.DataFrame(rows)
