"""Stage 0 rules and model-to-portfolio mappings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import (
    walk_forward_ar,
    walk_forward_kronos,
    walk_forward_kronos_finetune,
    walk_forward_kronos_finetune_static,
    walk_forward_lightgbm,
    walk_forward_lstm,
    walk_forward_lstm_gate,
    walk_forward_lstm_gate_norm,
    walk_forward_ridge,
    walk_forward_timesfm_finetune,
    walk_forward_timesfm,
    walk_forward_timesfm_ridge,
    walk_forward_tcn,
    walk_forward_transformer,
    walk_forward_xgboost,
)


ASSETS = ("BTC", "ETH")
FIXED_SIGNAL_THRESHOLD = 0.0


def _market_wide(market: pd.DataFrame) -> pd.DataFrame:
    wide = market.pivot(index="date", columns="asset", values=["open", "high", "low", "close", "volume"])
    wide.columns = [f"{asset}_{field}" for field, asset in wide.columns]
    return wide.sort_index()


def _equal_positive(predictions: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=[*ASSETS])
    for date, group in predictions.groupby("date"):
        chosen = group[group["prediction"] > threshold].sort_values("prediction", ascending=False)
        weights = {asset: (1.0 / len(chosen) if asset in set(chosen["asset"]) else 0.0) for asset in ASSETS}
        rows.append({"date": date, **weights})
    return pd.DataFrame(rows).set_index("date").sort_index()


def _covariance_ridge_targets(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    risk_aversion: float,
    turnover_penalty: float,
) -> pd.DataFrame:
    """Map Ridge forecasts to long-only weights using trailing covariance."""
    if predictions.empty:
        return pd.DataFrame(columns=[*ASSETS])
    returns = (
        features.pivot(index="date", columns="asset", values="ret_1")
        .reindex(columns=list(ASSETS))
        .sort_index()
    )
    previous = np.zeros(len(ASSETS), dtype=float)
    rows: list[dict[str, Any]] = []
    for date, group in predictions.groupby("date"):
        values = group.set_index("asset")["prediction"].reindex(ASSETS)
        if values.isna().any():
            continue
        history = returns.loc[returns.index <= pd.Timestamp(date)].tail(60).dropna()
        if len(history) < 20:
            covariance = np.eye(len(ASSETS), dtype=float) * 1e-4
        else:
            covariance = history.cov().to_numpy(dtype=float)
            covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
            covariance += np.eye(len(ASSETS), dtype=float) * 1e-8
        forecast = values.to_numpy(dtype=float)
        best_weights = np.zeros(len(ASSETS), dtype=float)
        best_value = -np.inf
        for btc_ticks in range(11):
            for eth_ticks in range(11 - btc_ticks):
                weights = np.array([btc_ticks, eth_ticks], dtype=float) / 10.0
                objective = (
                    float(weights @ forecast)
                    - float(risk_aversion) * float(weights @ covariance @ weights)
                    - float(turnover_penalty) * float(np.abs(weights - previous).sum())
                )
                if objective > best_value:
                    best_value = objective
                    best_weights = weights
        previous = best_weights
        rows.append({"date": date, **dict(zip(ASSETS, best_weights))})
    return pd.DataFrame(rows).set_index("date").sort_index()


def _blend_predictions(
    ridge: pd.DataFrame,
    ar: pd.DataFrame,
    ridge_weight: float,
) -> pd.DataFrame:
    if ridge.empty or ar.empty:
        return pd.DataFrame(columns=["date", "asset", "prediction", "positive"])
    left = ridge[["date", "asset", "prediction"]].rename(columns={"prediction": "ridge_prediction"})
    right = ar[["date", "asset", "prediction"]].rename(columns={"prediction": "ar_prediction"})
    merged = left.merge(right, on=["date", "asset"], how="inner")
    merged["prediction"] = (
        float(ridge_weight) * merged["ridge_prediction"]
        + (1.0 - float(ridge_weight)) * merged["ar_prediction"]
    )
    merged["positive"] = merged["prediction"] > 0.0
    return merged[["date", "asset", "prediction", "positive"]]


def _blend_ridge_lstm(
    ridge: pd.DataFrame,
    lstm: pd.DataFrame,
    ridge_weight: float,
) -> pd.DataFrame:
    if ridge.empty or lstm.empty:
        return pd.DataFrame(columns=["date", "asset", "prediction", "positive"])
    left = ridge[["date", "asset", "prediction"]].rename(columns={"prediction": "ridge_prediction"})
    right = lstm[["date", "asset", "prediction"]].rename(columns={"prediction": "lstm_prediction"})
    merged = left.merge(right, on=["date", "asset"], how="inner")
    merged["prediction"] = (
        float(ridge_weight) * merged["ridge_prediction"]
        + (1.0 - float(ridge_weight)) * merged["lstm_prediction"]
    )
    merged["positive"] = merged["prediction"] > 0.0
    return merged[["date", "asset", "prediction", "positive"]]


def _residual_gate(predictions: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    if predictions.empty or "uncertainty" not in predictions:
        return predictions
    gated = predictions.copy()
    gated["prediction"] = gated["prediction"] - float(multiplier) * gated["uncertainty"]
    gated["positive"] = gated["prediction"] > 0.0
    return gated


def _residual_size_targets(
    predictions: pd.DataFrame,
    multiplier: float,
    threshold: float,
) -> pd.DataFrame:
    if predictions.empty or "uncertainty" not in predictions:
        return pd.DataFrame(columns=[*ASSETS])
    scored = predictions.copy()
    scored["score"] = scored["prediction"] - float(multiplier) * scored["uncertainty"]
    rows: list[dict[str, Any]] = []
    for date, group in scored.groupby("date"):
        positive = group[group["score"] > float(threshold)]
        total = float(positive["score"].sum())
        weights = {
            asset: (float(positive.loc[positive["asset"].eq(asset), "score"].sum()) / total if total else 0.0)
            for asset in ASSETS
        }
        rows.append({"date": date, **weights})
    return pd.DataFrame(rows).set_index("date").sort_index()


def _apply_uncertainty_gate(predictions: pd.DataFrame, maximum: float) -> pd.DataFrame:
    if predictions.empty or maximum <= 0.0 or "uncertainty" not in predictions:
        return predictions
    gated = predictions.copy()
    mask = gated["uncertainty"] > maximum
    gated.loc[mask, "prediction"] = -gated.loc[mask, "prediction"].abs() - 1e-9
    return gated


def _inverse_vol_targets(features: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date in dates:
        current = features[features["date"].eq(date)]
        chosen = current[current["ret_20"] > 0.0].dropna(subset=["vol_20"])
        inverse = {row.asset: 1.0 / max(float(row.vol_20), 1e-6) for row in chosen.itertuples()}
        total = sum(inverse.values())
        rows.append({"date": date, **{asset: (inverse.get(asset, 0.0) / total if total else 0.0) for asset in ASSETS}})
    return pd.DataFrame(rows).set_index("date")


def _shift_to_next_open(raw_targets: pd.DataFrame, market_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if raw_targets.empty:
        return pd.DataFrame(columns=[*ASSETS])
    raw_targets = raw_targets.reindex(columns=list(ASSETS), fill_value=0.0).fillna(0.0).sort_index()
    rows: list[dict[str, Any]] = []
    for date, values in raw_targets.iterrows():
        positions = market_dates.searchsorted(pd.Timestamp(date), side="right")
        if positions >= len(market_dates):
            continue
        execution_date = market_dates[positions]
        row = {asset: float(values.get(asset, 0.0)) for asset in ASSETS}
        row["date"] = execution_date
        row["signal_date"] = pd.Timestamp(date)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*ASSETS])
    result = pd.DataFrame(rows).set_index("date").sort_index()
    # Do not rebalance when target weights are unchanged.
    weights = result[list(ASSETS)].round(12)
    changed = weights.ne(weights.shift(1)).any(axis=1)
    return result.loc[changed]


def build_targets(
    market: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    mode: str | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return execution-date targets and signal-date audit rows."""
    mode = mode or str(config["strategy"].get("mode", "stage0"))
    params = params or {}
    wide = _market_wide(market)
    dates = pd.DatetimeIndex(wide.index)
    if mode == "stage0":
        mode = "buy_hold_50_50"
    if mode == "cash":
        raw = pd.DataFrame([{ "date": dates[0], "BTC": 0.0, "ETH": 0.0 }]).set_index("date")
    elif mode == "buy_hold_btc":
        raw = pd.DataFrame([{ "date": dates[0], "BTC": 1.0, "ETH": 0.0 }]).set_index("date")
    elif mode == "buy_hold_eth":
        raw = pd.DataFrame([{ "date": dates[0], "BTC": 0.0, "ETH": 1.0 }]).set_index("date")
    elif mode == "buy_hold_50_50":
        raw = pd.DataFrame([{ "date": dates[0], "BTC": 0.5, "ETH": 0.5 }]).set_index("date")
    elif mode == "sma_cross":
        rows = []
        for asset in ASSETS:
            series = wide[f"{asset}_close"]
            fast = series.rolling(int(params.get("fast", 20)), min_periods=10).mean()
            slow = series.rolling(int(params.get("slow", 100)), min_periods=30).mean()
            for date in dates:
                if pd.notna(fast.get(date)) and pd.notna(slow.get(date)):
                    rows.append({"date": date, "asset": asset, "positive": bool(series.loc[date] > fast.loc[date] and fast.loc[date] > slow.loc[date])})
        signals = pd.DataFrame(rows)
        raw_rows = []
        for date, group in signals.groupby("date"):
            chosen = set(group[group["positive"]]["asset"])
            raw_rows.append({"date": date, **{asset: (1.0 / len(chosen) if asset in chosen else 0.0) for asset in ASSETS}})
        raw = pd.DataFrame(raw_rows).set_index("date") if raw_rows else pd.DataFrame(columns=[*ASSETS])
    elif mode == "vol_scaled":
        raw = _inverse_vol_targets(features, dates)
    elif mode == "momentum_12_1":
        rows = []
        for date in dates:
            current = features[features["date"].eq(date)]
            chosen = set(current[current["ret_252_skip_21"] > 0.0]["asset"]) if "ret_252_skip_21" in current else set()
            rows.append({"date": date, **{asset: (1.0 / len(chosen) if asset in chosen else 0.0) for asset in ASSETS}})
        raw = pd.DataFrame(rows).set_index("date")
    elif mode == "ridge":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "lstm":
        predictions = walk_forward_lstm(
            features,
            feature_columns,
            dates,
            config,
            hidden_size=int(params.get("hidden_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0001)),
            epochs=int(params.get("epochs", 4)),
            context=int(params.get("context", 32)),
            use_onchain=bool(params.get("use_onchain", False)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "tcn":
        predictions = walk_forward_tcn(
            features,
            feature_columns,
            dates,
            config,
            hidden_size=int(params.get("hidden_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0001)),
            epochs=int(params.get("epochs", 4)),
            context=int(params.get("context", 32)),
            use_onchain=bool(params.get("use_onchain", False)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "transformer":
        predictions = walk_forward_transformer(
            features,
            feature_columns,
            dates,
            config,
            hidden_size=int(params.get("hidden_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0001)),
            epochs=int(params.get("epochs", 4)),
            context=int(params.get("context", 32)),
            use_onchain=bool(params.get("use_onchain", False)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_covariance":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
        )
        raw = _covariance_ridge_targets(
            predictions,
            features,
            risk_aversion=float(params.get("risk_aversion", 1.0)),
            turnover_penalty=float(params.get("turnover_penalty", 0.01)),
        )
    elif mode == "ridge_label_clip":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            label_clip=float(params.get("label_clip", 0.05)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_adaptive_refit":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_residual_gate":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_residual_gate_rolling":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
            window_sessions=int(params.get("window_sessions", 750)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_residual_gate_mad":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
            uncertainty_estimator="mad",
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "xgboost_residual_gate":
        predictions = walk_forward_xgboost(
            features,
            feature_columns,
            dates,
            config,
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            return_uncertainty=True,
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_residual_gate_scaled":
        scaled = features.copy()
        for column in feature_columns:
            scaled[column] = scaled.groupby("asset")[column].transform(
                lambda series: (series - series.rolling(252, min_periods=63).mean())
                / series.rolling(252, min_periods=63).std().replace(0.0, np.nan)
            )
        predictions = walk_forward_ridge(
            scaled,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "lstm_onchain":
        predictions = walk_forward_lstm(
            features,
            feature_columns,
            dates,
            config,
            hidden_size=int(params.get("hidden_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0001)),
            epochs=int(params.get("epochs", 4)),
            context=int(params.get("context", 32)),
            use_onchain=bool(params.get("use_onchain", True)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_residual_gate_momentum":
        momentum_columns = [*feature_columns, "ret_252_skip_21"]
        predictions = walk_forward_ridge(
            features,
            momentum_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_residual_gate_momentum_z":
        augmented = features.copy()
        momentum = augmented.groupby("asset")["ret_252_skip_21"].transform(
            lambda series: (series - series.rolling(252, min_periods=63).mean())
            / series.rolling(252, min_periods=63).std().replace(0.0, np.nan)
        )
        augmented["momentum_12_1_z"] = momentum
        momentum_columns = [*feature_columns, "momentum_12_1_z"]
        predictions = walk_forward_ridge(
            augmented,
            momentum_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "lstm_residual_gate":
        predictions = walk_forward_lstm_gate(
            features,
            feature_columns,
            dates,
            config,
            hidden_size=int(params.get("hidden_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0001)),
            epochs=int(params.get("epochs", 4)),
            context=int(params.get("context", 32)),
            use_onchain=bool(params.get("use_onchain", False)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "lstm_residual_gate_norm":
        predictions = walk_forward_lstm_gate_norm(
            features,
            feature_columns,
            dates,
            config,
            hidden_size=int(params.get("hidden_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0001)),
            epochs=int(params.get("epochs", 4)),
            context=int(params.get("context", 32)),
            use_onchain=bool(params.get("use_onchain", False)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_lstm_blend":
        ridge = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        ridge = _residual_gate(
            ridge,
            float(params.get("uncertainty_multiplier", 0.0)),
        )
        lstm = walk_forward_lstm(
            features,
            feature_columns,
            dates,
            config,
            hidden_size=int(params.get("hidden_size", 32)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=float(params.get("weight_decay", 0.0001)),
            epochs=int(params.get("epochs", 4)),
            context=int(params.get("context", 32)),
            use_onchain=bool(params.get("use_onchain", False)),
        )
        predictions = _blend_ridge_lstm(
            ridge,
            lstm,
            float(params.get("ridge_weight", 0.5)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_residual_size":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        raw = _residual_size_targets(
            predictions,
            float(params.get("uncertainty_multiplier", 0.0)),
            float(params.get("threshold", FIXED_SIGNAL_THRESHOLD)),
        )
    elif mode == "ridge_conformal_gate":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
            calibration_fraction=0.2,
            uncertainty_quantile=float(params.get("uncertainty_quantile", 0.9)),
        )
        predictions = _residual_gate(
            predictions,
            float(params.get("uncertainty_multiplier", 1.0)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ridge_adaptive_covariance":
        predictions = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            refit_sessions=int(params.get("refit_sessions", 21)),
        )
        raw = _covariance_ridge_targets(
            predictions,
            features,
            risk_aversion=float(params.get("risk_aversion", 1.0)),
            turnover_penalty=float(params.get("turnover_penalty", 0.01)),
        )
    elif mode == "ridge_ar_blend":
        ridge = walk_forward_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
        )
        ar = walk_forward_ar(
            features,
            dates,
            config,
            horizon=int(params.get("horizon", 5)),
        )
        predictions = _blend_predictions(ridge, ar, float(params.get("ridge_weight", 0.5)))
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "ar":
        predictions = walk_forward_ar(features, dates, config, horizon=int(params.get("horizon", 5)))
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "timesfm":
        predictions = walk_forward_timesfm(
            features,
            dates,
            config,
            context=int(params.get("context", 256)),
            horizon=int(params.get("horizon", 5)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "timesfm_finetune":
        predictions = walk_forward_timesfm_finetune(
            features,
            dates,
            config,
            context=int(params.get("context", 512)),
            learning_rate=float(params.get("learning_rate", 1e-5)),
            steps=int(params.get("steps", 1)),
            refit_sessions=int(params.get("refit_sessions", 21)),
            horizon=int(params.get("horizon", 5)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "timesfm_finetune_static":
        predictions = walk_forward_timesfm_finetune(
            features,
            dates,
            config,
            context=int(params.get("context", 256)),
            learning_rate=float(params.get("learning_rate", 1e-5)),
            steps=int(params.get("steps", 1)),
            refit_sessions=10**9,
            horizon=int(params.get("horizon", 5)),
            static=True,
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "kronos":
        predictions = walk_forward_kronos(
            features,
            dates,
            config,
            context=int(params.get("context", 256)),
            horizon=int(params.get("horizon", 5)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "xgboost":
        predictions = walk_forward_xgboost(
            features,
            feature_columns,
            dates,
            config,
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            use_onchain=bool(params.get("use_onchain", True)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "lightgbm":
        predictions = walk_forward_lightgbm(
            features,
            feature_columns,
            dates,
            config,
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_alpha=float(params.get("reg_alpha", 0.0)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            use_onchain=bool(params.get("use_onchain", True)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "timesfm_confidence":
        predictions = walk_forward_timesfm(
            features,
            dates,
            config,
            context=int(params.get("context", 256)),
            horizon=int(params.get("horizon", 5)),
        )
        predictions = _apply_uncertainty_gate(
            predictions, float(params.get("max_uncertainty", 0.0))
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "kronos_finetune":
        predictions = walk_forward_kronos_finetune(
            features,
            dates,
            config,
            context=int(params.get("context", 128)),
            learning_rate=float(params.get("learning_rate", 1e-5)),
            steps=int(params.get("steps", 1)),
            refit_sessions=int(params.get("refit_sessions", 21)),
            horizon=int(params.get("horizon", 5)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "kronos_finetune_static":
        predictions = walk_forward_kronos_finetune_static(
            features,
            dates,
            config,
            context=int(params.get("context", 128)),
            learning_rate=float(params.get("learning_rate", 1e-5)),
            steps=int(params.get("steps", 1)),
            horizon=int(params.get("horizon", 5)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    elif mode == "hybrid_timesfm_ridge":
        predictions = walk_forward_timesfm_ridge(
            features,
            feature_columns,
            dates,
            config,
            alpha=float(params.get("alpha", 10.0)),
            use_onchain=bool(params.get("use_onchain", True)),
            context=int(params.get("context", 256)),
            horizon=int(params.get("horizon", 5)),
        )
        threshold = float(params.get("threshold", FIXED_SIGNAL_THRESHOLD))
        raw = _equal_positive(predictions, threshold=threshold)
    else:
        raise ValueError(f"Unknown strategy mode: {mode}")
    targets = _shift_to_next_open(raw, dates)
    audit = targets.reset_index().rename(columns={"date": "execution_date"})
    if not audit.empty and "signal_date" in audit:
        audit["signal_date"] = pd.to_datetime(audit["signal_date"])
        if not (audit["signal_date"] < audit["execution_date"]).all():
            raise ValueError("Signal executed on same or earlier bar")
    return targets, audit
