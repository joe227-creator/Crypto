"""Small causal models used before foundation-model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_TIMESFM_MODEL: Any = None
_TIMESFM_CACHE: dict[tuple[int, int], pd.DataFrame] = {}
_KRONOS_PREDICTOR: Any = None
_KRONOS_CACHE: dict[tuple[int, int], pd.DataFrame] = {}


def _load_timesfm() -> Any:
    global _TIMESFM_MODEL
    if _TIMESFM_MODEL is None:
        from timesfm import configs as _tfc
        from timesfm.timesfm_2p5 import timesfm_2p5_torch as _tft

        model_dir = Path(__file__).resolve().parents[1] / "models" / "timesfm-2.5-200m-pytorch"
        model = _tft.TimesFM_2p5_200M_torch.from_pretrained(
            model_dir, local_files_only=True
        )
        model.compile(
            _tfc.ForecastConfig(
                max_context=512,
                max_horizon=32,
                per_core_batch_size=16,
                normalize_inputs=True,
                infer_is_positive=False,
            )
        )
        _TIMESFM_MODEL = model
    return _TIMESFM_MODEL


def _load_kronos() -> Any:
    global _KRONOS_PREDICTOR
    if _KRONOS_PREDICTOR is None:
        import sys as _sys
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parents[1] / "external" / "kronos"
        if (root / "model" / "kronos.py").exists() and str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from model import Kronos, KronosPredictor, KronosTokenizer

        model_root = Path(__file__).resolve().parents[1] / "models"
        tokenizer = KronosTokenizer.from_pretrained(
            model_root / "kronos-tokenizer-base", local_files_only=True
        )
        kronos_model = Kronos.from_pretrained(
            model_root / "kronos-base", local_files_only=True
        )
        _KRONOS_PREDICTOR = KronosPredictor(model=kronos_model, tokenizer=tokenizer, max_context=512)
    return _KRONOS_PREDICTOR


def walk_forward_kronos(
    features: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    context: int = 256,
    horizon: int = 5,
) -> pd.DataFrame:
    """Zero-shot Kronos OHLCV forecasts mapped to 5-session log returns.

    Context is strictly historical per signal date; no labels or future bars are
    used. Forecasts are cached per (context, horizon) across Optuna trials.
    """
    key = (int(context), int(horizon))
    if key in _KRONOS_CACHE:
        return _KRONOS_CACHE[key]
    predictor = _load_kronos()
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_rows["date"] = pd.to_datetime(asset_rows["date"])
        dates = asset_rows["date"].to_numpy(dtype="datetime64[ns]")
        position = {d: i for i, d in enumerate(dates)}
        ohlcv = asset_rows[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        for signal_date in signal_dates:
            pos = position.get(np.datetime64(signal_date).astype("datetime64[ns]"))
            if pos is None:
                continue
            start = max(0, pos - int(context) + 1)
            window = ohlcv[start : pos + 1]
            if len(window) < 20:
                continue
            idx = pd.date_range(pd.Timestamp(dates[start]), periods=len(window), freq="D")
            df = pd.DataFrame(window, columns=["open", "high", "low", "close", "volume"], index=idx)
            x_stamp = pd.Series(idx)
            y_stamp = pd.Series(pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=int(horizon), freq="D"))
            pred_df = predictor.predict(df, x_stamp, y_stamp, pred_len=int(horizon), verbose=False)
            last_actual = float(ohlcv[pos][3])
            pred_close = float(pred_df["close"].iloc[-1])
            pred = np.log(pred_close) - np.log(last_actual) if last_actual > 0.0 and pred_close > 0.0 else 0.0
            rows.append(
                {"date": pd.Timestamp(dates[pos]), "asset": asset, "prediction": float(pred), "positive": float(pred) > 0.0}
            )
    result = pd.DataFrame(rows)
    _KRONOS_CACHE[key] = result
    return result


def walk_forward_timesfm(
    features: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    context: int = 256,
    horizon: int = 5,
) -> pd.DataFrame:
    """Zero-shot TimesFM log-close forecasts mapped to 5-session returns.

    Context is strictly historical per signal date; no labels or future bars are
    used. Forecasts are cached per (context, horizon) across Optuna trials.
    """
    key = (int(context), int(horizon))
    if key in _TIMESFM_CACHE:
        return _TIMESFM_CACHE[key]
    model = _load_timesfm()
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_rows["date"] = pd.to_datetime(asset_rows["date"])
        log_close = np.log(asset_rows["close"].to_numpy(dtype=float))
        dates = asset_rows["date"].to_numpy(dtype="datetime64[ns]")
        position = {d: i for i, d in enumerate(dates)}
        inputs: list[np.ndarray] = []
        indices: list[int] = []
        for signal_date in signal_dates:
            pos = position.get(np.datetime64(signal_date).astype("datetime64[ns]"))
            if pos is None:
                continue
            start = max(0, pos - int(context) + 1)
            ctx = log_close[start : pos + 1]
            if len(ctx) < 2:
                continue
            inputs.append(ctx.astype(np.float32))
            indices.append(pos)
        chunk = 1024
        for base in range(0, len(inputs), chunk):
            mean, quantiles = model.forecast(horizon=int(horizon), inputs=inputs[base : base + chunk])
            for offset, pos in enumerate(indices[base : base + chunk]):
                last = float(log_close[pos])
                pred = float(mean[offset][int(horizon) - 1]) - last
                q = quantiles[offset][int(horizon) - 1]
                uncertainty = float(np.nanmax(q) - np.nanmin(q))
                rows.append(
                    {
                        "date": pd.Timestamp(dates[pos]),
                        "asset": asset,
                        "prediction": pred,
                        "uncertainty": uncertainty,
                        "positive": pred > 0.0,
                    }
                )
    result = pd.DataFrame(rows)
    _TIMESFM_CACHE[key] = result
    return result


def walk_forward_timesfm_ridge(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    alpha: float,
    use_onchain: bool,
    context: int = 256,
    horizon: int = 5,
) -> pd.DataFrame:
    """Causal Ridge meta-layer using TimesFM forecast as one extra feature."""
    foundation = walk_forward_timesfm(features, signal_dates, config, context=context, horizon=horizon)
    augmented = features.copy()
    forecast = foundation[["date", "asset", "prediction"]].rename(columns={"prediction": "timesfm_pred"})
    augmented = augmented.merge(forecast, on=["date", "asset"], how="left")
    augmented["timesfm_pred"] = augmented["timesfm_pred"].fillna(0.0)
    augmented_columns = [*feature_columns, "timesfm_pred"]
    return walk_forward_ridge(
        augmented,
        augmented_columns,
        signal_dates,
        config,
        alpha=alpha,
        use_onchain=use_onchain,
    )


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


def walk_forward_xgboost(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    use_onchain: bool = True,
) -> pd.DataFrame:
    """Causal XGBoost per-asset, expanding-window, same features as Ridge."""
    import xgboost as _xgb

    selected_features = [c for c in feature_columns if use_onchain or not c.startswith("onchain_")]
    evaluation = config["evaluation"]
    train_start = pd.Timestamp(evaluation["train_start"])
    train_end = pd.Timestamp(evaluation["train_end"]) if evaluation.get("train_end") else pd.Timestamp.max
    validation_end = pd.Timestamp(evaluation["validation_end"]) if evaluation.get("validation_end") else pd.Timestamp.max
    min_train_days = int(evaluation["min_train_days"])
    refit_sessions = int(evaluation["refit_sessions"])
    embargo = int(evaluation.get("embargo_sessions", 1))
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_dates = pd.DatetimeIndex(asset_rows["date"])
        model: Any | None = None
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
                training_end = train_end if signal_date <= validation_end else signal_date
                train = asset_rows[
                    (asset_rows["date"] >= train_start)
                    & (asset_rows["date"] < signal_date)
                    & (asset_rows["date"] <= training_end)
                    & (asset_rows["label_end_date"] < label_cutoff)
                ].dropna(subset=[*selected_features, "label"])
                if train["date"].nunique() >= min_train_days:
                    x = train[selected_features].to_numpy(dtype=float)
                    y = train["label"].to_numpy(dtype=float)
                    model = _xgb.XGBRegressor(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        learning_rate=float(learning_rate),
                        subsample=float(subsample),
                        colsample_bytree=float(colsample_bytree),
                        reg_alpha=float(reg_alpha),
                        reg_lambda=float(reg_lambda),
                        objective="reg:squarederror",
                        verbosity=0,
                        n_jobs=1,
                        random_state=2026,
                    )
                    model.fit(x, y)
                    last_fit_index = signal_index
            if model is None:
                continue
            x_now = current[selected_features].to_numpy(dtype=float)
            if not np.isfinite(x_now).all():
                continue
            prediction = float(model.predict(x_now)[0])
            rows.append(
                {"date": signal_date, "asset": asset, "prediction": prediction, "positive": prediction > 0.0}
            )
    return pd.DataFrame(rows)


def walk_forward_lightgbm(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    use_onchain: bool = True,
) -> pd.DataFrame:
    """Causal LightGBM per-asset, expanding-window, same features as Ridge."""
    import lightgbm as _lgb

    selected_features = [c for c in feature_columns if use_onchain or not c.startswith("onchain_")]
    evaluation = config["evaluation"]
    train_start = pd.Timestamp(evaluation["train_start"])
    train_end = pd.Timestamp(evaluation["train_end"]) if evaluation.get("train_end") else pd.Timestamp.max
    validation_end = pd.Timestamp(evaluation["validation_end"]) if evaluation.get("validation_end") else pd.Timestamp.max
    min_train_days = int(evaluation["min_train_days"])
    refit_sessions = int(evaluation["refit_sessions"])
    embargo = int(evaluation.get("embargo_sessions", 1))
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_dates = pd.DatetimeIndex(asset_rows["date"])
        model: Any | None = None
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
                training_end = train_end if signal_date <= validation_end else signal_date
                train = asset_rows[
                    (asset_rows["date"] >= train_start)
                    & (asset_rows["date"] < signal_date)
                    & (asset_rows["date"] <= training_end)
                    & (asset_rows["label_end_date"] < label_cutoff)
                ].dropna(subset=[*selected_features, "label"])
                if train["date"].nunique() >= min_train_days:
                    x = train[selected_features].to_numpy(dtype=float)
                    y = train["label"].to_numpy(dtype=float)
                    model = _lgb.LGBMRegressor(
                        n_estimators=int(n_estimators),
                        max_depth=int(max_depth),
                        learning_rate=float(learning_rate),
                        subsample=float(subsample),
                        colsample_bytree=float(colsample_bytree),
                        reg_alpha=float(reg_alpha),
                        reg_lambda=float(reg_lambda),
                        objective="regression",
                        verbosity=-1,
                        random_state=2026,
                        n_jobs=1,
                    )
                    model.fit(x, y)
                    last_fit_index = signal_index
            if model is None:
                continue
            x_now = current[selected_features].to_numpy(dtype=float)
            if not np.isfinite(x_now).all():
                continue
            prediction = float(model.predict(x_now)[0])
            rows.append(
                {"date": signal_date, "asset": asset, "prediction": prediction, "positive": prediction > 0.0}
            )
    return pd.DataFrame(rows)


def walk_forward_ridge(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    alpha: float,
    use_onchain: bool,
    label_clip: float | None = None,
    refit_sessions: int | None = None,
    calibration_fraction: float | None = None,
    uncertainty_quantile: float = 0.9,
) -> pd.DataFrame:
    """Predict each asset using only labels ending before current signal date."""
    selected_features = [c for c in feature_columns if use_onchain or not c.startswith("onchain_")]
    evaluation = config["evaluation"]
    train_start = pd.Timestamp(evaluation["train_start"])
    train_end = pd.Timestamp(evaluation["train_end"]) if evaluation.get("train_end") else pd.Timestamp.max
    validation_end = pd.Timestamp(evaluation["validation_end"]) if evaluation.get("validation_end") else pd.Timestamp.max
    min_train_days = int(evaluation["min_train_days"])
    refit_sessions = int(refit_sessions or evaluation["refit_sessions"])
    embargo = int(evaluation.get("embargo_sessions", 1))
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_dates = pd.DatetimeIndex(asset_rows["date"])
        model: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        uncertainty = 0.0
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
                training_end = train_end if signal_date <= validation_end else signal_date
                train = asset_rows[
                    (asset_rows["date"] >= train_start)
                    & (asset_rows["date"] < signal_date)
                    & (asset_rows["date"] <= training_end)
                    & (asset_rows["label_end_date"] < label_cutoff)
                ].dropna(subset=[*selected_features, "label"])
                if train["date"].nunique() >= min_train_days:
                    calibration = pd.DataFrame()
                    fit_train = train
                    if calibration_fraction is not None and 0.0 < float(calibration_fraction) < 0.5:
                        calibration_size = max(20, int(len(train) * float(calibration_fraction)))
                        if len(train) - calibration_size >= min_train_days:
                            fit_train = train.iloc[:-calibration_size]
                            calibration = train.iloc[-calibration_size:]
                    x = fit_train[selected_features].to_numpy(dtype=float)
                    y = fit_train["label"].to_numpy(dtype=float)
                    if label_clip is not None:
                        y = np.clip(y, -float(label_clip), float(label_clip))
                    model = _fit_ridge(x, y, alpha)
                    if not calibration.empty:
                        calibration_x = calibration[selected_features].to_numpy(dtype=float)
                        calibration_y = calibration["label"].to_numpy(dtype=float)
                        residuals = np.abs(calibration_y - _predict(model, calibration_x))
                        uncertainty = max(
                            float(np.quantile(residuals, float(uncertainty_quantile))),
                            1e-6,
                        )
                    else:
                        residuals = y - _predict(model, x)
                        uncertainty = max(float(np.std(residuals, ddof=0)), 1e-6)
                    last_fit_index = signal_index
            if model is None:
                continue
            x_now = current[selected_features].to_numpy(dtype=float)
            if not np.isfinite(x_now).all():
                continue
            prediction = float(_predict(model, x_now)[0])
            rows.append(
                {
                    "date": signal_date,
                    "asset": asset,
                    "prediction": prediction,
                    "uncertainty": uncertainty,
                    "positive": prediction > 0.0,
                }
            )
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
    train_end = pd.Timestamp(evaluation["train_end"]) if evaluation.get("train_end") else pd.Timestamp.max
    validation_end = pd.Timestamp(evaluation["validation_end"]) if evaluation.get("validation_end") else pd.Timestamp.max
    min_train_days = int(evaluation["min_train_days"])
    refit_sessions = int(evaluation["refit_sessions"])
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date")
        returns = asset_rows.set_index("date")["ret_1"].dropna()
        model: tuple[np.ndarray, float] | None = None
        last_fit_index = -refit_sessions
        for signal_index, signal_date in enumerate(signal_dates):
            history_end = min(signal_date, train_end) if signal_date <= validation_end else signal_date
            history = returns[returns.index <= history_end]
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
