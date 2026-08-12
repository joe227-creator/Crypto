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


def _load_timesfm_trainable() -> Any:
    from timesfm.timesfm_2p5 import timesfm_2p5_torch as _tft

    model_dir = Path(__file__).resolve().parents[1] / "models" / "timesfm-2.5-200m-pytorch"
    return _tft.TimesFM_2p5_200M_torch.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_compile=False,
    )


def _timesfm_point_forecast(model: Any, contexts: Any) -> Any:
    import torch
    from timesfm.torch import util
    from timesfm.timesfm_2p5.timesfm_2p5_torch import revin

    patched = contexts.reshape(contexts.shape[0], -1, model.model.p)
    masks = torch.zeros_like(patched, dtype=torch.bool)
    n = torch.zeros(contexts.shape[0], device=contexts.device)
    mu = torch.zeros_like(n)
    sigma = torch.zeros_like(n)
    patch_mu: list[Any] = []
    patch_sigma: list[Any] = []
    for index in range(patched.shape[1]):
        (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, patched[:, index], masks[:, index])
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)
    normalized = revin(patched, context_mu, context_sigma, reverse=False)
    (__, __, output, __), __ = model.model(normalized, masks)
    output = revin(output, context_mu, context_sigma, reverse=True)
    output = output.reshape(contexts.shape[0], -1, model.model.o, model.model.q)
    return output[:, -1, :5, model.model.aridx]


def _fine_tune_timesfm_head(
    model: Any,
    contexts: Any,
    targets: Any,
    learning_rate: float,
    steps: int,
) -> None:
    import torch

    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.model.output_projection_point.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.model.output_projection_point.parameters(),
        lr=float(learning_rate),
        weight_decay=0.0001,
    )
    loss_fn = torch.nn.SmoothL1Loss()
    model.model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        prediction = _timesfm_point_forecast(model, contexts)
        loss = loss_fn(prediction, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.output_projection_point.parameters(), 1.0)
        optimizer.step()
    model.model.eval()


def walk_forward_timesfm_finetune(
    features: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    context: int = 512,
    learning_rate: float = 1e-5,
    steps: int = 1,
    refit_sessions: int = 21,
    horizon: int = 5,
) -> pd.DataFrame:
    """Causal TimesFM point-head fine-tuning on historical price paths."""
    import torch

    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        model = _load_timesfm_trainable()
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_rows["date"] = pd.to_datetime(asset_rows["date"])
        dates = asset_rows["date"].to_numpy(dtype="datetime64[ns]")
        label_end_dates = asset_rows["label_end_date"].to_numpy(dtype="datetime64[ns]")
        log_close = np.log(asset_rows["close"].to_numpy(dtype=float))
        position = {date: index for index, date in enumerate(dates)}
        last_fit_index = -int(refit_sessions)
        for signal_index, signal_date in enumerate(signal_dates):
            position_now = position.get(np.datetime64(signal_date).astype("datetime64[ns]"))
            if position_now is None:
                continue
            if signal_index - last_fit_index >= int(refit_sessions):
                cutoff_position = position_now - int(config["evaluation"].get("embargo_sessions", 1))
                if cutoff_position > 0:
                    label_cutoff = dates[cutoff_position]
                    train_end = pd.Timestamp(config["evaluation"].get("train_end", signal_date))
                    if pd.Timestamp(signal_date) > pd.Timestamp(config["evaluation"].get("validation_end", signal_date)):
                        train_end = pd.Timestamp(signal_date)
                    sample_positions = [
                        index
                        for index in range(int(context) - 1, position_now)
                        if pd.Timestamp(dates[index]) >= pd.Timestamp(config["evaluation"]["train_start"])
                        and pd.Timestamp(dates[index]) <= train_end
                        and label_end_dates[index] < label_cutoff
                    ][-128:]
                    if len(sample_positions) >= 16:
                        context_values = np.stack(
                            [log_close[index - int(context) + 1 : index + 1] for index in sample_positions]
                        ).astype(np.float32)
                        target_values = np.stack(
                            [log_close[index + 1 : index + 1 + int(horizon)] for index in sample_positions]
                        ).astype(np.float32)
                        device = model.model.device
                        context_tensor = torch.as_tensor(context_values, device=device)
                        target_tensor = torch.as_tensor(target_values, device=device)
                        _fine_tune_timesfm_head(
                            model,
                            context_tensor,
                            target_tensor,
                            learning_rate,
                            steps,
                        )
                        last_fit_index = signal_index
            start = max(0, position_now - int(context) + 1)
            current = log_close[start : position_now + 1]
            if len(current) < int(context):
                continue
            device = model.model.device
            context_tensor = torch.as_tensor(current[None, :].astype(np.float32), device=device)
            with torch.no_grad():
                forecast = _timesfm_point_forecast(model, context_tensor)[0]
            prediction = float(forecast[int(horizon) - 1] - log_close[position_now])
            rows.append(
                {
                    "date": pd.Timestamp(dates[position_now]),
                    "asset": asset,
                    "prediction": prediction,
                    "positive": prediction > 0.0,
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


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


def _make_neural_model(kind: str, input_size: int, hidden_size: int) -> Any:
    import torch
    from torch import nn

    class LSTMModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, values: Any) -> Any:
            encoded, _ = self.encoder(values)
            return self.head(encoded[:, -1]).squeeze(-1)

    class TCNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2),
                nn.GELU(),
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, values: Any) -> Any:
            encoded = self.network(values.transpose(1, 2))[:, :, : values.shape[1]]
            return self.head(encoded[:, :, -1]).squeeze(-1)

    class TransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(input_size, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4 if hidden_size % 4 == 0 else 1,
                dim_feedforward=hidden_size * 2,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, values: Any) -> Any:
            length = values.shape[1]
            mask = torch.triu(
                torch.ones(length, length, dtype=torch.bool, device=values.device),
                diagonal=1,
            )
            encoded = self.encoder(self.projection(values), mask=mask)
            return self.head(encoded[:, -1]).squeeze(-1)

    if kind == "lstm":
        return LSTMModel()
    if kind == "tcn":
        return TCNModel()
    if kind == "transformer":
        return TransformerModel()
    raise ValueError(f"Unknown neural model: {kind}")


def _fit_neural_model(
    kind: str,
    x: np.ndarray,
    y: np.ndarray,
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
) -> tuple[Any, np.ndarray, np.ndarray, Any]:
    import torch

    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat = x.reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    model = _make_neural_model(kind, x.shape[-1], int(hidden_size)).to(device)
    values = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    targets = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = torch.nn.SmoothL1Loss()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(values), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model, mean, scale, device


def _walk_forward_neural(
    kind: str,
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    context: int,
    use_onchain: bool,
) -> pd.DataFrame:
    import torch

    selected_features = [c for c in feature_columns if use_onchain or not c.startswith("onchain_")]
    evaluation = config["evaluation"]
    train_start = pd.Timestamp(evaluation["train_start"])
    train_end = pd.Timestamp(evaluation["train_end"]) if evaluation.get("train_end") else pd.Timestamp.max
    validation_end = pd.Timestamp(evaluation["validation_end"]) if evaluation.get("validation_end") else pd.Timestamp.max
    min_train_days = int(evaluation["min_train_days"])
    refit_sessions = int(evaluation["refit_sessions"])
    embargo = int(evaluation.get("embargo_sessions", 1))
    context = int(context)
    rows: list[dict[str, Any]] = []
    for asset in config["data"]["assets"]:
        asset_rows = features[features["asset"].eq(asset)].sort_values("date").copy()
        asset_dates = pd.DatetimeIndex(asset_rows["date"])
        values = asset_rows[selected_features].to_numpy(dtype=float)
        labels = asset_rows["label"].to_numpy(dtype=float)
        label_end_dates = pd.DatetimeIndex(asset_rows["label_end_date"])
        sequences = np.zeros((len(asset_rows), context, len(selected_features)), dtype=np.float32)
        valid_sequences = np.zeros(len(asset_rows), dtype=bool)
        for position in range(context - 1, len(asset_rows)):
            window = values[position - context + 1 : position + 1]
            if np.isfinite(window).all():
                sequences[position] = window.astype(np.float32)
                valid_sequences[position] = True
        model_state: tuple[Any, np.ndarray, np.ndarray, Any] | None = None
        last_fit_index = -refit_sessions
        for signal_index, signal_date in enumerate(signal_dates):
            current = asset_rows[asset_rows["date"].eq(signal_date)]
            if current.empty:
                continue
            asset_position = asset_dates.get_loc(signal_date)
            if signal_index - last_fit_index >= refit_sessions or model_state is None:
                cutoff_position = asset_position - embargo
                if cutoff_position <= 0:
                    continue
                label_cutoff = asset_dates[cutoff_position]
                training_end = train_end if signal_date <= validation_end else signal_date
                eligible = (
                    (asset_dates >= train_start)
                    & (asset_dates < signal_date)
                    & (asset_dates <= training_end)
                    & (label_end_dates < label_cutoff)
                    & np.isfinite(labels)
                    & valid_sequences
                )
                positions = np.flatnonzero(eligible)
                if len(positions) >= min_train_days:
                    model_state = _fit_neural_model(
                        kind,
                        sequences[positions],
                        labels[positions],
                        hidden_size,
                        learning_rate,
                        weight_decay,
                        epochs,
                    )
                    last_fit_index = signal_index
            if model_state is None or not valid_sequences[asset_position]:
                continue
            model, mean, scale, device = model_state
            current_sequence = (sequences[asset_position].astype(float) - mean) / scale
            tensor = torch.as_tensor(current_sequence[None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                prediction = float(model(tensor).detach().cpu().item())
            rows.append(
                {
                    "date": signal_date,
                    "asset": asset,
                    "prediction": prediction,
                    "positive": prediction > 0.0,
                }
            )
    return pd.DataFrame(rows)


def walk_forward_lstm(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    hidden_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 4,
    context: int = 32,
    use_onchain: bool = False,
) -> pd.DataFrame:
    return _walk_forward_neural(
        "lstm", features, feature_columns, signal_dates, config,
        hidden_size, learning_rate, weight_decay, epochs, context, use_onchain,
    )


def walk_forward_tcn(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    hidden_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 4,
    context: int = 32,
    use_onchain: bool = False,
) -> pd.DataFrame:
    return _walk_forward_neural(
        "tcn", features, feature_columns, signal_dates, config,
        hidden_size, learning_rate, weight_decay, epochs, context, use_onchain,
    )


def walk_forward_transformer(
    features: pd.DataFrame,
    feature_columns: list[str],
    signal_dates: pd.DatetimeIndex,
    config: dict[str, Any],
    hidden_size: int = 32,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 4,
    context: int = 32,
    use_onchain: bool = False,
) -> pd.DataFrame:
    return _walk_forward_neural(
        "transformer", features, feature_columns, signal_dates, config,
        hidden_size, learning_rate, weight_decay, epochs, context, use_onchain,
    )


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
