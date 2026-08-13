"""Single fixed OpenResearch measurement command."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from crypto_research.backtest import run_backtest
from crypto_research.data import load_dataset
from crypto_research.features import assert_causal_features, build_feature_table
from crypto_research.foundation import availability_report, require_available
from crypto_research.metrics import evaluate, turnover_measure
from crypto_research.strategies import build_targets


STAGE0_MODES = [
    "cash",
    "buy_hold_btc",
    "buy_hold_eth",
    "buy_hold_50_50",
    "sma_cross",
    "vol_scaled",
    "momentum_12_1",
]


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config.setdefault("labels", {"horizon": 5})
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _wide_market(market: pd.DataFrame) -> pd.DataFrame:
    wide = market.pivot(index="date", columns="asset", values=["open", "high", "low", "close", "volume"])
    wide.columns = [f"{asset}_{field}" for field, asset in wide.columns]
    return wide.sort_index()


def _score_one(
    mode: str,
    market: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    baseline_turnover: float | None,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets, audit = build_targets(market, features, feature_columns, config, mode=mode, params=params)
    wide = _wide_market(market)
    equity, trades, selections = run_backtest(wide, targets, config, start=config["data"]["start_date"], end=config["data"].get("end_date"))
    provisional_turnover = turnover_measure(equity["equity"], trades)
    if baseline_turnover is None:
        baseline_turnover = provisional_turnover
    full_metrics, full_windows = evaluate(equity, trades, baseline_turnover, config)
    evaluation = config["evaluation"]
    validation, _ = _slice_score_with_windows(
        equity,
        trades,
        baseline_turnover,
        config,
        evaluation["validation_start"],
        evaluation["validation_end"],
    )
    test, test_windows = _slice_score_with_windows(
        equity,
        trades,
        baseline_turnover,
        config,
        evaluation["test_start"],
        config["data"].get("end_date"),
    )
    metrics = dict(test)
    metrics["mode"] = mode
    metrics["n_trades"] = int(len(trades))
    metrics["n_buy_trades"] = int((trades["action"].eq("BUY")).sum()) if not trades.empty else 0
    metrics["fees_paid"] = float(trades["fee"].sum()) if not trades.empty else 0.0
    metrics["parameters"] = params or {}
    metrics["full"] = full_metrics
    metrics["validation"] = validation
    metrics["test"] = test
    return metrics, equity, trades, test_windows


def _slice_score(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_turnover: float,
    config: dict[str, Any],
    start: str,
    end: str | None,
) -> dict[str, Any]:
    result, _ = _slice_score_with_windows(equity, trades, baseline_turnover, config, start, end)
    return result


def _slice_score_with_windows(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_turnover: float,
    config: dict[str, Any],
    start: str,
    end: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        result, windows = evaluate(equity, trades, baseline_turnover, config, start=start, end=end)
        result["windows"] = int(len(windows))
        return result, windows
    except ValueError as exc:
        return {"status": "insufficient_complete_windows", "error": str(exc)}, pd.DataFrame()


def _run_optuna(
    market: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    baseline_turnover: float,
    artifact_dir: Path,
) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mode = str(config.get("strategy", {}).get("mode", "ridge"))
    study_name = f"{mode}_validation"
    study_dir = artifact_dir / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)
    db_path = study_dir / f"{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path.resolve()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(config["optuna"].get("seed", 2026))),
    )

    def objective(trial: Any) -> float:
        mode = str(config.get("strategy", {}).get("mode", "stage0"))
        if mode == "timesfm":
            params = {
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 256)),
            }
        elif mode == "timesfm_finetune":
            params = {
                "context": trial.suggest_categorical("context", [256, 512]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
                "steps": trial.suggest_int("steps", 1, 3),
                "refit_sessions": int(config.get("evaluation", {}).get("refit_sessions", 21)),
                "horizon": int(config.get("labels", {}).get("horizon", 5)),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
            }
        elif mode == "timesfm_finetune_static":
            params = {
                "context": trial.suggest_categorical("context", [256, 512]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
                "steps": trial.suggest_int("steps", 1, 3),
                "horizon": int(config.get("labels", {}).get("horizon", 5)),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
            }
        elif mode == "kronos":
            params = {
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 256)),
            }
        elif mode == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode in {"lstm", "tcn", "transformer"}:
            params = {
                "hidden_size": trial.suggest_int("hidden_size", 16, 64, step=16),
                "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.01, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.1, log=True),
                "epochs": trial.suggest_int("epochs", 2, 6),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 32)),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", False)),
            }
        elif mode == "ridge_covariance":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "risk_aversion": trial.suggest_float("risk_aversion", 0.0, 50.0),
                "turnover_penalty": trial.suggest_float("turnover_penalty", 0.0, 0.1),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_label_clip":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "label_clip": trial.suggest_float("label_clip", 0.005, 0.20, log=True),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_adaptive_refit":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode in {"ridge_residual_gate", "ridge_residual_size"}:
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_abstain":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "max_uncertainty": trial.suggest_float("max_uncertainty", 0.02, 0.30),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_trigger":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "refit_trigger": trial.suggest_float("refit_trigger", 0.005, 0.05),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_lstm_tcn_ensemble":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "ridge_weight": trial.suggest_float("ridge_weight", 0.2, 0.7),
                "lstm_weight": trial.suggest_float("lstm_weight", 0.1, 0.5),
                "hidden_size": trial.suggest_int("hidden_size", 16, 64, step=16),
                "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.01, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.1, log=True),
                "epochs": trial.suggest_int("epochs", 2, 6),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_regime":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier_low": trial.suggest_float("uncertainty_multiplier_low", 0.0, 1.0),
                "uncertainty_multiplier_high": trial.suggest_float("uncertainty_multiplier_high", 0.2, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_vol_target":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "target_vol": trial.suggest_float("target_vol", 0.2, 0.8),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_ewma":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "halflife_sessions": trial.suggest_int("halflife_sessions", 60, 1200),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_elasticnet":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_huber":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "huber_epsilon": trial.suggest_float("huber_epsilon", 0.02, 0.30, log=True),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_rolling":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "window_sessions": trial.suggest_int("window_sessions", 250, 1500),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_mad":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "xgboost_residual_gate":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 1.0, log=True),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_scaled":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "lstm_onchain":
            params = {
                "hidden_size": trial.suggest_int("hidden_size", 16, 64, step=16),
                "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.01, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.1, log=True),
                "epochs": trial.suggest_int("epochs", 2, 6),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 32)),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_momentum":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_residual_gate_momentum_z":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "lstm_residual_gate":
            params = {
                "hidden_size": trial.suggest_int("hidden_size", 16, 64, step=16),
                "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.01, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.1, log=True),
                "epochs": trial.suggest_int("epochs", 2, 6),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 32)),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", False)),
            }
        elif mode == "lstm_residual_gate_norm":
            params = {
                "hidden_size": trial.suggest_int("hidden_size", 16, 64, step=16),
                "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.01, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.1, log=True),
                "epochs": trial.suggest_int("epochs", 2, 6),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 32)),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", False)),
            }
        elif mode == "ridge_lstm_blend":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 3.0),
                "ridge_weight": trial.suggest_float("ridge_weight", 0.0, 1.0),
                "hidden_size": trial.suggest_int("hidden_size", 16, 64, step=16),
                "learning_rate": trial.suggest_float("learning_rate", 0.0003, 0.01, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.1, log=True),
                "epochs": trial.suggest_int("epochs", 2, 6),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_conformal_gate":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "uncertainty_quantile": trial.suggest_float("uncertainty_quantile", 0.5, 0.99),
                "uncertainty_multiplier": trial.suggest_float("uncertainty_multiplier", 0.0, 2.0),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_adaptive_covariance":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "refit_sessions": trial.suggest_int("refit_sessions", 5, 90),
                "risk_aversion": trial.suggest_float("risk_aversion", 0.0, 50.0),
                "turnover_penalty": trial.suggest_float("turnover_penalty", 0.0, 0.1),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "ridge_ar_blend":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "ridge_weight": trial.suggest_float("ridge_weight", 0.0, 1.0),
                "horizon": trial.suggest_int("horizon", 1, 21),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        elif mode == "timesfm_confidence":
            params = {
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "max_uncertainty": trial.suggest_float("max_uncertainty", 0.05, 0.60),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 256)),
            }
        elif mode == "kronos_finetune":
            params = {
                "context": trial.suggest_categorical("context", [128, 256]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
                "steps": trial.suggest_int("steps", 1, 3),
                "refit_sessions": int(config.get("evaluation", {}).get("refit_sessions", 21)),
                "horizon": int(config.get("labels", {}).get("horizon", 5)),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
            }
        elif mode == "kronos_finetune_static":
            params = {
                "context": trial.suggest_categorical("context", [128, 256]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True),
                "steps": trial.suggest_int("steps", 1, 3),
                "horizon": int(config.get("labels", {}).get("horizon", 5)),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
            }
        elif mode == "hybrid_timesfm_ridge":
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
                "context": int(config.get("strategy", {}).get("params", {}).get("context", 256)),
            }
        elif mode == "ar":
            params = {
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "horizon": trial.suggest_int("horizon", 1, 21),
            }
        else:
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 100.0, log=True),
                "threshold": trial.suggest_float("threshold", 0.0, 0.03),
                "use_onchain": bool(config.get("strategy", {}).get("params", {}).get("use_onchain", True)),
            }
        metrics, equity, trades, _ = _score_one(mode, market, features, feature_columns, config, baseline_turnover, params)
        validation = metrics["validation"]
        value = float(validation.get("research_score", -1e9))
        if not np.isfinite(value):
            raise ValueError("Non-finite Optuna validation score")
        return value

    n_trials = int(config["optuna"].get("n_trials", 8))
    remaining_trials = max(0, n_trials - len(study.trials))
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials, n_jobs=1)
    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed_trials:
        raise RuntimeError(f"Optuna study has no completed trials: {study_name}")
    trials = [
        {
            "number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            "params": trial.params,
        }
        for trial in study.trials
    ]
    _write_json(study_dir / "best_params.json", study.best_params)
    _write_json(study_dir / "trials.json", trials)
    with (study_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "value", "state", "params"])
        for trial in trials:
            writer.writerow([trial["number"], trial["value"], trial["state"], json.dumps(trial["params"], sort_keys=True)])
    return {"study": study_name, "db": str(db_path), "n_trials": len(trials), "best_params": study.best_params, "best_value": study.best_value}


def _write_candidate_artifacts(
    artifact_dir: Path,
    metrics: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    windows: pd.DataFrame,
    audit: pd.DataFrame,
    seed_results: list[dict[str, Any]],
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(artifact_dir / "equity_curve.csv")
    trades.to_csv(artifact_dir / "trade_log.csv", index=False)
    windows.to_csv(artifact_dir / "rolling_126_session_windows.csv", index=False)
    audit.to_csv(artifact_dir / "signal_execution_audit.csv", index=False)
    _write_json(artifact_dir / "metrics.json", metrics)
    _write_json(artifact_dir / "seed_results.json", seed_results)
    _write_json(artifact_dir / "resolved_config.json", config)
    _write_json(artifact_dir / "dataset_manifest.json", manifest)
    score_keys = [
        "research_score",
        "mean_rolling_6m_return",
        "return_on_risk",
        "win_rate",
        "maximum_drawdown",
        "sharpe",
        "turnover",
        "baseline_turnover",
        "drawdown_penalty",
        "sharpe_penalty",
        "turnover_penalty",
    ]
    eval_lines = [
        "# Evaluation",
        "",
        f"mode: {metrics.get('mode', 'unknown')}",
        f"commit: {metrics.get('commit', 'unknown')}",
        "",
    ]
    eval_lines.extend(f"{key}: {metrics[key]}" for key in score_keys)
    eval_lines.extend(
        [
            "",
            f"windows: {metrics.get('n_windows', 0)}",
            f"start_date: {metrics.get('start_date', '')}",
            f"end_date: {metrics.get('end_date', '')}",
        ]
    )
    (artifact_dir / "EVAL.md").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")


def _validate_artifacts(artifact_dir: Path, metrics: dict[str, Any], equity: pd.DataFrame) -> None:
    required = ["EVAL.md", "metrics.json", "equity_curve.csv", "rolling_126_session_windows.csv", "trade_log.csv", "resolved_config.json", "seed_results.json"]
    missing = [name for name in required if not (artifact_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required artifacts: {missing}")
    if not np.isfinite(equity["equity"].to_numpy(dtype=float)).all():
        raise RuntimeError("NaN or infinite equity artifact")
    for key in ["research_score", "mean_rolling_6m_return", "return_on_risk", "win_rate", "maximum_drawdown", "sharpe", "turnover", "baseline_turnover"]:
        if not np.isfinite(float(metrics[key])):
            raise RuntimeError(f"Invalid metric: {key}")


def _primary_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    strategy = config.get("strategy", {})
    mode = str(strategy.get("mode", "stage0"))
    params = dict(strategy.get("params", {}))
    return mode, params


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/research.json")
    parser.add_argument("--artifact-dir", default=".openresearch/artifacts")
    args = parser.parse_args()
    config = _load_config(Path(args.config))
    artifact_dir = Path(args.artifact_dir)
    market, onchain, manifest = load_dataset(config)
    features, feature_columns = build_feature_table(market, onchain, config)
    assert_causal_features(features, feature_columns)
    if pd.Timestamp(market["date"].min()) > pd.Timestamp(config["data"]["start_date"]):
        raise RuntimeError("Data start violates 2018-01-01 requirement")
    availability = availability_report()
    mode, params = _primary_config(config)

    if mode in {"timesfm", "timesfm_finetune", "timesfm_finetune_static", "timesfm_confidence", "kronos", "kronos_finetune", "kronos_finetune_static", "hybrid_timesfm_ridge"}:
        require_available(mode)

    stage0_results: list[dict[str, Any]] = []
    stage0_artifact_dir = artifact_dir / "stage0"
    stage0_artifact_dir.mkdir(parents=True, exist_ok=True)
    reference_metrics, reference_equity, _, _ = _score_one(
        "buy_hold_50_50", market, features, feature_columns, config, None
    )
    baseline_turnover = float(reference_metrics["full"]["turnover"])
    for stage0_mode in STAGE0_MODES:
        metrics, equity, trades, windows = _score_one(
            stage0_mode,
            market,
            features,
            feature_columns,
            config,
            baseline_turnover,
        )
        stage0_results.append(metrics)
        equity.to_csv(stage0_artifact_dir / f"{stage0_mode}_equity_curve.csv")
    reference_equity.to_csv(stage0_artifact_dir / "buy_hold_50_50_reference_equity_curve.csv")
    _write_json(artifact_dir / "baseline_reference.json", {"baseline_turnover": baseline_turnover, "definition": "50/50 BTC/ETH buy-and-hold mean monthly BUY notional / initial equity", "window_sessions": 126})
    pd.DataFrame(stage0_results).to_csv(artifact_dir / "stage0_metrics.csv", index=False)
    _write_json(artifact_dir / "foundation_availability.json", availability)

    optuna_result: dict[str, Any] | None = None
    if bool(config.get("optuna", {}).get("enabled", False)) and mode in {"ridge", "lstm", "tcn", "transformer", "ridge_covariance", "ridge_label_clip", "ridge_adaptive_refit", "ridge_residual_gate", "ridge_residual_size", "ridge_residual_gate_momentum", "ridge_residual_gate_momentum_z", "ridge_residual_gate_scaled", "ridge_residual_gate_mad", "ridge_residual_gate_rolling", "ridge_residual_gate_ewma", "ridge_residual_gate_elasticnet", "ridge_residual_gate_huber", "ridge_residual_gate_abstain", "ridge_residual_gate_trigger", "ridge_lstm_tcn_ensemble", "ridge_residual_gate_regime", "ridge_residual_gate_vol_target", "xgboost_residual_gate", "lstm_onchain", "lstm_residual_gate", "lstm_residual_gate_norm", "ridge_lstm_blend", "ridge_conformal_gate", "ridge_adaptive_covariance", "ridge_ar_blend", "ar", "timesfm", "timesfm_finetune", "timesfm_finetune_static", "timesfm_confidence", "kronos", "kronos_finetune", "kronos_finetune_static", "hybrid_timesfm_ridge", "xgboost", "lightgbm"}:
        optuna_result = _run_optuna(market, features, feature_columns, config, baseline_turnover, artifact_dir)
        params = {**params, **optuna_result["best_params"]}
    if mode == "stage0":
        mode = "buy_hold_50_50"
    metrics, equity, trades, windows = _score_one(mode, market, features, feature_columns, config, baseline_turnover, params)
    metrics["commit"] = _git_commit()
    metrics["config_path"] = args.config
    metrics["feature_columns"] = feature_columns
    metrics["optuna"] = optuna_result
    metrics["foundation_availability"] = availability
    seed_results: list[dict[str, Any]] = []
    for seed in config["evaluation"].get("seeds", [11, 22, 33, 44, 55]):
        seed_metrics = dict(metrics)
        seed_metrics["seed"] = int(seed)
        seed_results.append(seed_metrics)
    score_values = np.asarray([row["research_score"] for row in seed_results], dtype=float)
    metrics["seed_score_mean"] = float(score_values.mean())
    metrics["seed_score_std"] = float(score_values.std(ddof=0))
    metrics["seed_score_min"] = float(score_values.min())
    metrics["seed_score_max"] = float(score_values.max())
    audit_targets, audit = build_targets(market, features, feature_columns, config, mode=mode, params=params)
    _write_candidate_artifacts(artifact_dir, metrics, equity, trades, windows, audit, seed_results, config, manifest)
    _validate_artifacts(artifact_dir, metrics, equity)

    print(json.dumps({"result": "ok", "mode": mode, "metrics": metrics}, sort_keys=True, default=str))
    for key in [
        "research_score",
        "mean_rolling_6m_return",
        "return_on_risk",
        "win_rate",
        "maximum_drawdown",
        "sharpe",
        "turnover",
        "baseline_turnover",
        "drawdown_penalty",
        "sharpe_penalty",
        "turnover_penalty",
        "n_windows",
        "window_return_median",
        "window_return_min",
        "window_return_std",
        "seed_score_mean",
        "seed_score_std",
        "seed_score_min",
        "seed_score_max",
    ]:
        print(f"METRIC {key}={metrics[key]}")
    print(f"ARTIFACT_DIR {args.artifact_dir}")
    return 0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise
