"""Immutable Research_score and robustness metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


WINDOW_SESSIONS = 126
ANNUALIZATION_SESSIONS = 252


def complete_window_returns(equity: pd.Series, window_sessions: int = WINDOW_SESSIONS) -> pd.DataFrame:
    values = equity.dropna().sort_index()
    count = len(values) // window_sessions
    rows: list[dict[str, Any]] = []
    for number in range(count):
        chunk = values.iloc[number * window_sessions : (number + 1) * window_sessions]
        rows.append(
            {
                "window": number + 1,
                "start_date": pd.Timestamp(chunk.index[0]).date().isoformat(),
                "end_date": pd.Timestamp(chunk.index[-1]).date().isoformat(),
                "sessions": int(len(chunk)),
                "return": float(chunk.iloc[-1] / chunk.iloc[0] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def maximum_drawdown(equity: pd.Series) -> tuple[float, pd.Series]:
    values = equity.dropna().sort_index()
    drawdown = values / values.cummax() - 1.0
    return (float(drawdown.min()) if not drawdown.empty else 0.0), drawdown


def annualized_sharpe(equity: pd.Series, rf_annual: float = 0.0) -> float:
    values = equity.dropna().sort_index()
    returns = values.pct_change().dropna()
    if returns.empty:
        return 0.0
    std = float(returns.std(ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return 0.0
    excess = returns - rf_annual / ANNUALIZATION_SESSIONS
    return float(np.sqrt(ANNUALIZATION_SESSIONS) * excess.mean() / std)


def turnover_measure(equity: pd.Series, trade_log: pd.DataFrame) -> float:
    """Mean monthly BUY notional divided by initial equity."""
    if equity.empty or trade_log.empty or "action" not in trade_log:
        return 0.0
    buys = trade_log[trade_log["action"].eq("BUY")].copy()
    if buys.empty:
        return 0.0
    gross = pd.to_numeric(buys["gross_value"], errors="coerce")
    buys["gross_value"] = gross
    buys["month"] = pd.to_datetime(buys["date"]).dt.to_period("M")
    monthly_buy = buys.groupby("month")["gross_value"].sum().dropna()
    if monthly_buy.empty or float(equity.iloc[0]) == 0.0:
        return 0.0
    return float(monthly_buy.mean() / float(equity.iloc[0]))


def research_score(
    mean_window_return: float,
    maximum_dd: float,
    sharpe: float,
    turnover: float,
    baseline_turnover: float,
    win_rate: float,
) -> dict[str, float]:
    """Apply supplied formula without tunable weights or thresholds."""
    values = [mean_window_return, maximum_dd, sharpe, turnover, baseline_turnover, win_rate]
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"Non-finite score input: {values}")
    return_on_risk = mean_window_return / max(abs(maximum_dd), 1e-6)
    drawdown_penalty = 0.35 * max(0.0, -0.50 - maximum_dd)
    sharpe_penalty = 0.15 * max(0.0, 0.80 - sharpe)
    turnover_penalty = 0.10 * max(0.0, turnover - baseline_turnover)
    score = (
        mean_window_return
        + 0.20 * return_on_risk
        + 0.10 * win_rate
        - drawdown_penalty
        - sharpe_penalty
        - turnover_penalty
    )
    return {
        "research_score": float(score),
        "mean_rolling_6m_return": float(mean_window_return),
        "return_on_risk": float(return_on_risk),
        "win_rate": float(win_rate),
        "maximum_drawdown": float(maximum_dd),
        "sharpe": float(sharpe),
        "turnover": float(turnover),
        "baseline_turnover": float(baseline_turnover),
        "drawdown_penalty": float(drawdown_penalty),
        "sharpe_penalty": float(sharpe_penalty),
        "turnover_penalty": float(turnover_penalty),
    }


def evaluate(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    baseline_turnover: float,
    config: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    equity = equity_curve["equity"].copy()
    if start:
        equity = equity[equity.index >= pd.Timestamp(start)]
    if end:
        equity = equity[equity.index <= pd.Timestamp(end)]
    if equity.empty or (equity <= 0.0).any() or not equity.index.is_monotonic_increasing:
        raise ValueError("Invalid equity curve")
    windows = complete_window_returns(equity, int(config["evaluation"]["window_sessions"]))
    if len(windows) < 2:
        raise ValueError("At least two complete 126-session windows required")
    max_dd, drawdown = maximum_drawdown(equity)
    if start or end:
        trades = trade_log.copy()
        if not trades.empty:
            if start:
                trades = trades[trades["date"] >= pd.Timestamp(start)]
            if end:
                trades = trades[trades["date"] <= pd.Timestamp(end)]
    else:
        trades = trade_log
    mean_return = float(windows["return"].mean())
    win_rate = float((windows["return"] > 0.0).mean())
    metrics = research_score(
        mean_return,
        max_dd,
        annualized_sharpe(equity, float(config["evaluation"].get("rf_annual", 0.0))),
        turnover_measure(equity, trades),
        baseline_turnover,
        win_rate,
    )
    metrics.update(
        {
            "n_windows": int(len(windows)),
            "window_return_median": float(windows["return"].median()),
            "window_return_min": float(windows["return"].min()),
            "window_return_max": float(windows["return"].max()),
            "window_return_std": float(windows["return"].std(ddof=0)),
            "ending_equity": float(equity.iloc[-1]),
            "start_date": pd.Timestamp(equity.index[0]).date().isoformat(),
            "end_date": pd.Timestamp(equity.index[-1]).date().isoformat(),
            "max_drawdown_duration_sessions": int(_drawdown_duration(drawdown)),
        }
    )
    return metrics, windows


def _drawdown_duration(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
