"""Long-only cash portfolio simulator with explicit fee and slippage costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


ASSETS = ("BTC", "ETH")


def _price(market: pd.DataFrame, date: pd.Timestamp, asset: str, field: str) -> float:
    value = market.loc[date, f"{asset}_{field}"]
    return float(value) if pd.notna(value) and float(value) > 0.0 else np.nan


def _validate_targets(targets: pd.DataFrame, max_weight: float) -> None:
    if targets.empty:
        return
    values = targets.reindex(columns=list(ASSETS), fill_value=0.0).fillna(0.0)
    if (values < -1e-12).any().any() or (values > max_weight + 1e-12).any().any():
        raise ValueError("Invalid long-only target weight")
    if (values.sum(axis=1) > 1.0 + 1e-9).any():
        raise ValueError("Target weights exceed 100 percent")


def run_backtest(
    market: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run next-open rebalances; target index is execution date, not signal date."""
    dates = pd.DatetimeIndex(market.index).sort_values()
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if dates.empty:
        raise ValueError("No backtest dates")
    targets = targets.copy()
    targets.index = pd.to_datetime(targets.index).normalize()
    _validate_targets(targets, float(config["backtest"]["max_weight_per_asset"]))
    fee_rate = float(config["backtest"]["fee_bps"]) / 10_000.0
    slippage = float(config["backtest"]["slippage_bps"]) / 10_000.0
    cash = float(config["backtest"]["initial_capital"])
    holdings = {asset: 0.0 for asset in ASSETS}
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for date in dates:
        target = targets.loc[date] if date in targets.index else None
        if target is not None:
            target_values = {asset: float(target.get(asset, 0.0) or 0.0) for asset in ASSETS}
            target_rows.append({"date": date, "target_signal": target.attrs.get("signal_date") if hasattr(target, "attrs") else None, **target_values})
            # Fully liquidate before buying. This is conservative and makes every
            # rebalance cost-visible instead of hiding turnover in target weights.
            for asset in ASSETS:
                shares = holdings[asset]
                if shares <= 0.0:
                    continue
                mid = _price(market, date, asset, "open")
                if not np.isfinite(mid):
                    raise ValueError(f"Missing open for sell {asset} {date.date()}")
                execution = mid * (1.0 - slippage)
                gross = shares * execution
                fee = gross * fee_rate
                cash += gross - fee
                trade_rows.append(
                    {
                        "date": date,
                        "action": "SELL",
                        "asset": asset,
                        "shares": shares,
                        "price": execution,
                        "gross_value": gross,
                        "fee": fee,
                        "slippage_bps": float(config["backtest"]["slippage_bps"]),
                        "cash_after": cash,
                        "reason": "rebalance_exit",
                    }
                )
                holdings[asset] = 0.0

            total_target = sum(target_values.values())
            if total_target > 0.0:
                investable_cash = cash
                for asset in ASSETS:
                    weight = target_values[asset]
                    if weight <= 0.0:
                        continue
                    # Preserve residual cash when target asset weights sum below
                    # one; never renormalize a cash-aware target to 100% invested.
                    allocation = investable_cash * weight
                    mid = _price(market, date, asset, "open")
                    if not np.isfinite(mid):
                        raise ValueError(f"Missing open for buy {asset} {date.date()}")
                    execution = mid * (1.0 + slippage)
                    notional = allocation / (1.0 + fee_rate)
                    fee = notional * fee_rate
                    shares = notional / execution
                    spend = notional + fee
                    if spend > cash + 1e-7:
                        raise ValueError("Cash constraint violated")
                    cash -= spend
                    holdings[asset] = shares
                    trade_rows.append(
                        {
                            "date": date,
                            "action": "BUY",
                            "asset": asset,
                            "shares": shares,
                            "price": execution,
                            "gross_value": notional,
                            "fee": fee,
                            "slippage_bps": float(config["backtest"]["slippage_bps"]),
                            "cash_after": cash,
                            "reason": "rebalance_entry",
                        }
                    )

        marked = cash
        for asset in ASSETS:
            close = _price(market, date, asset, "close")
            if not np.isfinite(close):
                raise ValueError(f"Missing close for {asset} {date.date()}")
            marked += holdings[asset] * close
        equity_rows.append({"date": date, "equity": marked, "cash": cash, **{f"{a}_shares": holdings[a] for a in ASSETS}})

    # Charge exit costs at the end of the evaluation horizon.
    final_date = dates[-1]
    for asset in ASSETS:
        shares = holdings[asset]
        if shares <= 0.0:
            continue
        close = _price(market, final_date, asset, "close")
        execution = close * (1.0 - slippage)
        gross = shares * execution
        fee = gross * fee_rate
        cash += gross - fee
        trade_rows.append(
            {
                "date": final_date,
                "action": "SELL",
                "asset": asset,
                "shares": shares,
                "price": execution,
                "gross_value": gross,
                "fee": fee,
                "slippage_bps": float(config["backtest"]["slippage_bps"]),
                "cash_after": cash,
                "reason": "final_liquidation",
            }
        )
        holdings[asset] = 0.0
    equity_rows[-1]["equity"] = cash
    equity_rows[-1]["cash"] = cash
    curve = pd.DataFrame(equity_rows).set_index("date")
    curve["returns"] = curve["equity"].pct_change()
    trades = pd.DataFrame(trade_rows)
    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"])
    selections = pd.DataFrame(target_rows)
    return curve, trades, selections
