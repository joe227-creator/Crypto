from __future__ import annotations

import pandas as pd
import pytest

from crypto_research.backtest import run_backtest


def _market() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rows = {}
    for asset, base in [("BTC", 100.0), ("ETH", 50.0)]:
        for field, multiplier in [("open", 1.0), ("high", 1.01), ("low", 0.99), ("close", 1.0), ("volume", 1000.0)]:
            rows[f"{asset}_{field}"] = [base * multiplier * (1.0 + 0.01 * i) for i in range(4)]
    return pd.DataFrame(rows, index=dates)


def _config() -> dict:
    return {"backtest": {"initial_capital": 100000.0, "fee_bps": 10.0, "slippage_bps": 5.0, "max_weight_per_asset": 1.0}}


def test_backtest_charges_costs_and_stays_long_only() -> None:
    targets = pd.DataFrame({"BTC": [0.5], "ETH": [0.5]}, index=[pd.Timestamp("2020-01-02")])
    equity, trades, _ = run_backtest(_market(), targets, _config())
    assert (equity["equity"] > 0).all()
    assert (trades["fee"] >= 0).all()
    assert (trades["action"] == "BUY").sum() == 2


def test_backtest_rejects_short_target() -> None:
    targets = pd.DataFrame({"BTC": [-0.1], "ETH": [0.0]}, index=[pd.Timestamp("2020-01-02")])
    with pytest.raises(ValueError, match="long-only"):
        run_backtest(_market(), targets, _config())
