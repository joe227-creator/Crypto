from __future__ import annotations

import pandas as pd

from crypto_research.features import build_feature_table


def test_onchain_alignment_uses_prior_day() -> None:
    dates = pd.date_range("2018-01-01", periods=70, freq="D")
    market = pd.concat(
        [
            pd.DataFrame({"date": dates, "asset": asset, "open": 10.0, "high": 11.0, "low": 9.0, "close": range(10, 80), "volume": 100.0})
            for asset in ["BTC", "ETH"]
        ],
        ignore_index=True,
    )
    onchain = pd.concat(
        [pd.DataFrame({"date": dates, "asset": asset, "AdrActCnt": range(1, 71)}) for asset in ["BTC", "ETH"]],
        ignore_index=True,
    )
    config = {"backtest": {"fee_bps": 10.0, "slippage_bps": 5.0}, "evaluation": {"embargo_sessions": 1}, "data": {"assets": ["BTC", "ETH"], "onchain_ffill_limit": 3}}
    features, columns = build_feature_table(market, onchain, config)
    btc = features[features["asset"].eq("BTC")].sort_values("date")
    row = btc[btc["date"].eq(pd.Timestamp("2018-02-15"))].iloc[0]
    assert "onchain_adractcnt_z" in columns
    assert row["label_end_date"] > row["date"]
