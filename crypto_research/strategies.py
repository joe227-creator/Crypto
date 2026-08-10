"""Stage 0 rules and model-to-portfolio mappings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import walk_forward_ar, walk_forward_ridge


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
        raw = _equal_positive(predictions, threshold=FIXED_SIGNAL_THRESHOLD)
    elif mode == "ar":
        predictions = walk_forward_ar(features, dates, config, horizon=int(params.get("horizon", 5)))
        raw = _equal_positive(predictions, threshold=FIXED_SIGNAL_THRESHOLD)
    else:
        raise ValueError(f"Unknown strategy mode: {mode}")
    targets = _shift_to_next_open(raw, dates)
    audit = targets.reset_index().rename(columns={"date": "execution_date"})
    if not audit.empty and "signal_date" in audit:
        audit["signal_date"] = pd.to_datetime(audit["signal_date"])
        if not (audit["signal_date"] < audit["execution_date"]).all():
            raise ValueError("Signal executed on same or earlier bar")
    return targets, audit
