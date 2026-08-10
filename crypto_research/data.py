"""Free public data acquisition and deterministic processing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


ASSET_SYMBOLS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
COINMETRICS_METRICS = ["AdrActCnt", "TxCnt", "FeeTotNtv", "HashRate"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _request_json(url: str) -> tuple[Any, bytes]:
    response = requests.get(url, headers={"User-Agent": "crypto-long-only-research/0.1"}, timeout=60)
    response.raise_for_status()
    raw = response.content
    return response.json(), raw


def _save_raw(raw_dir: Path, stem: str, url: str, payload: Any, raw: bytes, start: str, end: str) -> None:
    fetch_date = _utc_now().date().isoformat()
    data_path = raw_dir / f"{stem}_{fetch_date}.json"
    metadata_path = raw_dir / f"{stem}_{fetch_date}.metadata.json"
    _write_json(data_path, payload)
    _write_json(
        metadata_path,
        {
            "source_url": url,
            "fetched_at_utc": _utc_now().isoformat(),
            "fetch_date_utc": fetch_date,
            "requested_start": start,
            "requested_end": end,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )


def fetch_yahoo(asset: str, start: str, end: str | None, raw_dir: Path) -> pd.DataFrame:
    """Fetch one daily Yahoo series and preserve the raw response metadata."""
    if asset not in ASSET_SYMBOLS:
        raise ValueError(f"Unsupported asset: {asset}")
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_value = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp(_utc_now().date() + timedelta(days=2), tz="UTC")
    end_ts = int(end_value.timestamp())
    symbol = ASSET_SYMBOLS[asset]
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?period1={start_ts}&period2={end_ts}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )
    payload, raw = _request_json(url)
    _save_raw(raw_dir, f"yahoo_{asset.lower()}_daily", url, payload, raw, start, _date_text(end_value))
    result = (payload.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError(f"Yahoo returned no chart result for {asset}")
    chart = result[0]
    timestamps = chart.get("timestamp") or []
    quote = ((chart.get("indicators") or {}).get("quote") or [{}])[0]
    frame = pd.DataFrame({"date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()})
    for column in ["open", "high", "low", "close", "volume"]:
        values = quote.get(column) or []
        frame[column] = values
    frame["asset"] = asset
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["volume"] = frame["volume"].fillna(0.0)
    if frame.empty or frame["date"].min() > pd.Timestamp(start):
        raise RuntimeError(f"Yahoo data for {asset} does not cover requested start {start}")
    return frame[["date", "asset", "open", "high", "low", "close", "volume"]]


def fetch_coinmetrics(start: str, end: str | None, raw_dir: Path) -> pd.DataFrame:
    """Fetch daily BTC/ETH aggregates from the free Coin Metrics endpoint."""
    params = {
        "assets": "btc,eth",
        "metrics": ",".join(COINMETRICS_METRICS),
        "frequency": "1d",
        "start_time": start,
        "page_size": "10000",
    }
    if end:
        params["end_time"] = end
    base_url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    url = requests.Request("GET", base_url, params=params).prepare().url
    pages: list[dict[str, Any]] = []
    next_url: str | None = url
    raw_parts: list[bytes] = []
    for _ in range(32):
        if not next_url:
            break
        payload, raw = _request_json(next_url)
        pages.append(payload)
        raw_parts.append(raw)
        next_url = (payload.get("next_page_url") or {}).get("href") if isinstance(payload.get("next_page_url"), dict) else payload.get("next_page_url")
    if not pages:
        raise RuntimeError("Coin Metrics returned no pages")
    records: list[dict[str, Any]] = []
    for page in pages:
        records.extend(page.get("data") or [])
    if not records:
        raise RuntimeError("Coin Metrics returned no BTC/ETH records")
    combined = {"data": records, "pages": len(pages), "next_page_url": None}
    _save_raw(raw_dir, "coinmetrics_btc_eth_daily", url, combined, b"\n".join(raw_parts), start, end or _date_text(_utc_now()))
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_convert(None).dt.normalize()
    frame["asset"] = frame["asset"].str.upper()
    for metric in COINMETRICS_METRICS:
        if metric not in frame:
            frame[metric] = np.nan
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[frame["asset"].isin(["BTC", "ETH"])]
    frame = frame.sort_values(["date", "asset"]).drop_duplicates(["date", "asset"], keep="last")
    return frame[["date", "asset", *COINMETRICS_METRICS]]


def _validate_market(frame: pd.DataFrame, start: str) -> None:
    required = {"date", "asset", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Market columns missing: {required - set(frame.columns)}")
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate market date/asset rows")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive market prices")
    if frame["date"].min() > pd.Timestamp(start):
        raise ValueError("Market data starts after configured start")


def load_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load or build processed market and on-chain data."""
    data_cfg = config["data"]
    start = str(data_cfg["start_date"])
    end = data_cfg.get("end_date")
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    market_path = processed_dir / "market_daily.csv"
    onchain_path = processed_dir / "onchain_daily.csv"
    manifest_path = processed_dir / "manifest.json"
    refresh = bool(data_cfg.get("refresh", False))
    if market_path.exists() and onchain_path.exists() and manifest_path.exists() and not refresh:
        market = pd.read_csv(market_path, parse_dates=["date"])
        onchain = pd.read_csv(onchain_path, parse_dates=["date"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_market(market, start)
        return market, onchain, manifest

    market_frames = [fetch_yahoo(asset, start, end, raw_dir) for asset in data_cfg["assets"]]
    market = pd.concat(market_frames, ignore_index=True).sort_values(["date", "asset"])
    _validate_market(market, start)
    onchain = fetch_coinmetrics(start, end, raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    market.to_csv(market_path, index=False, date_format="%Y-%m-%d")
    onchain.to_csv(onchain_path, index=False, date_format="%Y-%m-%d")
    manifest = {
        "built_at_utc": _utc_now().isoformat(),
        "market_source": data_cfg["price_source"],
        "onchain_source": data_cfg["onchain_source"],
        "assets": list(data_cfg["assets"]),
        "requested_start": start,
        "market_start": _date_text(market["date"].min()),
        "market_end": _date_text(market["date"].max()),
        "market_rows": int(len(market)),
        "onchain_start": _date_text(onchain["date"].min()),
        "onchain_end": _date_text(onchain["date"].max()),
        "onchain_rows": int(len(onchain)),
        "onchain_alignment": "availability date is source date plus one UTC day; backward as-of join",
        "onchain_forward_fill_limit_days": int(data_cfg.get("onchain_ffill_limit", 3)),
    }
    _write_json(manifest_path, manifest)
    return market, onchain, manifest
