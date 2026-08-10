# Data layout

`data/raw/` contains fetched JSON payloads named with UTC fetch date. Each
payload has a sibling metadata JSON containing source URL, fetch timestamp,
requested range, and SHA-256 digest.

`data/processed/` contains deterministic tidy CSVs:

- `market_daily.csv`: UTC date, BTC/ETH OHLCV from Yahoo Finance.
- `onchain_daily.csv`: Coin Metrics Community daily metrics when available.
- `manifest.json`: source, row counts, date ranges, and alignment policy.

Features use only observations available at the signal close. On-chain rows are
lagged one day before an as-of backward join and forward-filled no more than
three days. A failed source must be recorded in `research/DECISIONS.md` before
using a replacement.
