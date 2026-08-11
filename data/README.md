# Data layout

`data/raw/` contains fetched JSON payloads named with UTC fetch date. Each
payload has a sibling metadata JSON containing source URL, fetch timestamp,
requested range, and SHA-256 digest.

`data/processed/` contains deterministic tidy CSVs:

- `market_daily.csv`: UTC date, BTC/ETH OHLCV from Yahoo Finance.
- `onchain_daily.csv`: Coin Metrics Community daily metrics when available.
- `manifest.json`: source, row counts, date ranges, and alignment policy.

Current fair-comparison snapshot is pinned to market end `2026-08-10`.
Fair contract/configured Yahoo request end is `2026-08-11`; cached Yahoo
metadata records wider exclusive request end `2026-08-12` from snapshot
construction. SHA-256 values are recorded in
`research/DECISIONS.md` and answered fair-run artifacts. Processed snapshot
files are ignored from Git; the manifest and contract remain versioned.

Pipeline contract: `data/raw` cached pulls -> deterministic
`data/processed` CSVs -> causal feature table -> cost-aware labels ->
next-open long-only backtest. Fair runs use BTC and ETH only, start in cash,
and keep snapshot, split dates, costs, labels, embargo, and score fixed.

Features use only observations available at the signal close. On-chain rows are
lagged one day before an as-of backward join and forward-filled no more than
three days. A failed source must be recorded in `research/DECISIONS.md` before
using a replacement.
