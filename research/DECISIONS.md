# Research Decisions

## Data source fallback

Binance public klines returned HTTP 451 from the WSL execution environment.
Yahoo Finance daily chart API is reachable and is used as the documented free
price fallback. Coin Metrics Community API is reachable for daily BTC/ETH
on-chain aggregates. Raw response metadata records URL and fetch date.

## Fixed score and execution

The score follows the supplied formula exactly. The implementation preserves
126-session non-overlapping windows, 252-session annualized Sharpe, and a fixed
baseline turnover reference computed from 50/50 BTC/ETH buy-and-hold. Signals
form at close and execute at next open with 10 bps fees and 5 bps slippage per
side. No model or Optuna trial can change these values.
