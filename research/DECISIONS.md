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

Signal threshold is fixed at `0.0`. Optuna tunes Ridge regularization only;
threshold, labels, costs, score weights, and evaluation partitions remain fixed.

Dependency bounds keep pandas below version 3 because pandas 3 changes default
datetime resolution and can make `merge_asof` reject otherwise identical daily
keys. Feature join keys are explicitly normalized to `datetime64[ns]` as a
second guard.

## OpenResearch execution blocker

Project `019fec29-be2c-7d20-b098-70b35fefda7d` was created and its root command
was set exactly once to `bash measure.sh`. Managed CPU launch returned HTTP 402
`billing_required` / `Out of credits` before provisioning. Local backend is
available only for projects created by `orx up`, and this server project cannot
be converted by the installed CLI. Therefore no orx run ID, EVAL artifact, or
scientific promotion exists yet. Direct local `bash measure.sh` output is kept
as supporting reproducibility evidence only and is never treated as an orx
experiment answer.

## First literature bush

Before Stage 1 branches, `orx lit` and arXiv evidence were reviewed. Bysik and
Slepaczuk, arXiv:2606.00060, report that naive sign trading collapses after
10-bps costs and that walk-forward, cost-aware execution dominates architectural
claims. Fang and Slepaczuk, arXiv:2606.09478, report weak state-dependent return
predictability and value in volatility/regime variables, with implementation
controls required. The first bush therefore varies only feature information:
technical Ridge versus technical-plus-lagged-on-chain Ridge. No branch changes
labels, costs, thresholds in the fixed score, split dates, or run command.
