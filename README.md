# BTC / ETH Long-Only ML Research

Research harness for a long-only, cash-aware Bitcoin and Ethereum spot-style
portfolio. It starts in cash, never shorts, never borrows, and evaluates every
candidate with one immutable transaction-cost and walk-forward protocol.

## Scope

- Universe: BTC and ETH only.
- Bars: daily UTC bars from `2018-01-01` through latest available bar.
- Price source: Yahoo Finance chart API, used because Binance is restricted in
  the execution environment. The source and fetch date are saved with every raw
  pull.
- On-chain source: Coin Metrics Community API. Metrics are joined as-of with a
  one-day availability lag and forward-filled for at most three days.
- Costs: 10 bps fee plus 5 bps slippage on every buy and sell.
- Execution: signals are formed at close and executed at the next bar open.
- Portfolio: non-negative BTC, ETH, and cash weights summing to one.
- Score: fixed `Research_score` in `crypto_research/metrics.py`; weights,
  thresholds, labels, costs, and evaluation windows are not tuned.

## Reproduce

```bash
bash measure.sh
```

Optional configuration/path arguments are passed through unchanged:

```bash
bash measure.sh --config config/research.json --artifact-dir .openresearch/artifacts
```

The first run creates a local `.venv`, downloads/cache raw data under
`data/raw/`, writes deterministic processed data under `data/processed/`, and
emits text artifacts under `.openresearch/artifacts/`.

## Evaluation partitions

- Development train: `2018-01-01` through `2021-12-31`.
- Validation: `2022-01-01` through `2023-12-31`, used only for Optuna decisions.
- Frozen test: `2024-01-01` through latest available bar.
- Primary score: full available evaluation period after data warm-up, using
  complete non-overlapping 126-session windows. Validation and test slices are
  reported separately for robustness.

Classical ML models fit only rows whose labels end strictly before each signal
date. Scalers fit on the same historical rows. Signals execute on the next bar,
so no close or on-chain observation from the future can enter a position.

## Research ladder

1. Stage 0: cash, buy-and-hold BTC/ETH, 50/50 buy-and-hold, SMA crossover,
   volatility-scaled momentum, and 12-1 momentum.
2. Stage 1: deterministic Ridge walk-forward model with lagged returns,
   volatility, range, volume, and lagged on-chain z-scores; optional Optuna
   tuning is validation-only.
3. Stage 2: causal autoregressive return forecast.
4. Stage 3: optional TimesFM and Kronos adapters. They remain disabled unless
   their public dependencies and checkpoints are available; unavailable
   foundation models are reported as blocked, never silently replaced by a
   different model.

OpenResearch uses the fixed `bash measure.sh` command for the baseline and all
children. Hyperparameters live in committed branch config files, not in the
run command.
