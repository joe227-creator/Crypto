# BTC / ETH Long-Only ML Research

Research harness for a long-only, cash-aware Bitcoin and Ethereum spot-style
portfolio. It starts in cash, never shorts, never borrows, and evaluates every
candidate with one immutable transaction-cost and walk-forward protocol.

## Scope

- Universe: BTC and ETH only.
- Bars: daily UTC bars from `2018-01-01` through pinned market bar
  `2026-08-10`. Fair contract/configured Yahoo request end is `2026-08-11`
  because its end bound is exclusive. Cached raw metadata records a wider
  `2026-08-12` request from snapshot construction; processed snapshot ends at
  `2026-08-10`.
- Price source: Yahoo Finance chart API, used because Binance is restricted in
  the execution environment. The source and fetch date are saved with every raw
  pull.
- On-chain source: Coin Metrics Community API. Metrics are joined as-of with a
  one-day availability lag and forward-filled for at most three days.
- Costs: 10 bps fee plus 5 bps slippage on every buy and sell.
- Execution: signals are formed at close and executed at the next bar open.
- Portfolio: non-negative BTC, ETH, and cash weights summing to one.
- Score: fixed `Research_score` implemented by `research_score()` in
  `crypto_research/metrics.py`; weights,
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

- Training/in-sample: `2018-01-01` through `2022-12-31`.
- Validation: `2023-01-01` through `2024-12-31`, used only for Optuna and model
  selection decisions.
- Frozen test: `2025-01-01` through `2026-08-10`; configured evaluation/request
  end bound is `2026-08-11`. Cached Yahoo pulls used wider exclusive
  `2026-08-12` and returned no bar after `2026-08-10`.
- Primary `Research_score`: frozen test only, using complete non-overlapping
  126-session windows. Full-history, validation, and test slices are retained
  separately for robustness.

Classical ML models fit only rows whose labels end strictly before each signal
date. Scalers fit on the same historical rows. Signals execute on the next bar,
so no close or on-chain observation from the future can enter a position.

## Research Status

- Fair contract: training `2018-01-01..2022-12-31`, validation
  `2023-01-01..2024-12-31`, frozen test `2025-01-01..2026-08-10`.
- Pinned snapshot hashes and contract rationale: `research/DECISIONS.md` and
  `research/ADR-001-balanced-temporal-contract.md`.
- Fair control test score: `-0.2189362`. Cost-aware threshold Ridge is
  promoted as research candidate: test `0.2132863`, validation `0.1135487`.
  Research promotion is not deployment approval; retain controls and
  paper-trade nothing.
- Full experiment tree, run IDs, metrics, risks, and stop decision:
  `research/FINAL_REPORT.md`.
- Chronological run ledger: `research/experiment_ledger.csv`.
- Stage 4 TimesFM hybrid and uncertainty-gate children were tested and rejected;
  deeper architectures remain unstarted.
- Local Stage 3 assets: `models/README.md`; adapters load repository-local
  TimesFM 2.5 and Kronos base/tokenizer files, isolated from shared caches.

## Research ladder

1. Stage 0: cash, buy-and-hold BTC/ETH, 50/50 buy-and-hold, SMA crossover,
   volatility-scaled momentum, and 12-1 momentum.
2. Stage 1: deterministic Ridge walk-forward model with fixed zero signal
   threshold, lagged returns,
   volatility, range, volume, and lagged on-chain z-scores; optional Optuna
   tuning is validation-only; branch-specific parameters never enter frozen
   test selection.
3. Stage 2: causal autoregressive return forecast.
4. Stage 3: TimesFM 2.5 and Kronos zero-shot adapters. Local weights and source
   repositories are present; experiments remain sequential and never silently
   replace a foundation model with a different model.

OpenResearch uses the fixed `bash measure.sh` command for the baseline and all
children. Hyperparameters live in committed branch config files, not in the
run command.
