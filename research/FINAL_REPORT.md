# BTC ETH Long-Only ML Research

**Date:** 2026-08-10  
**Repository:** `https://github.com/joe227-creator/Crypto`  
**OpenResearch project:** `019fec29-be2c-7d20-b098-70b35fefda7d`  
**Status:** protocol and Stage 0 complete locally; orx execution blocked before first run by billing

## Objective

Build and research a Bitcoin/Ethereum-only, long-only, cash-aware ML trading
system. Start state is 100% cash. Portfolio weights are non-negative and sum to
one across BTC, ETH, and cash. No shorting, leverage, margin, borrowing, or
perpetual trade legs.

## Fixed Evaluation Protocol

- Data range: `2018-01-01` inclusive through latest available daily bar.
- Price source: Yahoo Finance daily `BTC-USD` and `ETH-USD`; Binance returned
  HTTP 451 from WSL and was documented as unavailable.
- On-chain source: Coin Metrics Community daily BTC/ETH metrics where published:
  active addresses, transaction count, native fees, and hash rate.
- Alignment: source observations receive one UTC-day availability lag, backward
  as-of join, forward-fill limited to three days.
- Signal timing: form at daily close, execute target at next daily open.
- Costs: 10 bps fee plus 5 bps slippage per side. Costs never set to zero.
- Signal threshold: fixed at `0.0`; Optuna tunes Ridge regularization only.
- Labels: five-session forward simple return less fixed round-trip cost for
  classical models. Labels are used only when label end date is before the
  signal date minus one-session embargo.
- Splits: expanding train from `2018-01-01`; validation `2022-01-01` to
  `2023-12-31`; frozen test `2024-01-01` onward; minimum train history 365 days;
  refit cadence 21 sessions.
- Primary windows: 126-session non-overlapping windows, complete windows only.
- Sharpe: annualized using 252 sessions and zero annual risk-free rate.
- Baseline turnover: 50/50 BTC/ETH buy-and-hold mean monthly BUY notional over
  initial equity, `0.9990010`.
- Score: unchanged supplied formula in `crypto_research/metrics.py`.

```text
Research_score = mean_rolling_6m_return
               + 0.20 * return_on_risk
               + 0.10 * win_rate
               - 0.35 * max(0, -0.50 - maximum_drawdown)
               - 0.15 * max(0, 0.80 - Sharpe)
               - 0.10 * max(0, turnover - baseline_turnover)
```

## Environment And Reproduction

- WSL2 Ubuntu 22.04, CPU-only local validation.
- Main implementation commit: `0202dd4f295d09e883e7614443a9da5825f052cc`.
- Root initial local reproduction commit: `5b3340fdfd1e6b4d7ec0092c4c7a29ff6fda278c`.
- Data manifest snapshot: `data/processed/manifest.json`, built 2026-08-10;
  market rows 6,288, on-chain rows 6,286, market end 2026-08-10.
- Exact command: `bash measure.sh`.
- Tests: `python -m pytest -q` returned `7 passed`.
- Baseline command also emits `.openresearch/artifacts/EVAL.md`, metrics JSON,
  equity curve, trade log, 126-session windows, signal/execution audit, seed
  results, resolved config, and dataset manifest.

Raw downloads remain cached locally under `data/raw/` with fetch-date metadata,
source URL, requested range, and SHA-256. Generated data and run artifacts are
ignored from Git to avoid committing mutable snapshots; manifest and commands
remain versioned.

## Stage 0 Local Results

These are direct local fixed-contract results. They are not orx run results.

| Strategy | Research_score | Max DD | Sharpe | Turnover | Windows |
|---|---:|---:|---:|---:|---:|
| Cash | -0.1200000 | 0.0000 | 0.0000 | 0.0000 | 24 |
| Buy-and-hold BTC | 0.1275554 | -0.8153 | 0.5035 | 0.9990 | 24 |
| Buy-and-hold ETH | 0.1196255 | -0.9396 | 0.4548 | 0.9990 | 24 |
| Buy-and-hold BTC/ETH 50/50 | 0.1115835 | -0.8788 | 0.4702 | 0.9990 | 24 |
| SMA cross | -0.8565624 | -0.7505 | 0.3770 | 9.6819 | 24 |
| Volatility-scaled | -1.2232821 | -0.9499 | -0.1148 | 10.5541 | 24 |
| Momentum 12-1 | -2.5001832 | -0.5994 | 0.6109 | 28.3356 | 24 |

No Stage 0 rule beats BTC buy-and-hold under fixed cost and turnover penalty.
Buy-and-hold 50/50 remains frozen score reference for `baseline_turnover`, not
an assertion that it is best deployable portfolio.

## Experiment Tree

```text
Stage 0 Baseline
├── Stage 1 Ridge technical features
└── Stage 1 Ridge plus on-chain
```

OpenResearch IDs:

| Experiment | ID | Branch | Commit | Orx run |
|---|---|---|---|---|
| Stage 0 Baseline | `019fec2a-3c90-74f8-a1c6-fddf122b3d4c` | `orx/stage-0-baseline-30c6d37a` | root branch pending sync | none |
| Stage 1 Ridge technical features | `019fec2f-a592-70ab-9915-1612dbfb4bbf` | `orx/stage-1-ridge-technical-features-104f71c6` | `68f70f9` | none |
| Stage 1 Ridge plus on-chain | `019fec2f-aa59-7ead-be05-a8b7b6dc81c0` | `orx/stage-1-ridge-plus-on-chain-e5162886` | `2fe5010` | none |

All nodes inherit fixed `bash measure.sh`. Stage 1 children vary only feature
information and run eight sequential Optuna trials on validation data.

## Stage 1 Local Smoke Evidence

Smoke values validate code and artifacts only. They do not answer orx nodes and
cannot trigger promotion.

| Direction | Full score | Validation score | Test score | Turnover | Optuna best |
|---|---:|---:|---:|---:|---|
| Ridge technical only | invalidated | invalidated | invalidated | invalidated | threshold tuning was fixed-protocol drift |
| Ridge plus lagged on-chain | pending recomputation | pending recomputation | pending recomputation | pending recomputation | alpha-only Optuna after threshold lock |

Prior smoke values were invalidated after detecting threshold tuning against the
fixed protocol. Recompute both siblings with zero entry threshold before any
comparison. Neither has orx evidence or robustness audit. No promotion made.

## Robustness And Leakage

- Seeds configured: `11, 22, 33, 44, 55`. Stage 0 is deterministic, so seed
  spread is zero. Stage 1 smoke currently repeats deterministic strategy output;
  full multi-seed training/evaluation remains pending orx execution.
- Validation and test slices report complete 126-session windows: validation 5,
  test 7 for baseline and Stage 1 smoke.
- Regime slices, parameter sensitivity around Optuna optima, per-seed curves,
  feature importance stability, and formal walk-forward promotion audit remain
  unmeasured because no orx run reached execution.
- Leakage guards present: next-open execution, as-of on-chain join with lag,
  train-only feature scaling, label-end embargo, causal expanding fits, and
  signal/execution audit artifact.
- Regression checks cover score formula, incomplete windows, positive equity,
  fee charging, short rejection, cash-sleeve preservation, and on-chain date
  alignment.

## Foundation Models

TimesFM and Kronos research was reviewed through `orx lit`, `orx paper`, and
official repositories. TimesFM public package/checkpoint path and PyTorch are
not installed in current CPU environment. Kronos official repository and public
model zoo are documented, but PyTorch is also unavailable. The harness reports
`blocked_dependency` and fails loudly for `timesfm`/`kronos`; it never silently
replaces a foundation model with another model. Stage 3 remains open, not
rejected.

## Blocker And Recommendation

Managed CPU launch of root returned HTTP 402 `billing_required` / `Out of
credits` before provisioning. Installed `orx` supports local backend only for
projects created by `orx up`; current server project cannot be converted by CLI.
Therefore no orx node has answered, no child is scientifically promoted, and
no final model recommendation is valid.

Recommendation: **keep Stage 0 reference and paper-trade nothing** until OpenResearch
credits are restored. Then run root and siblings sequentially, collect full
5-10 seed and regime evidence, and only promote a candidate that beats frozen
reference on robustness-adjusted score. Do not deploy current Ridge smoke
variants.

## Exact Next Commands

```bash
# Restore OpenResearch credits first, then from repository root:
wsl.exe -- bash -lc 'orx exp run 019fec2a-3c90-74f8-a1c6-fddf122b3d4c --cpu cpu5c --vcpus 8'
wsl.exe -- bash -lc 'orx exp wait --project 019fec29-be2c-7d20-b098-70b35fefda7d'
wsl.exe -- bash -lc 'orx runs 019fec29-be2c-7d20-b098-70b35fefda7d'
```

Re-read all terminal runs after each wait tick. Analyze `EVAL.md` before
creating or promoting next bush. Never edit answered node; never change
`bash measure.sh`, score, costs, labels, splits, or thresholds.
