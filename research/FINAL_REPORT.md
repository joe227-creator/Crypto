# BTC ETH Long-Only ML Research

**Date:** `2026-08-11`
**Repository:** `https://github.com/joe227-creator/Crypto`  
**OpenResearch server project:** `019fec29-be2c-7d20-b098-70b35fefda7d`
**OpenResearch local project:** `528668ac-3d0c-414f-ab6f-5316030509e9`
**Status:** balanced temporal contract completed; cost-aware threshold Ridge
promoted as research candidate; no deployment

## Objective

Build and research a Bitcoin/Ethereum-only, long-only, cash-aware ML trading
system. Start state is 100% cash. Portfolio weights are non-negative and sum to
one across BTC, ETH, and cash. No shorting, leverage, margin, borrowing, or
perpetual trade legs.

## Fixed Evaluation Protocol

- Data range: `2018-01-01` inclusive through pinned market bar `2026-08-10`.
  Fair contract/configured Yahoo request end is `2026-08-11`; cached metadata
  records wider exclusive request end `2026-08-12` from snapshot construction.
- Price source: Yahoo Finance daily `BTC-USD` and `ETH-USD`; Binance returned
  HTTP 451 from WSL and was documented as unavailable.
- On-chain source: Coin Metrics Community daily BTC/ETH metrics where published:
  active addresses, transaction count, native fees, and hash rate.
- Alignment: source observations receive one UTC-day availability lag, backward
  as-of join, forward-fill limited to three days.
- Signal timing: form at daily close, execute target at next daily open.
- Costs: 10 bps fee plus 5 bps slippage per side. Costs never set to zero.
- Signal threshold: fixed at `0.0`; Optuna tunes only branch-declared model
  parameters on validation data.
- Labels: five-session forward simple return less fixed round-trip cost for
  classical models. Labels are used only when label end date is before the
  signal date minus one-session embargo.
- Splits: training/in-sample `2018-01-01` through `2022-12-31`; validation
  `2023-01-01` through `2024-12-31`; frozen test `2025-01-01` through
  `2026-08-10`; minimum train history 365 days; refit cadence 21 sessions.
- Validation is selection-only. Frozen test is primary `Research_score` and is
  not used by Optuna or model decisions.
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

- WSL2 CPU-only local validation; local OpenResearch runner uses WSL ext4 clone
  `/home/user/.cache/openresearch/repos/joe227-creator/crypto-local`.
- Current main smoke artifacts are ignored generated outputs; their commit field
  identifies HEAD at generation time and is not full-source or fair-run
  provenance.
- Balanced contract reference commit: `572a211` on the answered control branch.
- Root initial local reproduction commit:
  `5b3340fdfd1e6b4d7ec0092c4c7a29ff6fda278c`.
- Data manifest snapshot: `data/processed/manifest.json`, built 2026-08-10;
  market rows 6,288, on-chain rows 6,286, market end 2026-08-10.
- Exact command: `bash measure.sh`.
- Tests: `python -m pytest -q` returned `8 passed` after temporal-contract
  changes. Fair-run branches passed JSON/Python syntax and artifact validation.
- Baseline command also emits `.openresearch/artifacts/EVAL.md`, metrics JSON,
  equity curve, trade log, 126-session windows, signal/execution audit, seed
  results, resolved config, and dataset manifest. The checked-out
  `.openresearch/artifacts/EVAL.md` and `metrics.json` are ignored local smoke
  artifacts. Their deterministic control metrics may match the fair control,
  but they are not archived fair-anchor artifacts.
  Answered fair-run artifacts are stored under
  `/home/user/.local/share/openresearch/local-runs/<run-id>/repo/.openresearch/artifacts`.

Raw downloads remain cached locally under `data/raw/` with fetch-date metadata,
source URL, requested range, and SHA-256. Generated data and run artifacts are
ignored from Git to avoid committing mutable snapshots; manifest and commands
remain versioned.

## Full-History Stage 0 Reference Results

These are direct local full-history results from the current main smoke. They
are not fair primary test results and are not orx run results.

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

Historical root `3b0fb31c-6fed-4c8d-9b26-0160fa56e4d3` and its old-split
children remain frozen historical evidence only. Fair tree:

```text
Fair snapshot anchor `51b44ade-505c-44fc-8ca2-adb724bb9a63`
└── Temporal split anchor `ec6bed56-2490-4113-b2fb-981e9bbe06a5`
    └── Balanced temporal contract `b405429f-01aa-4f18-9492-005bb3ab17b4`
        ├── Technical Ridge
        ├── Ridge plus lagged on-chain
        ├── ARIMA
        ├── ETS
        ├── Cross-asset Ridge
        ├── Pooled Ridge
        ├── EMA Ridge
        ├── Top-1 Ridge
        ├── Hysteresis Ridge
        ├── Volatility-gated Ridge
        └── Cost-aware threshold Ridge [promoted research candidate]
            ├── TimesFM 2.5 zero-shot [rejected]
            ├── Kronos base zero-shot [rejected]
             ├── TimesFM-to-Ridge hybrid [rejected]
-            └── TimesFM uncertainty gate [rejected]
+            ├── TimesFM uncertainty gate [rejected]
+            ├── XGBoost cost-aware [rejected]
+            └── LightGBM cost-aware [rejected]
```

Historical server planning and smoke nodes:

| Experiment | ID | Branch | Commit | Orx run |
|---|---|---|---|---|
| Stage 0 Baseline | `019fec2a-3c90-74f8-a1c6-fddf122b3d4c` | `orx/stage-0-baseline-30c6d37a` | `d5cf46b` | none |
| Stage 1 Ridge technical features | `019fec2f-a592-70ab-9915-1612dbfb4bbf` | `orx/stage-1-ridge-technical-features-104f71c6` | `9dae8cd` | none |
| Stage 1 Ridge plus on-chain | `019fec2f-aa59-7ead-be05-a8b7b6dc81c0` | `orx/stage-1-ridge-plus-on-chain-e5162886` | `13ddaf3` | none |

The local root and all children inherit fixed `bash measure.sh`. Fair children
vary one model-family, feature, or portfolio-mapping decision and run eight
sequential validation-only Optuna trials where applicable. Answered branches are
frozen. No fair child became parent for a stacked bush.

Fair snapshot, split, control, and sibling nodes:

| Experiment | ID | Parent | Code commit | Orx run |
|---|---|---|---|---|
| Fair snapshot anchor | `51b44ade-505c-44fc-8ca2-adb724bb9a63` | none | snapshot anchor | none |
| Temporal split anchor | `ec6bed56-2490-4113-b2fb-981e9bbe06a5` | `51b44ade-505c-44fc-8ca2-adb724bb9a63` | temporal anchor | none |
| Balanced 50/50 control | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `ec6bed56-2490-4113-b2fb-981e9bbe06a5` | `572a211` | `9ec6329e-3c8b-409f-95f1-f0ebe9794e16` |
| Technical Ridge | `a903c343-b32b-47ff-a19e-1b4050d7f793` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `ceb9f33` | `5b151771-3dae-4439-9dac-428a44ede82f` |
| Ridge plus on-chain | `c2c974b4-abe1-4953-8efb-feaa477cdb18` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `3078fba` | `8cb61076-5cac-4538-a91e-cc4a1f5e880f` |
| ARIMA | `a8eee72f-942d-4ce9-8461-443638d01197` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `05bb40e` | `9eb8a8de-5258-4b08-a082-2a890961f6f9` |
| ETS | `61b47e35-bc71-4d8c-b793-38bbfd654387` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `990021b` | `405a267c-7509-452e-9bdb-22b18a180bab` |
| Cross-asset Ridge | `ac10fa19-d0e3-4376-bd36-97431b5368f6` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `a6ba973` | `c27c2db6-c8bb-43b8-bd80-7361f43ef68f` |
| Pooled Ridge | `4a0eb7f8-bb3c-4542-ad31-71a9cbd9e3af` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `eaa3651` | `bf7df15e-9f45-431f-a6a7-2029f20bccf4` |
| EMA Ridge | `f92e75e0-a0f8-483c-9981-f7526afc7ac6` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `363e381` | `f79e767b-13c8-40b3-9eb8-943aac42e3fc` |
| Top-1 Ridge | `335af757-cb6d-429a-a0d5-be9df4b2ee40` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `990d1b2` | `505056ac-5b31-4ed2-8c38-29add2e4cf99` |
| Hysteresis Ridge | `8d6e2d4d-719a-4ddb-87fb-9f27f2d36316` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `dcd263f` | `f14e7395-4126-4a0f-9ecb-35f87ff101a4` |
| Volatility-gated Ridge | `d2669976-1668-429c-acf4-a6b16f6d188a` | `b405429f-01aa-4f18-9492-005bb3ab17b4` | `8e93d14` | `140323a1-1d02-4081-9bf9-00a3537fb07b` |

## Historical Local Results

Earlier local runs used a different validation/test boundary and are retained for
history only. They must not be compared directly with fair-contract scores.
Historical root score was `0.1113547`; old technical Ridge, on-chain Ridge,
ARIMA, and ETS scores were `-1.1859575`, `-1.0763066`, `-0.8241698`, and
`-1.1998268` respectively.

## Fair-Contract Results

All rows below use pinned snapshot hashes, balanced dates, fixed execution/cost
protocol, and local tracked runs. Each emitted `EVAL.md`, metrics, equity, trade,
window, audit, seed, config, and data-manifest text artifacts. Test score is
primary; validation score is selection-only.

| Node | Run | Validation | Frozen test | Turnover | Selected parameter |
|---|---|---:|---:|---:|---|
| 50/50 control | `9ec6329e-3c8b-409f-95f1-f0ebe9794e16` | 0.5314679 | -0.2189362 | 0.0000 | none |
| Technical Ridge | `5b151771-3dae-4439-9dac-428a44ede82f` | -0.7239327 | -0.9475211 | 5.3080 | alpha 89.1666 |
| Ridge plus on-chain | `8cb61076-5cac-4538-a91e-cc4a1f5e880f` | -0.1714904 | -0.6419978 | 3.9480 | alpha 44.3466 |
| ARIMA | `9eb8a8de-5258-4b08-a082-2a890961f6f9` | 0.5506139 | -0.7781348 | 4.7139 | order 2 |
| ETS | `405a267c-7509-452e-9bdb-22b18a180bab` | 1.0205072 | -0.1819630 | 0.8950 | alpha 0.5053 |
| Cross-asset Ridge | `c27c2db6-c8bb-43b8-bd80-7361f43ef68f` | -0.9148506 | -1.2068932 | 7.4232 | alpha 80.6384 |
| Pooled Ridge | `bf7df15e-9f45-431f-a6a7-2029f20bccf4` | -0.6372513 | -0.9904385 | 5.1874 | alpha 0.0754 |
| EMA Ridge | `f79e767b-13c8-40b3-9eb8-943aac42e3fc` | 0.1572936 | -0.4812781 | 1.9593 | alpha 1.7100, span 17 |
| Top-1 Ridge | `505056ac-5b31-4ed2-8c38-29add2e4cf99` | -0.4045567 | -0.7593821 | 4.3029 | alpha 89.1666 |
| Hysteresis Ridge | `f14e7395-4126-4a0f-9ecb-35f87ff101a4` | 0.0617274 | -0.3176550 | 2.1506 | alpha 0.8263, band 0.00988 |
| Volatility-gated Ridge | `140323a1-1d02-4081-9bf9-00a3537fb07b` | -0.6569587 | -0.7909709 | 4.7239 | alpha 80.6384, threshold 0.7222 |
| Cost-aware threshold Ridge | local `ba9dab5` / branch `dc031c4` | 0.1135487 | 0.2132863 | 1.4794 | alpha 0.1527, threshold 0.02682 |
| TimesFM 2.5 zero-shot | local `84484e5` | 0.1488174 | -0.4697066 | 3.3396 | context 256, threshold 0.01438 |
| Kronos base zero-shot | local `84484e5` | -0.7314223 | -0.8563088 | 5.1900 | context 256, threshold 0.02735 |
| TimesFM-to-Ridge hybrid | local Stage 4 child | 0.0335998 | -0.0728067 | 2.1762 | TimesFM feature, alpha 0.1527, threshold 0.02682 |
| TimesFM uncertainty gate | local Stage 4 child | 0.2775677 | -0.5483675 | 2.5748 | max uncertainty 0.4808, threshold 0.01675 |
| XGBoost cost-aware | local Stage 1 tree child | -0.3681458 | -0.4176403 | 3.0949 | n=435, depth=3, lr=0.017, threshold=0.0196 |
| LightGBM cost-aware | local Stage 1 tree child | -0.4197973 | -0.4602400 | 2.8373 | n=473, depth=5, lr=0.011, threshold=0.0223 |

Cost-aware threshold Ridge beats fair control score `-0.2189362`, ETH
buy-and-hold `-0.0746825`, and all prior fair siblings. Test mean rolling return
is `0.0375513`, maximum drawdown `-0.0364`, Sharpe `0.7491`, turnover `1.4794`,
and turnover penalty `0.0480`. Promote as research parent only. Four test
windows contain one positive window and three zero-return windows; seed spread is
zero because implementation is deterministic. Threshold sensitivity is material
and formal multiple-testing adjustment remains missing. No deployment.

Follow-up directions were screened locally under same contract: prediction
clipping test `0.1946880`, partial adjustment `0.1323719`, pooled Ridge
`0.1021681`, cash overlay `0.1197785`, hysteresis `0.1946880`, volatility gates
`-0.4211` or lower, and AR threshold `-0.8077`. None beats promoted threshold
Ridge on frozen test with stronger robustness evidence.

Foundation follow-ups were then tested from promoted threshold parent using
repository-local weights. TimesFM 2.5 scored validation `0.1488174` but frozen
test `-0.4697066`; Kronos base scored validation `-0.7314223` and frozen test
`-0.8563088`. Both were rejected after costs.

Stage 4 TimesFM composition was tested next. A causal TimesFM forecast feature
into Ridge scored validation `0.0335998` and frozen test `-0.0728067`. A
TimesFM uncertainty abstention gate scored validation `0.2775677` and frozen
test `-0.5483675`, with high test drawdown and negative Sharpe. Both lost to
promoted threshold Ridge and were rejected.

XGBoost and LightGBM gradient boosting models were also tested under the same
causal walk-forward contract with identical technical features and labels,
motivated by arXiv:2606.00060, which documents XGBoost outperforming neural
alternatives on hourly BTC data. XGBoost scored validation `-0.3681458`,
frozen test `-0.4176403`, with turnover `3.0949` and Sharpe `-0.3669`.
LightGBM scored validation `-0.4197973`, frozen test `-0.4602400`, with
turnover `2.8373` and Sharpe `-0.5272`. Both were heavily rejected against
the promoted threshold Ridge `0.2132863`. Linear Ridge with cost-aware
threshold remains dominant.

## Historical / Pre-Balanced Stage 2 Screening

Stage 2 literature review used arXiv:2602.10785, which emphasizes double
out-of-sample walk-forward selection with conservative fees, and
arXiv:2606.27100, which cautions that forecast accuracy gains in noisy financial
returns do not establish economic alpha. Existing causal recursive AR code was
screened locally before creating a tree node:

| Horizon | Full score | Validation | Test | Turnover | Verdict |
|---:|---:|---:|---:|---:|---|
| 1 | -2.6286336 | -1.1072398 | -1.5539944 | 25.7258 | reject |
| 5 | -1.7015701 | -0.6832194 | -0.8427689 | 18.3135 | reject |
| 21 | -0.7378930 | -0.4975945 | -0.4542211 | 8.3825 | least bad, reject |

AR horizon 21 is still below historical baseline `0.1115835`. This screening
used the pre-balanced contract and is not fair-comparable. After the balanced
control answered, ARIMA and ETS were tested as fair siblings under that confirmed
parent; both lost, so no deeper stacked-bush node was created.

## Robustness And Leakage

- All fair runs used configured seeds `11, 22, 33, 44, 55`; seed score spread
  was zero because current forecasters are deterministic, so repeated seed rows
  are not independent stochastic evidence.
- Fair validation and test slices report 5 and 4 complete 126-session windows.
- Promoted threshold Ridge has only one positive frozen-test window out of four;
  sparse-window concentration blocks production promotion.
- Regime slices, parameter sensitivity beyond selected Optuna values, feature
  importance stability, and formal Deflated Sharpe or equivalent multiple-testing
  adjustment remain unmeasured. These are explicit promotion gaps, not positive
  evidence.
- Leakage guards present: next-open execution, as-of on-chain join with lag,
  train-only feature scaling, label-end embargo, causal expanding fits, and
  signal/execution audit artifact.
- Regression checks cover score formula, incomplete windows, positive equity,
  fee charging, short rejection, cash-sleeve preservation, on-chain date
  alignment, and session-based embargo across weekends.

## Foundation Models

TimesFM and Kronos research was reviewed through `orx lit`, `orx paper`, and
official repositories. Source repositories are cloned under ignored
`external/timesfm` and `external/kronos`; weights are copied under ignored
repository-local `models/` paths. TimesFM 2.5 zero-shot validation score was
`0.1488174`, but frozen-test score was `-0.4697066`, with mean test rolling
return `-0.0628`, Sharpe `-0.5150`, and turnover `3.3396`. Kronos base zero-shot
validation score was `-0.7314223`, frozen-test score `-0.8563088`, mean test
rolling return `-0.1397`, Sharpe `-0.7259`, and turnover `5.1900`. Both were
rejected against promoted threshold Ridge. Stage 3 foundation bush and first
Stage 4 hybrid bush are complete and rejected; deeper LSTM/TCN/Transformer
architectures remain unstarted.

## Blocker And Recommendation

Managed CPU launch on server project still returns HTTP 402 `billing_required` /
`Out of credits` before provisioning. Local `orx up` project provides a working
CPU backend. Balanced anchor, ten fair model siblings, and local follow-up
directions answered. Cost-aware threshold Ridge is promoted as research parent;
robustness gaps still block deployment.

Recommendation: **promote cost-aware threshold Ridge for further research,
paper-trade nothing**. Do not deploy threshold Ridge or any fair ARIMA, ETS,
cross-asset, or allocation variant. Next valid bush requires independent
stochastic seeds, regime windows, threshold sensitivity, and formal
multiple-testing adjustment.

## Exact Next Commands

```bash
# Local project and answered root:
orx projects
orx runs 528668ac-3d0c-414f-ab6f-5316030509e9
# Continue only with promoted threshold-Ridge robustness checks or a new
# literature-backed question. Never edit answered branches.
```

Re-read all terminal runs after each wait tick. Analyze `EVAL.md` before
creating or promoting next bush. Never edit answered node; never change
`bash measure.sh`, score, costs, labels, splits, or thresholds.
