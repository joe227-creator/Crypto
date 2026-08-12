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

Base signal threshold is fixed at `0.0`. Branches with an explicit cost-aware
filter declare threshold as tunable; Optuna tunes only those branch-declared
parameters on validation. Labels, costs, score weights, and evaluation
partitions remain fixed.

Embargo is session-based, not calendar-day-based: one-session purge excludes
the immediately preceding market session even across weekends.

Dependency bounds keep pandas below version 3 because pandas 3 changes default
datetime resolution and can make `merge_asof` reject otherwise identical daily
keys. Feature join keys are explicitly normalized to `datetime64[ns]` as a
second guard.

## OpenResearch execution

Server project `019fec29-be2c-7d20-b098-70b35fefda7d` remains blocked by HTTP 402
`billing_required` / `Out of credits`. Local dashboard project
`528668ac-3d0c-414f-ab6f-5316030509e9` was created through `orx up` and bound to
an ext4 clone at `/home/user/.cache/openresearch/repos/joe227-creator/crypto-local`.
The fixed command stayed `bash measure.sh`. A Windows-mounted source first
failed with `Invalid cross-device link`; moving source clone to ext4 repaired
runner setup without changing code or protocol.

Historical local project root answered successfully, so its branch is frozen.
Four historical direct children were run sequentially in two literature-backed
bushes. A balanced-contract anchor and ten fair siblings were then answered
under one pinned snapshot. All produced valid EVAL output; ETS narrowly beat
fair control numerically, but failed robust promotion criteria at that stage.
Cost-aware threshold Ridge was promoted later as research parent.

## First literature bush

Before Stage 1 branches, `orx lit` and arXiv evidence were reviewed. Bysik and
Slepaczuk, arXiv:2606.00060, report that naive sign trading collapses after
10-bps costs and that walk-forward, cost-aware execution dominates architectural
claims. Fang and Slepaczuk, arXiv:2606.09478, report weak state-dependent return
predictability and value in volatility/regime variables, with implementation
controls required. The first bush therefore varies only feature information:
technical Ridge versus technical-plus-lagged-on-chain Ridge. No branch changes
labels, costs, thresholds in the fixed score, split dates, or run command.

## Stage 2 Screening And Tree Rule

arXiv:2602.10785 supports double out-of-sample walk-forward evaluation with
conservative fees. arXiv:2606.27100 cautions that TSFM forecast rankings do not
automatically imply trading alpha. Existing recursive AR horizon screening was
performed locally only; all horizons lost to frozen baseline. After local root
answered, ARIMA and ETS siblings were created under confirmed root and evaluated
sequentially. Both lost to root, so stacked-bush depth stops at root; no child
is promoted.

## Local run conclusions

Root `3b0fb31c-6fed-4c8d-9b26-0160fa56e4d3` / run
`35c0d312-e63a-44c6-a672-a9f21268b29e` scored `0.1113547041`.

Technical Ridge `d717b334-1577-46d1-ae40-acd4d65456da` / run
`2cd860f7-e2b6-4b8d-82e8-872aa47a7fc7` scored `-1.1859574994`; lagged on-chain
Ridge `b0b6a0e1-f859-4ce9-a0cb-5a05ad492956` / run
`bede066d-683c-4f9b-a3a4-40d0966a4675` scored `-1.0763065591`.

ARIMA `4f43dbdd-79ef-4c74-9a46-7bae616b0c7a` / run
`e3af2a2a-f8dc-4ebf-a461-84eb85c64c53` scored `-0.8241698015`; ETS
`dd1d9581-2f47-4339-9feb-a446907499e0` / run
`6bacf5cd-ec09-42d9-8398-285001038901` scored `-1.1998268261`. ETS validation
score `0.1605075262` reversed in full score because turnover reached
`16.0372545`. Keep root as confirmed control; do not deploy model children.

## Balanced Temporal Contract

Earlier local answers used shorter validation and mutable live-data comparisons.
Those answers remain frozen historical evidence and are not used for fair
promotion. Selected contract uses chronological approximately 60/20/20
availability: training `2018-01-01..2022-12-31`, validation
`2023-01-01..2024-12-31`, frozen test `2025-01-01..2026-08-10`. Fair
contract/configured Yahoo request end is `2026-08-11`; cached Yahoo metadata
records wider exclusive request end `2026-08-12` from snapshot construction,
with no returned market bar after `2026-08-10`.

Validation is selection-only. Test is primary `Research_score` and stays
untouched until final scoring. Every fair run uses pinned processed snapshot,
126-session windows, one-session embargo, five-session labels, fixed costs, zero
threshold, seeds `11|22|33|44|55`, eight sequential Optuna trials, and
`bash measure.sh`.

Pinned snapshot hashes: market
`e9e18955f476e9b13019900ac377aade7e7f1f6cb3dfda3fa797182e84d45133`, on-chain
`e53ff12518c3bbf080dafd4f9b8c4d6daf48fc48c759c42f42e62a36baf42cfe`, manifest
`c26d693a894be8adc866767e3db7d81e617f94833cc520c022fa05116d465796`.

The filesystem ADR for this decision is
`research/ADR-001-balanced-temporal-contract.md`.

## Fair Bush Result

Balanced control anchor `b405429f-01aa-4f18-9492-005bb3ab17b4` answered with
frozen-test score `-0.2189362363`, five validation windows, and four test
windows. Ten fair siblings answered. ETS was closest at `-0.1819630`, narrowly
above control, but had negative test rolling return, lost to ETH buy-and-hold
test score `-0.0746825`, and was not robust enough to promote. All other fair
directions scored below control.

Cost-aware threshold Ridge was then tested from the control parent using the
transaction-cost filter described by arXiv:2606.00060. Optuna searched alpha
and threshold on validation only. Local result: validation `0.1135487`, frozen
test `0.2132863`, mean test rolling return `0.0375513`, maximum drawdown
`-0.0364`, Sharpe `0.7491`, turnover `1.4794`, and turnover penalty `0.0480`.
This direction is promoted as research parent, not deployment model.

Follow-up local ablations did not improve it: clipping `0.1946880`, partial
adjustment `0.1323719`, pooled Ridge `0.1021681`, cash overlay `0.1197785`,
hysteresis `0.1946880`, volatility gates `-0.4211` or lower, and AR threshold
`-0.8077` on frozen test. These remain rejected siblings.

Stage 3 foundation siblings then ran from promoted threshold parent using
repository-local model files. TimesFM 2.5 zero-shot scored validation `0.1488174`
and frozen test `-0.4697066`; Kronos base zero-shot scored validation `-0.7314223`
and frozen test `-0.8563088`. Both lost after costs and were rejected. Model
files are isolated under `models/`; adapters do not use shared cache paths.

Stage 4 TimesFM composition then tested one causal forecast feature in Ridge:
validation `0.0335998`, frozen test `-0.0728067`, mean test rolling return
`0.0201`, maximum drawdown `-0.0756`, Sharpe `0.4444`, and turnover `2.1762`.
An uncertainty abstention gate scored validation `0.2775677` and frozen test
`-0.5483675`, with turnover `2.5748`, maximum drawdown `-0.4704`, and Sharpe
`-0.8996`. Both were rejected against promoted
threshold Ridge. First Stage 4 bush is closed; deeper architectures remain
unstarted.

Promoted threshold result has one positive and three zero-return test windows;
seed score spread is zero because implementation is deterministic. Repeated seed
rows are not independent stochastic evidence. Regime slices, threshold
sensitivity, and formal Deflated Sharpe or equivalent multiple-testing adjustment
remain pending. No paper trading or production deployment.

Fair-run artifacts are preserved outside the checkout at
`/home/user/.local/share/openresearch/local-runs/<run-id>/repo/.openresearch/artifacts`.
The ignored `.openresearch/artifacts/EVAL.md` and `metrics.json` in this checkout
are local smoke output. Their commit field identifies HEAD at generation time.
Deterministic control metrics may match the fair control, but these files are
not archived fair-run anchor artifacts.

TimesFM and Kronos foundation bush is answered and rejected. Deeper architectures
remain unstarted. Any new bush requires fresh literature evidence, independent
stochastic evaluation, and completion of promoted residual-gate Ridge robustness
checks.
Server compute remains credit-blocked.

XGBoost and LightGBM were then tested as cost-aware tree-model children under
promoted threshold Ridge. XGBoost scored frozen test `-0.4176403`; LightGBM
scored `-0.4602400`. Both were rejected. Promoted threshold Ridge remains
historical parent; residual-gate Ridge is current research parent.

## Optuna Audit And Robustness Bush

Ledger now contains 32 experiment rows, 28 rows with declared Optuna studies,
and every declared study records eight trials. Current and new local SQLite
artifacts independently contain eight completed trials; historical branch
databases were overwritten or not retained, so ledger rows remain the only
record for those past trial executions. `_run_optuna` now caps resumed studies
at configured `n_trials` instead of appending another eight trials on rerun.
The Stage 4 confidence route was also restored after a later tree-model commit
made its branch unreachable; its original answered row came from the earlier
correct commit.

New literature round: arXiv:2501.12841 motivates time-varying moments and
turnover-penalized portfolio construction. arXiv:2209.12383 warns that
transaction costs can erase apparent robust gains. Four new bushes and one
append were tested with validation-only Optuna and frozen-test scoring:

- Covariance-aware Ridge: validation `0.0745508`, test `-0.2496991`; reject.
- Ridge/AR blend: validation `0.1135487`, test `0.1946880`; reject below parent.
- Training-label clipping: validation `-0.0938810`, test `-0.2307729`; reject.
- Adaptive-refit Ridge: validation `0.0761147`, test `0.2751355`; promote as
  research parent only. It has lower drawdown and baseline turnover than prior
  parent, but lower validation score, slightly lower Sharpe, and one positive
  frozen-test window out of four.
- Adaptive-refit plus covariance: validation `0.1313989`, test `-0.0569918`;
  reject.

Adaptive-refit parameters at refit sessions 40 and 46 tied on validation and
produced identical frozen-test metrics. No deployment or paper trading.

Residual uncertainty round used arXiv:2508.15922 and arXiv:2601.10591 for
probabilistic residual and uncertainty controls. Residual-scale gating scored
validation `0.0761147` and frozen test `0.3901627`, with maximum drawdown
`-0.0294`, Sharpe `0.9421`, and turnover `1.0662`. It is promoted as current
research parent only. Seven of eight validation trials scored `-0.1200`; one
trial scored `0.0761`, and one of four test windows was positive. Removing
uncertainty penalty at identical model settings scored `-0.4634` on frozen test.
Residual signal-to-noise sizing produced identical targets. An 80/20 conformal
calibration gate scored frozen test `-0.2809`; both children were rejected.
