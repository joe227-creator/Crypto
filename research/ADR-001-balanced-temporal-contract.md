# ADR-001: Balanced Temporal Contract

- Status: accepted
- Date: 2026-08-10
- Scope: BTC/ETH long-only research and all fair comparison siblings

## Context

Earlier local runs used shorter validation periods and mutable live-data
comparisons. Their scores remain useful as historical evidence, but they cannot
support fair promotion against runs using a different temporal boundary or data
snapshot. A reproducible comparison needs one immutable market/on-chain snapshot,
explicit chronological partitions, and a test period untouched by selection.

## Decision

Use one balanced chronological contract for the fair bush:

- Universe: BTC and ETH only; start in cash; non-negative BTC, ETH, and cash
  weights summing to one.
- Training/in-sample: `2018-01-01..2022-12-31`.
- Validation: `2023-01-01..2024-12-31`, selection-only.
- Frozen primary test: `2025-01-01..2026-08-10`.
- Fair contract/configured Yahoo request end: `2026-08-11`, since its end bound
  is exclusive.
- Cached raw metadata records wider request end `2026-08-12` from snapshot
  construction; returned market data ends at `2026-08-10`.
- Signal at daily close; target executes at next daily open.
- Costs: 10 bps fee plus 5 bps slippage per side.
- Labels: five-session forward simple return less fixed round-trip cost.
- Embargo: one market session between label end and signal date.
- Score: fixed `Research_score`; 126-session complete non-overlapping windows;
  252-session annualized Sharpe; base strategies use zero signal threshold,
  while explicit cost-aware filter branches may declare threshold tuning.
- Seeds: `11|22|33|44|55`; eight sequential validation-only Optuna trials where
  a branch declares tunable parameters; command `bash measure.sh`.

Pinned snapshot digests:

- Market: `e9e18955f476e9b13019900ac377aade7e7f1f6cb3dfda3fa797182e84d45133`.
- On-chain: `e53ff12518c3bbf080dafd4f9b8c4d6daf48fc48c759c42f42e62a36baf42cfe`.
- Manifest: `c26d693a894be8adc866767e3db7d81e617f94833cc520c022fa05116d465796`.

## Consequences

Fair siblings can be compared directly without changing costs, labels, score,
execution, split boundaries, or snapshot. Validation may select parameters, but
the frozen test cannot influence tuning or promotion. Historical pre-balanced
answers remain explicitly non-comparable evidence.

The balanced control and ten fair siblings were answered locally. A later
cost-aware threshold Ridge child, motivated by arXiv:2606.00060, scored
`0.2132863` on frozen test versus control `-0.2189362`, with mean test rolling
return `0.0375513`, maximum drawdown `-0.0364`, Sharpe `0.7491`, and turnover
`1.4794`. It is promoted as research parent, not deployment model.

Clipping, partial adjustment, pooled Ridge, cash overlay, hysteresis,
volatility-gate, and AR-threshold follow-ups did not improve promoted candidate.

Stage 3 foundation siblings used repository-local TimesFM 2.5 and Kronos base
weights. TimesFM scored frozen test `-0.4697066`; Kronos scored `-0.8563088`.
Both were rejected after costs. Deeper architectures remain unstarted.

Stage 4 TimesFM composition used one causal forecast feature in Ridge and scored
frozen test `-0.0728067`. An uncertainty abstention gate scored `-0.5483675`.
Both were rejected against promoted threshold Ridge. Deeper architectures remain
unstarted.

The next robustness bush tested covariance-aware allocation, Ridge/AR blending,
training-label clipping, and adaptive Ridge refit cadence. All branches used
validation-only Optuna with eight trials and preserved the accepted contract.
The adaptive-refit Ridge child scored frozen test `0.2751355`, versus prior
threshold Ridge `0.2132863`, with maximum drawdown `-0.0294`, Sharpe `0.7184`,
and turnover `0.9990`. It is promoted as a research parent only. Its validation
score was lower (`0.0761147` versus `0.1135487`), one of four frozen windows was
positive, and the adaptive covariance append failed at `-0.0569918`; these gaps
block deployment.

A residual-uncertainty gate was then added to adaptive Ridge. It uses only
expanding-fit residual scale available before each signal date. It scored frozen
test `0.3901627`, maximum drawdown `-0.0294`, Sharpe `0.9421`, and turnover
`1.0662`; it is the current research parent only. Seven of eight validation
trials scored `-0.1200`, one scored `0.0761`, and one of four test windows was
positive. Removing uncertainty penalty scored `-0.4634`. Signal-to-noise sizing
was identical; an 80/20 conformal calibration gate scored `-0.2809`. These
results preserve contract but leave strong selection and regime-concentration
gaps. No deployment.

Expanded neural and foundation tests preserved contract. Causal LSTM scored
frozen test `-0.1157`, TCN `-0.7860`, Transformer `-0.4588`, and static TimesFM
point-head fine-tuning `-0.5189`; none beat residual-gate Ridge on robustness.
Dynamic TimesFM fine-tuning exceeded runtime budget after partial Optuna
execution and has no scientific result. Residual-gate multiplier sensitivity
was severe, but mid-range `0.15` and `0.20` settings scored `0.3202` and
`0.2751`; doubled fee/slippage at `0.15` scored `0.2561`. These results support
continued research only, not deployment.

Static Kronos dual-head fine-tuning used teacher-forced token loss on
pre-validation OHLCV paths. Six of eight Optuna trials completed before two
timeouts. Selected rerun scored frozen test `-0.7654`, maximum drawdown
`-0.6107`, Sharpe `-0.4848`, and turnover `5.4815`; fine-tuning did not rescue
zero-shot Kronos. Reject without changing accepted contract.

A final gate append-and-ablation round descended on residual-gate Ridge. Raw and
z-scored 12-1 momentum, trailing-252d feature standardization, MAD gate scale,
and rolling-window Ridge all lost to the parent. Ridge+LSTM blend and
LSTM-on-chain overfit validation then reversed on the frozen test. LSTM residual
gates (raw and normalized) suppressed all signal. XGBoost residual gate and
dynamic TimesFM fine-tune both failed the frozen test; dynamic Kronos fine-tune
was aborted before an answer. Optuna wiring verified clean for every branch. No
child beat the parent, confirming the accepted contract while the tree plateaus.

## Evidence Gaps

Promoted candidate has one positive and three zero-return frozen-test windows.
Repeated seed rows have zero spread because current implementations are
deterministic; they are not independent stochastic evidence. Regime slices,
parameter sensitivity, feature stability, and formal Deflated Sharpe or
equivalent multiple-testing adjustment remain unmeasured. These gaps block
deployment and require next research round, but do not justify changing the
accepted contract retrospectively.
