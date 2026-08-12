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
  252-session annualized Sharpe; zero signal threshold.
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

## Evidence Gaps

Promoted candidate has one positive and three zero-return frozen-test windows.
Repeated seed rows have zero spread because current implementations are
deterministic; they are not independent stochastic evidence. Regime slices,
parameter sensitivity, feature stability, and formal Deflated Sharpe or
equivalent multiple-testing adjustment remain unmeasured. These gaps block
deployment and require next research round, but do not justify changing the
accepted contract retrospectively.
