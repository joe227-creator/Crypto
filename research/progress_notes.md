# Progress Notes

## Stage 0

Local fixed-contract reproduction completed on 2026-08-10. 6 initial tests,
then 7 after cash-sleeve regression coverage. 2018-01-01 through 2026-08-10
Yahoo daily OHLCV plus Coin Metrics daily aggregates loaded. 50/50 BTC/ETH
buy-and-hold defines baseline turnover `0.9990010` and local Research_score
`0.1115835`. BTC buy-and-hold scores `0.1275554`; ETH scores `0.1196255`; cash
scores `-0.1200000`. No local rule beats BTC buy-and-hold after fixed costs.

OpenResearch root launch attempted through managed CPU backend. API returned
HTTP 402 `billing_required` before provisioning. Root has no run ID and remains
provisional. Local numbers are reproducibility evidence, not orx answers.

## Stage 1 Bush 1

Literature evidence: arXiv:2606.00060 emphasizes transaction-cost-aware
walk-forward execution; arXiv:2606.09478 finds weak state-dependent return
predictability and value in regime/volatility variables. Bush varies one decision:
feature information, not model family, score, labels, or costs.

Technical-only Ridge fixed-protocol local smoke: score `-1.1859575`, validation
`-0.6547601`, test `-1.0324355`, turnover `12.4341234`, alpha `89.1665528`.

Technical plus lagged on-chain Ridge fixed-protocol local smoke: score
`-1.0763066`, validation `-0.6487563`, test `-0.5311196`, turnover `11.4842383`,
alpha `89.1665528`.

On-chain child is locally less bad but not promoted. Neither child has an orx
run ID because same billing blocker prevented execution. Local smoke does not
substitute for sequential orx runs, 5-10 seed robustness, regime slices, or
promotion evidence.

Session-based embargo regression passed. Fixed smoke values remained unchanged.

## Next Tick After Credits

1. Run root `019fec2a-3c90-74f8-a1c6-fddf122b3d4c` sequentially through orx.
2. Read root `EVAL.md`; freeze root only after successful answer.
3. Run technical child, then on-chain child, one at a time; read each artifact.
4. Promote only if full score and robustness-adjusted test evidence beats the
   frozen reference. Do not promote from local smoke values.
