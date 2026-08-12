# Progress Notes

## Historical / Pre-Balanced Stage 0

Local fixed-contract reproduction completed on 2026-08-10. 6 initial tests,
then 7 after cash-sleeve regression coverage. 2018-01-01 through 2026-08-10
Yahoo daily OHLCV plus Coin Metrics daily aggregates loaded. 50/50 BTC/ETH
buy-and-hold defines baseline turnover `0.9990010` and local Research_score
`0.1115835`. BTC buy-and-hold scores `0.1275554`; ETH scores `0.1196255`; cash
scores `-0.1200000`. No local rule beats BTC buy-and-hold after fixed costs.

Initial server root launch attempted through managed CPU backend and returned
HTTP 402 `billing_required` before provisioning. That server project remains
unanswered. Later local OpenResearch execution answered the historical root;
those results are documented below and remain separate from the fair contract.

## Historical / Pre-Balanced Stage 1 Bush 1

Literature evidence: arXiv:2606.00060 emphasizes transaction-cost-aware
walk-forward execution; arXiv:2606.09478 finds weak state-dependent return
predictability and value in regime/volatility variables. Bush varies one decision:
feature information, not model family, score, labels, or costs.

Technical-only Ridge fixed-protocol local smoke: score `-1.1859575`, validation
`-0.6547601`, test `-1.0324355`, turnover `12.4341234`, alpha `89.1665528`.

Technical plus lagged on-chain Ridge fixed-protocol local smoke: score
`-1.0763066`, validation `-0.6487563`, test `-0.5311196`, turnover `11.4842383`,
alpha `89.1665528`.

On-chain child was locally less bad but not promoted. These original server
planning nodes have no orx run IDs because the same billing blocker prevented
execution. Local smoke did not substitute for sequential orx runs, robustness,
regime slices, or promotion evidence; fair-contract reruns are documented below.

Session-based embargo regression passed. Fixed smoke values remained unchanged.

## Historical / Pre-Balanced Stage 2 Screening

Fresh literature review covered arXiv:2602.10785 and arXiv:2606.27100. The first
supports strict double out-of-sample walk-forward testing with conservative
costs; the second cautions that daily financial return predictability is weak
and TSFM forecast gains do not imply trading alpha.

Existing causal recursive AR implementation was screened locally at horizons 1,
5, and 21. Scores were `-2.6286336`, `-1.7015701`, and `-0.7378930`; horizon 21
was least bad but remained below frozen baseline `0.1115835`. Screening is not an
orx answer, so no node was created, no direction was promoted, and no parent was
chosen for a new stacked bush.

## Superseded Plan

Earlier credit-recovery and pre-balanced launch plans are superseded by the
answered local runs and balanced fair contract below. They are not active
commands. Server compute remains blocked; later local robustness work promoted
adaptive-refit Ridge as research parent.

## Local OpenResearch Execution

Local dashboard project `528668ac-3d0c-414f-ab6f-5316030509e9` was created on
WSL ext4 after the Windows-mounted checkout triggered an `Invalid cross-device
link` clone failure. Fixed command remained `bash measure.sh`; all runs below
completed sequentially and emitted `EVAL.md` plus required text artifacts.

Root `3b0fb31c-6fed-4c8d-9b26-0160fa56e4d3`, run
`35c0d312-e63a-44c6-a672-a9f21268b29e`, commit `aa85267`, answered with score
`0.1113547041`, validation `-0.3271167699`, test `0.0573868441`, and turnover
`0.9990010`. Root is frozen control.

Feature-information bush under root completed without promotion:

- Technical Ridge `d717b334-1577-46d1-ae40-acd4d65456da`, run
  `2cd860f7-e2b6-4b8d-82e8-872aa47a7fc7`: score `-1.1859574994`, validation
  `-0.6547601164`, test `-1.0324354803`, turnover `12.4341234`.
- Ridge plus lagged on-chain `b0b6a0e1-f859-4ce9-a0cb-5a05ad492956`, run
  `bede066d-683c-4f9b-a3a4-40d0966a4675`: score `-1.0763065591`, validation
  `-0.6487562985`, test `-0.5311196384`, turnover `11.4842383`.

Classical model-family bush under root also completed without promotion:

- ARIMA(1,0,0) `4f43dbdd-79ef-4c74-9a46-7bae616b0c7a`, run
  `e3af2a2a-f8dc-4ebf-a461-84eb85c64c53`: validation selected order 1; score
  `-0.8241698015`, validation `-0.6370704203`, test `-0.6293706158`, turnover
  `8.2565330`.
- ETS level smoothing `dd1d9581-2f47-4339-9feb-a446907499e0`, run
  `6bacf5cd-ec09-42d9-8398-285001038901`: validation selected alpha
  `0.2583784`; score `-1.1998268261`, validation `0.1605075262`, test
  `-0.0888851444`, turnover `16.0372545`. Validation gain did not survive full
  evaluation and turnover penalty.

All five local runs had zero seed-score spread because current forecasters are
deterministic. Four model siblings regressed versus confirmed root; no model is
promoted or suitable for deployment. Stop at this checkpoint pending foundation
model dependencies or a separately justified research direction.

## Balanced Contract And Fair Bush

Historical local answers above are not directly comparable: they used an older
split and/or mutable data. New child contract was selected and frozen: training
`2018-01-01..2022-12-31`, validation `2023-01-01..2024-12-31`, frozen test
`2025-01-01..2026-08-10`, with fair contract/configured request end
`2026-08-11`. Cached Yahoo metadata records wider exclusive request end
`2026-08-12` from snapshot construction. Pinned snapshot hashes are recorded
in `DECISIONS.md`.

Balanced control anchor `b405429f-01aa-4f18-9492-005bb3ab17b4`, run
`9ec6329e-3c8b-409f-95f1-f0ebe9794e16`: validation `0.5314679`, frozen test
`-0.2189362`, full `0.1115835`, five validation windows, four test windows.

Fair siblings, all under same snapshot and protocol:

- Technical Ridge `a903c343-b32b-47ff-a19e-1b4050d7f793`, run
  `5b151771-3dae-4439-9dac-428a44ede82f`: validation `-0.7239327`, test
  `-0.9475211`, turnover `5.3080`.
- Ridge plus on-chain `c2c974b4-abe1-4953-8efb-feaa477cdb18`, run
  `8cb61076-5cac-4538-a91e-cc4a1f5e880f`: validation `-0.1714904`, test
  `-0.6419978`, turnover `3.9480`.
- ARIMA `a8eee72f-942d-4ce9-8461-443638d01197`, run
  `9eb8a8de-5258-4b08-a082-2a890961f6f9`: validation `0.5506139`, test
  `-0.7781348`, turnover `4.7139`.
- ETS `61b47e35-bc71-4d8c-b793-38bbfd654387`, run
  `405a267c-7509-452e-9bdb-22b18a180bab`: validation `1.0205072`, test
  `-0.1819630`, turnover `0.8950`; validation overfit did not transfer.
- Cross-asset Ridge `ac10fa19-d0e3-4376-bd36-97431b5368f6`, run
  `c27c2db6-c8bb-43b8-bd80-7361f43ef68f`: validation `-0.9148506`, test
  `-1.2068932`, turnover `7.4232`.
- Pooled Ridge `4a0eb7f8-bb3c-4542-ad31-71a9cbd9e3af`, run
  `bf7df15e-9f45-431f-a6a7-2029f20bccf4`: validation `-0.6372513`, test
  `-0.9904385`, turnover `5.1874`.
- EMA Ridge `f92e75e0-a0f8-483c-9981-f7526afc7ac6`, run
  `f79e767b-13c8-40b3-9eb8-943aac42e3fc`: validation `0.1572936`, test
  `-0.4812781`, turnover `1.9593`.
- Top-1 Ridge `335af757-cb6d-429a-a0d5-be9df4b2ee40`, run
  `505056ac-5b31-4ed2-8c38-29add2e4cf99`: validation `-0.4045567`, test
  `-0.7593821`, turnover `4.3029`.
- Hysteresis Ridge `8d6e2d4d-719a-4ddb-87fb-9f27f2d36316`, run
  `f14e7395-4126-4a0f-9ecb-35f87ff101a4`: validation `0.0617274`, test
  `-0.3176550`, turnover `2.1506`.
- Volatility-gated Ridge `d2669976-1668-429c-acf4-a6b16f6d188a`, run
  `140323a1-1d02-4081-9bf9-00a3537fb07b`: validation `-0.6569587`, test
  `-0.7909709`, turnover `4.7239`.

Fair bush stopped. No child beats fixed control robustly, no parent is promoted,
and no deployment recommendation changes: retain simple controls and paper-trade
nothing. Continue only with new evidence-backed hypotheses or unblocked
foundation-model dependencies.

## Stage 3 Foundation Bush

TimesFM 2.5 and Kronos source repositories plus local checkpoints were copied
under ignored `external/` and `models/` paths. Adapters load repository-local
files only; shared Hugging Face cache is not used.

- TimesFM 2.5 zero-shot, local commit `84484e5`: validation `0.1488174`, frozen
  test `-0.4697066`, turnover `3.3396`, Sharpe `-0.5150`; rejected.
- Kronos base zero-shot, local commit `84484e5`: validation `-0.7314223`, frozen
  test `-0.8563088`, turnover `5.1900`, Sharpe `-0.7259`; rejected.

Stage 3 foundation bush stopped. Threshold Ridge remains promoted research
parent; deeper architectures remain unstarted. Paper-trade nothing.

## Stage 4 Hybrid Bush

Literature round included arXiv:2607.20002 on TSFM post-training composition,
uncertainty control, and output processing. TimesFM forecast was appended as one
causal Ridge feature, then an uncertainty abstention gate was tested.

- TimesFM-to-Ridge hybrid: validation `0.0335998`, frozen test `-0.0728067`,
  turnover `2.1762`, Sharpe `0.4444`; rejected.
- TimesFM uncertainty gate: validation `0.2775677`, frozen test `-0.5483675`,
  turnover `2.5748`, Sharpe `-0.8996`; rejected.

Stage 4 first bush stopped. Promoted threshold Ridge remains research parent;
deeper architectures remain unstarted. Paper-trade nothing.

## Stage 1 Tree Model Bush

Literature arXiv:2606.00060 reports XGBoost as strongest configuration on
hourly BTC walk-forward. XGBoost and LightGBM were tested as cost-aware
siblings under promoted threshold Ridge using identical features, labels,
splits, costs, and causal walk-forward contract. Each ran eight Optuna
trials on validation only.

- XGBoost: validation `-0.3681458`, frozen test `-0.4176403`, turnover
  `3.0949`, Sharpe `-0.3669`; rejected.
- LightGBM: validation `-0.4197973`, frozen test `-0.4602400`, turnover
  `2.8373`, Sharpe `-0.5272`; rejected.

Both were comprehensively below promoted threshold Ridge `0.2132863`.
Tree models cannot extract a tradable signal from these features under
these costs. Linear Ridge remains the single research parent.
Paper-trade nothing.

## Robustness Bush

New literature round used arXiv:2501.12841 for time-varying moments and
turnover-penalized allocation, plus arXiv:2209.12383 for transaction-cost
robustness cautions. Every branch below used eight validation-only Optuna
trials and the unchanged frozen-test contract.

- Covariance-aware Ridge allocator: validation `0.0745508`, test `-0.2496991`;
  rejected.
- Ridge/AR forecast blend: validation `0.1135487`, test `0.1946880`; rejected
  below prior parent with higher turnover.
- Training-label clipping: validation `-0.0938810`, test `-0.2307729`;
  rejected.
- Adaptive-refit Ridge: validation `0.0761147`, frozen test `0.2751355`,
  maximum drawdown `-0.0294`, Sharpe `0.7184`, turnover `0.9990`; promoted as
  new research parent only. Refit sessions `40` and `46` tied on validation
  and reproduced identical frozen metrics.
- Adaptive-refit covariance append: validation `0.1313989`, test `-0.0569918`;
  rejected despite lower turnover.

Adaptive-refit Ridge beats prior threshold Ridge score `0.2132863` and lowers
drawdown/turnover, but validation score is lower and frozen test still has one
positive window out of four. No deployment or paper trading.

## Residual Uncertainty Bush

Fresh literature included arXiv:2508.15922 on residual-based probabilistic
volatility forecasts and arXiv:2601.10591 on uncertainty decomposition. New
branches used eight validation-only Optuna trials under unchanged contract.

- Residual uncertainty Ridge gate: validation `0.0761147`, frozen test
  `0.3901627`, maximum drawdown `-0.0294`, Sharpe `0.9421`, turnover `1.0662`;
  promoted as current research parent only. Parameters: alpha `0.1527`, refit
  `81`, residual multiplier `0.1445`, threshold `0.01335`.
- Residual signal-to-noise sizing: identical validation and frozen-test metrics
  and identical targets; rejected as no incremental direction.
- Conformal residual calibration gate: validation `-0.1200`, frozen test
  `-0.2809`; rejected.

Residual gate validation search was fragile: seven trials scored `-0.1200`, one
scored `0.0761`. Fixed no-uncertainty ablation collapsed to frozen test
`-0.4634`. One of four test windows was positive. No deployment or paper trading.
