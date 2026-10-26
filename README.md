# Limit Order Book Alpha Research & Market Microstructure Engine


Research question: *Do causal order-book imbalance, order flow and liquidity features forecast short-horizon midprice moves, and can we monetize predictions after crossing the spread and paying fees?*

> **Status:** A runnable, tested research **starter**, not a validated profitable HFT strategy. Synthetic output is strictly for software tests. The one-day LOBSTER sample is only a demonstration, not enough for credible generalization or live execution claims.

## 1. Quick start

Python 3.10+ (Windows / macOS / Linux):

```bash
python -m venv .venv
# On Windows PowerShell: .venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate
pip install -e '.[dev,lightgbm]'
pytest -q
lob-alpha demo-data --out data --events 6000
lob-alpha verify-demo --message data/synthetic_message.csv --book data/synthetic_orderbook.csv
lob-alpha run --messages data/synthetic_message.csv --books data/synthetic_orderbook.csv --model logistic --out reports/demo
```

If PowerShell has quoting issues, use `pip install -e .` instead; LightGBM and pytest can be installed separately. The main output is `reports/demo/results.json` plus a trade log for each forecasting horizon.

### Real NASDAQ data

Download an AAPL 10-level **message** + **orderbook** pair from [LOBSTER's official sample page](https://php.lobsterdata.com/info/DataSamples.php). Unzip the files into `data/`. Locate the pair (typically filenames containing `message` and `orderbook`). Then run:

```bash
lob-alpha run --messages data/YOUR_AAPL_message.csv --books data/YOUR_AAPL_orderbook.csv --model logistic --out reports/aapl
lob-alpha run --messages data/YOUR_AAPL_message.csv --books data/YOUR_AAPL_orderbook.csv --model lightgbm --latency 5 --out reports/aapl_lgbm_latency5
```

For meaningful out-of-sample tests, acquire *several distinct full trading days* and provide both lists in chronological order (one pair per session):

```bash
lob-alpha run --messages data/day01_message.csv data/day02_message.csv data/day03_message.csv --books data/day01_orderbook.csv data/day02_orderbook.csv data/day03_orderbook.csv --out reports/multiday
```

For >=3 sessions, model training/validation/test is split chronologically by **whole days**; otherwise it splits chronologically within the sample and emits a warning. Never interpret one-day test results as cross-day out-of-sample evidence.

## 2. What's included

| File | Purpose |
|---|---|
| `data.py` | Official LOBSTER six-column messages and 4xN orderbook row loader; fixed-point prices → USD; basic checks |
| `book.py` | Exact ID-based limit-order reconstruction from **complete** message streams; audit synthetic snapshots |
| `synthetic.py` | Coherent reproducible L3 event and snapshot generator for smoke tests, never research evidence |
| `features.py` | Causal OBI-1, OBI-5, OFI, microprice, pressure, depth, spread, rolling trade/cancel intensity |
| `labels.py` | Three-way price movement labels at 10/50/100 event horizons; day-safe purged split |
| `models.py` | Logistic and optional LightGBM; train-only preprocessing; balanced accuracy, F1, log loss, Brier, ECE |
| `execution.py` | Nonoverlapping long/short marketable order proxy using actual bid/ask, fees, top-level depth, latency |
| `pipeline.py` | End-to-end CLI orchestrator, experiment JSON and trade CSVs |
| `tests/` | Deterministic tests for book replay, temporal causality, label purging, execution and integration |

## 3. The signal research

At event t, bid/ask prices are B_t and A_t, bid/ask available sizes are q^B_t and q^A_t.

- **Mid:** (A_t+B_t)/2; **spread:** A_t - B_t.
- **Queue imbalance:** OBI = (q^B - q^A) / (q^B + q^A).
- **Microprice:** (A*q^B+B*q^A)/(q^B+q^A).
- **OFI:** [1(B_t>=B_{t-1})q^B_t - 1(B_t<=B_{t-1})q^B_{t-1}] - [1(A_t<=A_{t-1})q^A_t - 1(A_t>=A_{t-1})q^A_{t-1}].
- **Trade pressure:** sign of aggressor * executed quantity; a trade against a displayed sell order has positive buy-aggressor direction. 10/50 event rolling window.
- **Cancel pressure:** cancellation at bid removes bid depth (negative); cancellation at ask removes ask depth (positive). 50-event rolling window.
- **Targets:** sign(mid[t+h] - mid[t]) with neutral band of ±0.5 tick; horizons 10/50/100 events.

All features are computed at and before t. All target horizons are future and purged at split/day boundaries. `lightgbm` is an optional extra, logistic requires only base dependencies.

## 4. Execution interpretation

- A feature is observed after event **t**. An order may execute at **t+1+latency_events** (never at event t's already-observed price). It exits after an additional h events.
- Long opens at the *ask* and closes at the *bid*; short opens at the *bid* and covers at the *ask*.
- Orders are rejected when top-of-book available size is below order size; explicit per-side share fees are included; trading never crosses the session boundary.
- The strategy trades only when up or down predicted probability exceeds 0.40 and differs from the opposite direction by >=0.07. Thresholds are illustrative and **not** optimized on the test set.
- One position at a time avoids overlapping trades. **Do not add overlapping per-event P&L to make a backtest look profitable.**
- The backtest makes an *optimistic* execution assumption: sufficient displayed top-of-book shares can be fully traded at the observed quote. It excludes market impact, routing, adverse fill conditions, borrow costs, maker/taker differentiation, queue position, and real wall-clock latency. Hence its profitability is **not proof of a live-executable alpha**.

## 5. Order-by-order reconstruction caveat

LOBSTER samples are *level-truncated*: the accompanying message file may not contain all older resting orders, events outside the requested depth, or the complete history needed to recover the full order-id queue from empty state. Therefore:

- **Use provided LOBSTER snapshot files as the ground truth** for features and forecasting.
- **Do not** run `verify-demo` on a partial official sample and conclude the order-level engine is incorrect merely because unseen orders are missing.
- `book.py` can exactly reconstruct an **entire ordered stream starting from empty**, demonstrated by `demo-data` and `verify-demo`.
- A full order-level historical engine on NASDAQ ITCH should ingest the *entire feed from session initialization*, handle exchange-specific amendments/trade corrections, and verify exchange-consistent depths, queue ordering and time.

Official description: [LOBSTER output schema](https://php.lobsterdata.com/info/DataStructure.php) and [reconstruction notes](https://php.lobsterdata.com/info/HowDoesItWork.php).

## 6. Next research milestones (not implemented yet)

1. **Microstructure evidence:** proper multi-day data; plots by time-of-day, imbalance deciles, spread regimes; event-time and clock-time horizons; economic statistical significance with dependent data.
2. **Hyperparameter tuning:** train/validation-only model and threshold tuning, walk-forward across days and multiple tickers, embargo for correlated labels, calibration, bootstrapped uncertainty.
3. **Passive fill engine:** price/time priority, queue-ahead volume, real order-ID order matching, cancellations ahead versus behind, adverse selection and maker/taker fees. Cannot accurately infer individual queue positions from aggregated top-10 snapshots.
4. **DeepLOB:** CNN-LSTM or alternative sequence model with train-only normalized L2 order-book tensors; compare speed vs predictive benefit.
5. **More realistic execution:** actual timestamp latency, impact, partial fills, quote lifetime, venue/multi-market fragmentation, short stock borrow, position/inventory constraints.
6. **Research paper / dashboard:** ablations (OBI vs OFI, logistic vs LightGBM, event horizon, latency), calibration plots, performance and drawdown report.

## 7. Recommended experiment matrix

| Comparison | Hypothesis |
|---|---|
| OBI vs OFI | Event-delta flow could contain more information than static queue snapshot |
| Logistic vs LightGBM | Nonlinear interactions help particularly when spread/depth regimes change |
| 10 vs 50 vs 100 events | Signal information decays with horizon |
| 0 vs 5 vs 20 extra events | Fill delay erodes short-lived predictive value |
| Aggressive vs passive (future) | Price improvement competes with fill probability and adverse selection |

**Portfolio claim only after collecting genuine cross-day test results.** Until then, say you *built a reproducible LOB alpha research and execution simulation framework*, not that you found profitable HFT alpha.

### Additional comparison commands

```bash
# OBI vs OFI using the same model and the same data horizon:
lob-alpha run --messages data/YOUR_AAPL_message.csv --books data/YOUR_AAPL_orderbook.csv --features obi --horizons 50 --out reports/obi
lob-alpha run --messages data/YOUR_AAPL_message.csv --books data/YOUR_AAPL_orderbook.csv --features ofi --horizons 50 --out reports/ofi
# Sensitivity to extra event latency:
lob-alpha run --messages data/YOUR_AAPL_message.csv --books data/YOUR_AAPL_orderbook.csv --latency 20 --out reports/latency20
```

Each experiment also saves `summary.md`, held-out confusion-matrix PNGs, and cumulative net P&L PNGs (when trades occur). `snapshot_checks.visible_event_delta_check` compares same-price liquidity changes across adjacent official snapshots when the order price remains visible in both.
