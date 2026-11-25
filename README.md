# Limit Order Book Alpha Research & Market Microstructure Engine

**Real market data workflow** for quantitative trading / HFT research. Based on the user's original code, with an official LOBSTER data downloader and real-data CLI. **No synthetic results are presented as alpha evidence.**

## Completed real-market run

The package now includes the official **AAPL 10-level NASDAQ sample for 21 June 2012** in `data/raw/`, and completed results in `reports/aapl_full_session/`. All **400,391** event/snapshot pairs were used; there was no row cap and no synthetic data in this research run.

- Model: logistic regression with train-only imputation/scaling and balanced class weights.
- Split: chronological 60% / 20% / 20% within one day, with label-horizon purging and feature warmup.
- Horizons: 10, 50 and 100 events; 10 shares per simulated trade; $0.003/share/side fees; earliest entry at the next event.
- No missing top quotes, locked/crossed books, reversed timestamps or unordered depth were found. All 178,116 eligible visible-volume transitions matched.
- Simulated net P&L was **negative at every horizon**. These are diagnostic results, not a profitable-strategy claim.
- The data are historical, not a live feed; one date cannot establish cross-day generalisation.

| Horizon | Test accuracy | Majority baseline accuracy | Simulated net P&L |
|---:|---:|---:|---:|
| 10 events | 39.97% | 51.32% | $-4,449.02 |
| 50 events | 53.73% | 45.38% | $-1,214.16 |
| 100 events | 53.32% | 49.38% | $-608.50 |

The majority baseline always predicts the most frequent **training** class. The horizons are separate experiments; do not add their P&Ls together.

To reproduce the included run from the extracted project folder:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m lob_alpha run-real --data-dir data/raw --symbol AAPL --levels 10 --model logistic --out reports/aapl_full_session
```

`requirements-tested.txt` records the versions used. The reports include file fingerprints, configuration, package versions, confusion matrices, equity curves and trade logs. The download manifest records the official source URL and hashes. Synthetic generation remains available only for software tests.

## What is already implemented

- Official LOBSTER paired message + orderbook loader (NASDAQ samples, fixed-point prices to USD).
- Causal microstructure features: top-of-book and five-level imbalance, order-flow imbalance, microprice deviation, trade/cancel pressure, spread, depth.
- Three-class next-10/50/100-event midprice prediction with purged chronological splits, logistic and LightGBM models, and held-out diagnostic plots.
- An **aggressive execution proxy** that enters at post-decision bid/ask with configurable event delay, exits after the chosen horizon, checks displayed liquidity and pays share-based fees.
- Event-level smoke tests, price/depth sanity checks, and human-readable report exports.

## Quick start (real NASDAQ market data)

From the root of this project, using Python 3.10+:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -e ".[dev,lightgbm]"
python -m pytest -q
```

### Step 1: Obtain the official **real** AAPL dataset (already included)

LOBSTER publishes a public full-day 10-level historical NASDAQ AAPL sample from **2012-06-21**. The data is derived from the official NASDAQ Historical TotalView-ITCH sample; it is **not synthetically generated**.

Try the built-in helper:

```bash
lob-alpha download-sample --symbol AAPL --levels 10 --out data/raw
```

If the server blocks scripted downloads, open the official source at https://php.lobsterdata.com/info/DataSamples.php and download the **Apple / AAPL / Level 10** ZIP yourself. Extract the matching **`*_message_10.csv` and `*_orderbook_10.csv`** files into `data/raw/`. This also works if you obtained your data through a licensed LOBSTER account. Please obey the provider's usage and redistribution terms.

### Step 2: Run the research pipeline on real data

```bash
lob-alpha run-real --data-dir data/raw --symbol AAPL --levels 10 --model logistic --max-rows 100000 --out reports/aapl_first_run
```

`--max-rows 100000` is a resource-friendly diagnostic; omit it for the full session (higher RAM). If one session's early period has low class diversity, use the full sample instead.

If LightGBM is installed:

```bash
lob-alpha run-real --data-dir data/raw --symbol AAPL --model lightgbm --max-rows 100000 --latency 5 --out reports/aapl_lightgbm
```

Run OBI vs OFI ablations:

```bash
lob-alpha run-real --data-dir data/raw --symbol AAPL --features obi --horizons 50 --max-rows 100000 --out reports/obi
lob-alpha run-real --data-dir data/raw --symbol AAPL --features ofi --horizons 50 --max-rows 100000 --out reports/ofi
```

Each run writes `results.json`, `summary.md`, and test-trade CSV and plots under `--out`. Inspect the held-out test accuracy, class support, Spearman IC, trade count, net P&L, and snapshot-integrity warnings.

### Step 3: Acquire multiple independent trading sessions (required for research claims)

The freely available LOBSTER stock samples are **different tickers on one calendar date**. They are NOT independent cross-day evidence. The built-in `run-real` command intentionally selects one symbol and discovers one data pair per session date.

For a credible experiment, obtain at least **10 distinct days for the same symbol** where possible, placing each pair in `data/raw/` with LOBSTER's original names. The code sorts session dates and uses whole-day chronological train/validation/test splits when it sees >=3 days.

```bash
lob-alpha run-real --data-dir data/raw --symbol AAPL --model logistic --out reports/aapl_multiday
```

Buying additional historical days directly from LOBSTER may require an account/license. Alternatives such as FI-2010 (preprocessed five-stock/ten-day benchmark) and historical crypto L2 snapshots **are not drop-in compatible with the LOBSTER 6-column message schema**. They need independent data adapters and adjustments to execution modelling. Do not pretend they are identical.

## How to interpret the analysis

- **Prediction is not profit.** A 60% direction classifier can still lose money by crossing a wide spread.
- **No future information at entry.** The backtester observes features through event `t`, then buys/sells no earlier than event `t+1+latency`.
- **Event latency is not clock latency.** Ten events may represent dramatically different elapsed times across sessions and symbols.
- **Execution is an optimistic approximation.** It assumes all specified shares trade at the displayed best bid/ask if size is sufficient, and excludes market impact, queue dynamics, routing, delayed/cancelled fills and borrow.
- **Model selection is not a repeated test.** Keep final test days unseen while tuning features, horizons, parameters and execution thresholds on train/validation data. Run repeated rolling-origin experiments only within allowed train/validation history.
- **Historical samples are old.** Results from June 2012 cannot be represented as current NASDAQ performance.

## Structure

```text
src/lob_alpha/
  real_data.py   # NEW: official sample downloader, manifests and deterministic session discovery
  cli.py         # MODIFIED: download-sample and run-real commands
  data.py        # LOBSTER data parsing and integrity checks
  features.py    # causal microstructure features
  labels.py      # three-class labels, purged chronological splitting
  models.py      # logistic / LightGBM and metrics
  execution.py   # aggressive bid/ask backtest with explicit latency and fees
  pipeline.py    # end-to-end run, diagnostics and experiment results
  book.py        # L3 order-ID engine for complete streams only
  synthetic.py   # synthetic generator for software unit tests ONLY
  visuals.py     # visual reports
tests/            # unit/integration tests
data/raw/         # official public sample for local use; observe provider terms
reports/          # generated outputs
```

## Research next steps

1. Run real full-day AAPL and inspect snapshot quality, base rates and midprice tick behaviour.
2. Acquire multiple historical sessions for robust out-of-sample testing.
3. Add session-by-session performance and block-bootstrap confidence intervals for IC and net profit.
4. Compare OBI, OFI and trade-flow features under varying event horizons and execution latency.
5. Replace optimistic aggressive fills with more conservative sensitivity assumptions (spread shocks / delayed fills) before making economic claims.
6. Consider true L3 queue-aware simulation only when a full order-ID feed is available.

**Portfolio wording today:** "Built a reproducible historical limit order book alpha research framework using NASDAQ event and depth data, leak-aware classification and an explicit-spread execution backtester." Add numeric performance only once genuine multi-day experiments have been executed and validated.
