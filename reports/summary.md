# LOB Alpha Research — Experiment Summary

Data mode: **SYNTHETIC_DEMO_ONLY**; model: **logistic**; feature set: **ofi**.

**Important:** Outputs from synthetic data demonstrate software behavior, NOT market predictability.
A single trading session does not provide independent day-level out-of-sample evidence.
The execution simulation assumes immediate full fills at displayed quotes and excludes market impact.

| Horizon (events) | Test accuracy | Test macro-F1 | Test log loss | Spearman IC | Trades | Net P&L (USD) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.492 | 0.373 | 1.024 | 0.214 | 20 | -5.90 |

## Audit

See `results.json` for snapshot validation and per-horizon confusion matrices; the trade log contains event indices and actual entry/exit quote prices.
