# LOB Alpha Research — Experiment Summary

Data mode: **SYNTHETIC_DEMO_ONLY**; model: **logistic**; feature set: **obi**.

**Important:** Outputs from synthetic data demonstrate software behavior, NOT market predictability.
A single trading session does not provide independent day-level out-of-sample evidence.
The execution simulation assumes immediate full fills at displayed quotes and excludes market impact.

| Horizon (events) | Test accuracy | Test macro-F1 | Test log loss | Spearman IC | Trades | Net P&L (USD) |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.537 | 0.397 | 1.003 | 0.257 | 21 | -5.56 |

## Audit

See `results.json` for snapshot validation and per-horizon confusion matrices; the trade log contains event indices and actual entry/exit quote prices.
