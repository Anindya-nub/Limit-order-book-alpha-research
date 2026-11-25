# Validation

- Software checks: `python -m pytest -q` — 12 passed. One warning originates from the environment's external AnyIO pytest plugin.
- Real-data run: `python -m lob_alpha run-real --data-dir data/raw --symbol AAPL --levels 10 --model logistic --out reports/aapl_full_session` — completed at all three horizons.
- All 400,391 real observations were loaded; no row cap or synthetic input.
- SHA-256 checksums of the delivered CSV files match both the download manifest and experiment report.
- No missing top quotes, locked/crossed books, non-monotonic depth, unexpected event codes or time reversals were detected.
- All 178,116 eligible visible-event volume transitions matched. This is a partial depth audit, not proof of complete L3 reconstruction.
- The single-session evaluation and simplified execution assumptions limit interpretation. All three simulated net returns were negative; no profitability or current-market claim is made.
