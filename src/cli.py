"""Command-line entry point."""
from __future__ import annotations
import argparse
import csv
from .pipeline import run_experiment
from .synthetic import generate
from .book import validate_reconstruction


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lob-alpha", description="Limit order book alpha research")
    sub = parser.add_subparsers(dest="command", required=True)
    p_gen = sub.add_parser("demo-data", help="Generate internally consistent synthetic messages and book")
    p_gen.add_argument("--out", default="data")
    p_gen.add_argument("--events", type=int, default=6000)
    p_gen.add_argument("--seed", type=int, default=42)

    p_check = sub.add_parser("verify-demo", help="Replay complete synthetic L3 messages and verify EVERY book snapshot")
    p_check.add_argument("--message", default="data/synthetic_message.csv")
    p_check.add_argument("--book", default="data/synthetic_orderbook.csv")
    p_check.add_argument("--levels", type=int, default=10)

    p_run = sub.add_parser("run", help="Run ML training, temporal evaluation and aggressive backtest")
    p_run.add_argument("--messages", nargs="+", required=True, help="One message CSV per trading day, chronological order")
    p_run.add_argument("--books", nargs="+", required=True, help="Corresponding orderbook CSV files")
    p_run.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    p_run.add_argument("--features", choices=["all", "obi", "ofi", "trade_flow"], default="all")
    p_run.add_argument("--horizons", nargs="+", type=int, default=[10, 50, 100])
    p_run.add_argument("--latency", type=int, default=0, help="Extra events after minimum t+1 reaction")
    p_run.add_argument("--shares", type=int, default=10)
    p_run.add_argument("--fee", type=float, default=0.003, help="Per share, per side, USD")
    p_run.add_argument("--tick-size", type=float, default=0.01)
    p_run.add_argument("--flat-threshold-ticks", type=float, default=0.5)
    p_run.add_argument("--max-rows", type=int)
    p_run.add_argument("--out", default="reports")

    args = parser.parse_args(argv)
    if args.command == "demo-data":
        a, b = generate(args.out, args.events, seed=args.seed)
        print(f"Synthetic DEMO ONLY:\n  Messages: {a}\n  Snapshots: {b}")
    elif args.command == "verify-demo":
        with open(args.message, newline="") as f:
            messages = list(csv.reader(f))
        with open(args.book, newline="") as f:
            snapshots = list(csv.reader(f))
        result = validate_reconstruction(messages, snapshots, levels=args.levels)
        print(result)
        if not result["matched"]:
            raise SystemExit(1)
    elif args.command == "run":
        run_experiment(args.messages, args.books, output_dir=args.out,
                       horizons=tuple(args.horizons), model=args.model, feature_set=args.features,
                       latency_events=args.latency, shares=args.shares,
                       fee_per_share=args.fee, max_rows=args.max_rows,
                       tick_size=args.tick_size, flat_threshold_ticks=args.flat_threshold_ticks)


if __name__ == "__main__":
    main()
