"""Synthetic event stream for smoke testing ONLY; not alpha evidence."""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
from .book import OrderBook


def generate(output_dir: str | Path, events: int = 6000, levels: int = 10, seed: int = 42) -> tuple[Path, Path]:
    """Generate internally consistent event file + post-message book snapshots."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    msg_path = path / "synthetic_message.csv"
    book_path = path / "synthetic_orderbook.csv"
    rng = np.random.default_rng(seed)
    lob = OrderBook()
    next_id = 1
    step = 0
    latent = 0.
    with msg_path.open("w", newline="") as mf, book_path.open("w", newline="") as bf:
        mw, bw = csv.writer(mf), csv.writer(bf)

        def write(typ, oid, sz, price, side):
            nonlocal step
            lob.apply(typ, oid, sz, price, side)
            mw.writerow([f"{34200 + step * 0.03:.5f}", typ, oid, sz, price, side])
            bw.writerow(lob.snapshot(levels))
            step += 1

        # Opening state built explicitly from events, so replay from empty works.
        for k in range(levels + 12):
            write(1, next_id, int(rng.integers(80, 240)), 1_000_000 - 100 * k, 1)
            next_id += 1
            write(1, next_id, int(rng.integers(80, 240)), 1_000_100 + 100 * k, -1)
            next_id += 1

        for _ in range(events):
            latent = 0.985 * latent + 0.08 * rng.normal()
            # Resting ask execution is a buyer-initiated trade.
            ask_exec_prob = 1 / (1 + np.exp(-2.5 * latent))
            r = rng.random()
            if r < 0.32 or not lob.orders or not lob.bids or not lob.asks:
                side = (1 if not lob.bids else -1 if not lob.asks else 1 if rng.random() < 0.5 else -1)
                best_bid = max(lob.bids) if lob.bids else min(lob.asks) - 100
                best_ask = min(lob.asks) if lob.asks else max(lob.bids) + 100
                d = int(rng.choice([0, 0, 0, 1, 2, 3]))
                if side == 1:
                    px = best_bid - 100 * d
                    if best_ask - best_bid > 100 and rng.random() < 0.12:
                        px = best_bid + 100
                    px = min(px, best_ask - 100)
                else:
                    px = best_ask + 100 * d
                    if best_ask - best_bid > 100 and rng.random() < 0.12:
                        px = best_ask - 100
                    px = max(px, best_bid + 100)
                write(1, next_id, int(rng.integers(25, 200)), px, side)
                next_id += 1
            elif r < 0.91:
                side = -1 if rng.random() < ask_exec_prob else 1
                levels_for_side = lob.asks if side == -1 else lob.bids
                if not levels_for_side:
                    continue
                best = min(levels_for_side) if side == -1 else max(levels_for_side)
                candidates = [i for i, o in lob.orders.items() if o.side == side and o.price == best]
                if not candidates:
                    continue
                oid = int(rng.choice(candidates))
                o = lob.orders[oid]
                write(4, oid, min(o.remaining, int(rng.integers(20, 200))), o.price, o.side)
            else:
                oid = int(rng.choice(list(lob.orders)))
                o = lob.orders[oid]
                cut = o.remaining if rng.random() < 0.55 else min(o.remaining, int(rng.integers(1, 80)))
                typ = 3 if cut == o.remaining else 2
                write(typ, oid, cut, o.price, o.side)
    return msg_path, book_path
