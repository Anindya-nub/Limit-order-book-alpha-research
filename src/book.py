"""Exact order-ID book for complete, ordered event streams.

IMPORTANT: LOBSTER level-truncated sample files may omit prior and out-of-range
orders. Their snapshots should be treated as authoritative; do not call this
an exact full-history reconstruction of a partial sample.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Order:
    side: int  # 1 bid; -1 ask
    price: int  # fixed point price, e.g. $100.01 -> 1000100
    remaining: int


class OrderBook:
    def __init__(self):
        self.orders: dict[int, Order] = {}
        self.bids: dict[int, int] = defaultdict(int)
        self.asks: dict[int, int] = defaultdict(int)

    def apply(self, event_type: int, order_id: int, size: int, price: int, direction: int):
        """Apply LOBSTER event in integer price units (not USD floats)."""
        if event_type == 1:
            if order_id in self.orders or size <= 0 or direction not in (-1, 1):
                raise ValueError(f"Invalid new order id={order_id}")
            self.orders[order_id] = Order(int(direction), int(price), int(size))
            side = self.bids if direction == 1 else self.asks
            side[int(price)] += int(size)
        elif event_type in (2, 3, 4):
            if order_id not in self.orders:
                raise KeyError(f"Unknown order {order_id}; missing history or bad input")
            o = self.orders[order_id]
            if o.price != int(price) or o.side != int(direction):
                raise ValueError(f"Order {order_id} changed price or side")
            if event_type == 3 and int(size) != o.remaining:
                raise ValueError(f"Full deletion size mismatch for {order_id}")
            if size <= 0 or int(size) > o.remaining:
                raise ValueError(f"Size exceeds remaining for {order_id}")
            side = self.bids if o.side == 1 else self.asks
            side[o.price] -= int(size)
            if side[o.price] == 0:
                del side[o.price]
            o.remaining -= int(size)
            if o.remaining == 0:
                del self.orders[order_id]
        elif event_type in (5, 6, 7):
            # Hidden executions/cross trades/halts do not modify visible depth.
            return
        else:
            raise ValueError(f"Unknown event type {event_type}")

    def snapshot(self, levels: int = 10) -> list[int]:
        """LOBSTER's [ask_px, ask_sz, bid_px, bid_sz] repeated layout."""
        asks = sorted(self.asks.items())[:levels]
        bids = sorted(self.bids.items(), reverse=True)[:levels]
        row = []
        for i in range(levels):
            ap, az = asks[i] if i < len(asks) else (9999999999, 0)
            bp, bz = bids[i] if i < len(bids) else (-9999999999, 0)
            row.extend([ap, az, bp, bz])
        return row


def validate_reconstruction(messages, snapshots, levels: int = 10) -> dict:
    """Check every snapshot; only valid on a complete stream from empty state."""
    engine = OrderBook()
    n = 0
    errors = []
    for msg, snapshot in zip(messages, snapshots):
        engine.apply(int(msg[1]), int(msg[2]), int(msg[3]), int(msg[4]), int(msg[5]))
        if engine.snapshot(levels) != [int(x) for x in snapshot]:
            if len(errors) < 5:
                errors.append(n)
        n += 1
    return {"events_checked": n, "first_mismatch_rows": errors, "matched": len(errors) == 0}
