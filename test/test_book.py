import csv
from lob_alpha.book import OrderBook, validate_reconstruction
from lob_alpha.synthetic import generate


def test_order_book_add_cancel_execute_and_delete():
    book = OrderBook()
    book.apply(1, 1, 100, 999900, 1)
    book.apply(1, 2, 60, 1000100, -1)
    assert book.snapshot(1) == [1000100, 60, 999900, 100]
    book.apply(2, 1, 40, 999900, 1)
    book.apply(4, 2, 20, 1000100, -1)
    assert book.snapshot(1) == [1000100, 40, 999900, 60]
    book.apply(3, 1, 60, 999900, 1)
    assert book.snapshot(1) == [1000100, 40, -9999999999, 0]


def test_synthetic_stream_reconstructs_every_snapshot(tmp_path):
    mp, bp = generate(tmp_path, events=300, seed=10)
    with mp.open() as f:
        messages = list(csv.reader(f))
    with bp.open() as f:
        books = list(csv.reader(f))
    summary = validate_reconstruction(messages, books, levels=10)
    assert summary["matched"]
    assert summary["events_checked"] >= 300
