import numpy as np
import pandas as pd
from lob_alpha.execution import aggressive_backtest


def test_buy_crosses_spread_both_sides_and_pays_fees():
    f = pd.DataFrame({"day": ["D"] * 20, "time": range(20),
                      "bid_price_1": [99.99] * 20, "ask_price_1": [100.01] * 20,
                      "bid_size_1": [200] * 20, "ask_size_1": [200] * 20})
    f.loc[1:, ["bid_price_1", "ask_price_1"]] += 0.02
    trades, metrics = aggressive_backtest(f, np.array([0]), np.array([[.1, .1, .8]]),
                                          horizon=5, shares=10, fee_per_share=.003)
    assert len(trades) == 1
    # Entry at t+1=100.03 ask, exit at t+6=100.01 bid: -0.20 - fees 0.06
    assert np.isclose(metrics["net_pnl_usd"], -.26)
    assert trades.iloc[0]["entry_index"] == 1


def test_no_cross_session_trade():
    f = pd.DataFrame({"day": ["D1"] * 5 + ["D2"] * 10, "time": range(15),
                      "bid_price_1": [100.] * 15, "ask_price_1": [100.01] * 15,
                      "bid_size_1": [100] * 15, "ask_size_1": [100] * 15})
    trades, metrics = aggressive_backtest(f, np.array([3]), np.array([[.1, .1, .8]]), horizon=5)
    assert len(trades) == 0
