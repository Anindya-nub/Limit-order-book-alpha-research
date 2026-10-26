import numpy as np
import pandas as pd
from lob_alpha.features import make_features
from lob_alpha.labels import add_labels, temporal_split


def sample_frames(n=400):
    # Two-sided book that changes mid at an event; direction code is resting side.
    return pd.DataFrame({
        "day": ["D1"] * n,
        "time": np.arange(n),
        "event_type": [1] * n,
        "direction": [1] * n,
        "size": [100] * n,
        "ask_price_1": 100.01 + np.arange(n) * .01,
        "bid_price_1": 100.00 + np.arange(n) * .01,
        "ask_size_1": [60] * n,
        "bid_size_1": [140] * n,
    })


def test_obi_microprice_and_features_are_past_only():
    raw = sample_frames()
    features = make_features(raw)
    assert np.isclose(features.loc[100, "obi_1"], .4)
    assert features.loc[100, "microprice"] > features.loc[100, "mid"]
    # Changing future rows must not alter the features of an earlier event.
    raw.loc[101:, "bid_size_1"] = 9999
    updated = make_features(raw)
    for f in ["obi_1", "microprice_deviation_bps", "ofi_normalized", "signed_trade_pressure_50"]:
        assert np.isclose(features.loc[100, f], updated.loc[100, f], equal_nan=True)


def test_labels_and_split_are_purged_at_boundaries():
    frame = add_labels(make_features(sample_frames()), (10, 50))
    split = temporal_split(frame, 50)
    assert split["train"].max() + 50 < split["validation"].min()
    assert split["validation"].max() + 50 < split["test"].min()
    assert frame["label_50"].iloc[-50:].isna().all()


def test_snapshot_event_delta_audit(tmp_path):
    from lob_alpha.synthetic import generate
    from lob_alpha.data import read_lobster, check_visible_event_deltas
    m, b = generate(tmp_path, events=120)
    result = check_visible_event_deltas(read_lobster(m, b))
    assert result["eligible_events"] > 30
    assert result["mismatched_events"] == 0
