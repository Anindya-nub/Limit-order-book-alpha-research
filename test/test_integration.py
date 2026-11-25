from lob_alpha.synthetic import generate
from lob_alpha.pipeline import run_experiment


def test_end_to_end_demo(tmp_path):
    mp, bp = generate(tmp_path, events=850, seed=5)
    result = run_experiment([str(mp)], [str(bp)], output_dir=str(tmp_path / "reports"), horizons=(10, 50))
    assert result["mode"] == "SYNTHETIC_DEMO_ONLY"
    assert set(result["experiments"]) == {"10", "50"}
    assert (tmp_path / "reports" / "results.json").exists()
