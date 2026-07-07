from pathlib import Path

from bayes_infer.cli import main


def test_multi_d_theta_scan_uses_standard_option_names(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "models-generic.dat").write_text("1 1.0\n", encoding="utf-8")
    (data_dir / "exp-generic.dat").write_text("obs 1.0 0.1\n", encoding="utf-8")
    (data_dir / "sim-obs-generic.dat").write_text("1.0\n", encoding="utf-8")
    out = tmp_path / "scan"
    rc = main([
        "multi-d", "theta-scan",
        "--data-dir", str(data_dir),
        "--start", "1",
        "--stop", "10",
        "--num-theta", "2",
        "--spacing", "linear",
        "--output-root", str(out),
    ])
    assert rc == 0
    assert (out / "theta_000000" / "thetas.dat").exists()
    assert (out / "theta_000001" / "thetas.dat").exists()
