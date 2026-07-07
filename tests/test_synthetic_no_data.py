from __future__ import annotations

from pathlib import Path

import numpy as np

from bayes_infer.cli import main
from bayes_infer.mean_var.mean_variance import fit, read_input
from bayes_infer.multi_d.backends.bioen_log_weights import run_local_generic
from bayes_infer.multi_d.generic_io import load_generic_data


def test_mean_var_synthetic_fit_and_theta_scan(tmp_path):
    inp = tmp_path / "toy.input"
    inp.write_text(
        "1.0\n"
        "0.50 0.10\n"
        "0.020 0.010\n"
        "0.25 0.40\n"
        "0.25 0.45\n"
        "0.25 0.55\n"
        "0.25 0.60\n"
    )
    data = read_input(inp)
    result = fit(data, ntheta=2, pow_factor=1.5, niter=10)
    assert np.all(result.weights > 0)
    assert np.isclose(np.sum(result.weights), 1.0)
    assert np.isfinite(result.chi2)
    assert np.isfinite(result.entropy)

    out_root = tmp_path / "scan"
    rc = main([
        "mean-var", "theta-scan",
        "--templates", str(inp),
        "--start", "0.5",
        "--stop", "1.0",
        "--num-theta", "2",
        "--output-root", str(out_root),
        "--ntheta", "1",
        "--niter", "10",
    ])
    assert rc == 0
    assert (out_root / "theta_000000" / "output_toy.dat").exists()
    assert (out_root / "theta_000001" / "theta_toy.log").exists()
    assert (out_root / "theta_scan_summary.tsv").exists()


def test_multi_d_synthetic_generic_fit(tmp_path):
    data_dir = tmp_path / "BioEN_Files"
    data_dir.mkdir()
    (data_dir / "models-generic.dat").write_text("model_a\nmodel_b\nmodel_c\n")
    (data_dir / "exp-generic.dat").write_text(
        "obs1 1.0 0.5\n"
        "obs2 2.0 0.5\n"
    )
    (data_dir / "sim-obs1-generic.dat").write_text("0.8\n1.0\n1.2\n")
    (data_dir / "sim-obs2-generic.dat").write_text("2.2\n2.0\n1.8\n")

    loaded = load_generic_data(data_dir)
    assert loaded.nmodels == 3
    assert loaded.nrestraints == 2

    result = run_local_generic(data_dir, theta=10.0, out_dir=tmp_path / "fit")
    assert np.all(result.weights > 0)
    assert np.isclose(np.sum(result.weights), 1.0)
    assert np.isfinite(result.chi2)
    assert np.isfinite(result.entropy)
    assert (tmp_path / "fit" / "summary.json").exists()
