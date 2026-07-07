from pathlib import Path

from bayes_infer.mean_var.mean_variance import fit, read_input


def test_read_open_example():
    root = Path(__file__).resolve().parents[1]
    data = read_input(root / "examples" / "mean_var" / "open.input")
    assert data.theta > 0
    assert len(data.y) > 0


def test_short_fit_runs():
    root = Path(__file__).resolve().parents[1]
    data = read_input(root / "examples" / "mean_var" / "open.input")
    result = fit(data, ntheta=1, pow_factor=1.5, niter=3)
    assert len(result.weights) == len(data.y)
