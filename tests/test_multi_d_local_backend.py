from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bayes_infer import cli
from bayes_infer.multi_d import nmr_prep
from bayes_infer.multi_d.generic_io import load_generic_data, resolve_data_ids
from bayes_infer.multi_d.nmr_prep import Atom, build_restraints_nmr, nmr_index_distance, residue_anchor_indices
from bayes_infer.multi_d.backends.bioen_log_weights import (
    bioen_log_posterior_base,
    grad_bioen_log_posterior_base,
    run_local_generic,
    uniform_weights,
)
from bayes_infer.plotting.lcurve import read_multi_d_theta_scan


def write_generic_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "generic_data"
    data_dir.mkdir()
    (data_dir / "exp-generic.dat").write_text(
        "#ID measurement noise\n"
        "noe_1 5.2 0.5\n"
        "noe_2 4.6 0.7\n"
        "distance_1 14.2 3.5\n"
        "pre_1 19.2 5.5\n"
        "r2-J1JNCA 11.36 0.59\n"
    )
    (data_dir / "models-generic.dat").write_text("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n")
    values = {
        "noe_1": [5.3, 4.4, 3.7, 5.6, 5.8, 7.1, 8.3, 5.4, 3.9, 2.9],
        "noe_2": [4.6, 4.4, 6.3, 5.6, 5.1, 4.1, 8.3, 3.7, 3.0, 2.9],
        "distance_1": [14.3, 14.4, 16.6, 15.6, 15.5, 14.1, 8.3, 13.7, 13.9, 22.1],
        "pre_1": [15.3, 4.3, 13.7, 5.6, 15.8, 17.9, 18.3, 5.4, 13.9, 2.9],
        "r2-J1JNCA": [
            1.173975165357293804e1,
            1.188630729167003963e1,
            1.171892605325109038e1,
            1.133960260055274816e1,
            1.122317872360144619e1,
            1.172723176304797121e1,
            1.158267066162416903e1,
            1.112580013820116953e1,
            1.213680852487016537e1,
            1.169117713080245835e1,
        ],
    }
    for data_id, rows in values.items():
        body = "#for_each_model_single_measurement\n" + "\n".join(str(x) for x in rows) + "\n"
        (data_dir / f"sim-{data_id}-generic.dat").write_text(body)
    return data_dir


def parse_multi_d_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="system", required=True)
    cli._add_multi_d_commands(sub)
    return parser.parse_args(argv)


def test_multi_d_theta_scan_runs_by_default():
    args = parse_multi_d_args([
        "multi-d",
        "theta-scan",
        "--data-dir",
        "multi_d_input",
        "--output-root",
        "theta_scan",
        "--start",
        "1e-2",
        "--stop",
        "1e5",
    ])
    assert args.run is True


def test_multi_d_theta_scan_no_run_only_prepares_commands():
    args = parse_multi_d_args([
        "multi-d",
        "theta-scan",
        "--data-dir",
        "multi_d_input",
        "--output-root",
        "theta_scan",
        "--start",
        "1e-2",
        "--stop",
        "1e5",
        "--no-run",
    ])
    assert args.run is False


def test_generic_reader_resolves_prefix_matched_id(tmp_path):
    data_dir = write_generic_data(tmp_path)
    assert resolve_data_ids(data_dir, "r2-J1JNC") == ("r2-J1JNCA",)


def test_nmr_r1rho_spreadsheet_keeps_numeric_first_row(monkeypatch):
    raw_df = pd.DataFrame({
        "Assignment": ["R5N-H"],
        "MTSL at C15_2 15uM": [0.07267053640000001],
    })
    r1rho_df = pd.DataFrame([
        [1, 3.0],
        [2, 3.0],
        [3, 3.0],
        [4, 3.0],
        [5, 3.0],
        [6, 3.2],
    ])

    def fake_read_excel(path, *args, **kwargs):
        if str(path) == "raw.xls":
            return raw_df
        if str(path) == "r1rho.xls":
            assert kwargs.get("header") is None
            return r1rho_df
        raise AssertionError(f"Unexpected spreadsheet path: {path}")

    monkeypatch.setattr(nmr_prep.pd, "read_excel", fake_read_excel)

    restraints = build_restraints_nmr(Path("raw.xls"), Path("r1rho.xls"))

    assert len(restraints) == 1
    assert restraints[0].data_id == "15_5"
    assert np.isclose(restraints[0].measurement, 13.775319973892492)


def test_nmr_residue_anchor_indices_match_distance_offsets():
    atoms = [
        Atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0),
        Atom(2, "CA", "ALA", "A", 1, 10.0, 0.0, 0.0),
        Atom(3, "C", "ALA", "A", 1, 20.0, 0.0, 0.0),
        Atom(4, "O", "ALA", "A", 1, 30.0, 0.0, 0.0),
        Atom(5, "N", "GLY", "A", 2, 40.0, 0.0, 0.0),
        Atom(6, "CA", "GLY", "A", 2, 55.0, 0.0, 0.0),
        Atom(7, "CB", "ALA", "A", 1, 23.0, 0.0, 0.0),
        Atom(8, "C", "GLY", "A", 2, 70.0, 0.0, 0.0),
        Atom(9, "O", "GLY", "A", 2, 80.0, 0.0, 0.0),
        Atom(10, "CB", "GLY", "A", 2, 60.0, 0.0, 0.0),
        Atom(11, "H", "GLY", "A", 2, 100.0, 0.0, 0.0),
    ]
    anchors, max_resseq = residue_anchor_indices(atoms)
    assert max_resseq == 2
    assert anchors[1] == 0
    assert anchors[2] == 3
    assert nmr_index_distance(atoms, anchors, probe=1, query=1) == 3.0
    assert nmr_index_distance(atoms, anchors, probe=2, query=2) == 5.0


def test_default_log_weight_gradient_matches_finite_difference(tmp_path):
    data = load_generic_data(write_generic_data(tmp_path))
    w0 = uniform_weights(data.nmodels)
    log_weights = np.array([0.01, -0.02, 0.03, -0.04, 0.05, 0.02, -0.01, 0.04, -0.03, 0.00])
    theta = 1000.0
    grad = grad_bioen_log_posterior_base(log_weights, w0, data.sim_scaled, data.exp_scaled, theta)
    num = np.zeros_like(grad)
    step = 1e-6
    for i in range(len(log_weights)):
        plus = log_weights.copy()
        plus[i] += step
        minus = log_weights.copy()
        minus[i] -= step
        num[i] = (
            bioen_log_posterior_base(plus, w0, data.sim_scaled, data.exp_scaled, theta)
            - bioen_log_posterior_base(minus, w0, data.sim_scaled, data.exp_scaled, theta)
        ) / (2.0 * step)
    assert np.allclose(grad, num, rtol=2e-5, atol=2e-5)


def test_default_log_weight_backend_matches_generic_reference_values(tmp_path):
    # These values are generated by the local BioEn-compatible default
    # log-weight objective. BioEn reports chi2 as 0.5 * sum(residual^2).
    data_dir = write_generic_data(tmp_path)
    expected = {
        1_000_000.0: (1.185198146888505, 7.723826747630006e-13),
        1000.0: (1.1836515218050114, 7.691298481900266e-07),
    }
    for theta, (expected_chi2, expected_s) in expected.items():
        out = tmp_path / f"theta_{theta:g}"
        result = run_local_generic(data_dir, theta, out)
        assert np.isclose(result.chi2, expected_chi2, rtol=5e-8, atol=5e-8)
        assert np.isclose(result.entropy, expected_s, rtol=5e-5, atol=5e-12)
        saved = json.loads((out / "summary.json").read_text())
        assert saved["backend"] == "local-log-weights"
        assert saved["nmodels"] == 10
        assert saved["nrestraints"] == 5
        assert saved["chi2_half_sum"] == result.chi2
        assert (out / "result.pkl").exists()


def test_multi_d_plot_reader_uses_current_summary_keys(tmp_path):
    scan_dir = tmp_path / "theta_scan"
    theta_dir = scan_dir / "theta_000000"
    theta_dir.mkdir(parents=True)
    (theta_dir / "summary.json").write_text(json.dumps({
        "backend": "local-log-weights",
        "theta": 3000.0,
        "chi2_half_sum": 1.25,
        "chi2_full_sum": 2.5,
        "entropy_skl": 0.01,
    }))

    half = read_multi_d_theta_scan(scan_dir, chi2="half")["multi-d"]
    full = read_multi_d_theta_scan(scan_dir, chi2="full")["multi-d"]

    assert len(half) == 1
    assert half[0].theta == 3000.0
    assert half[0].chi2_total == 1.25
    assert full[0].chi2_total == 2.5
