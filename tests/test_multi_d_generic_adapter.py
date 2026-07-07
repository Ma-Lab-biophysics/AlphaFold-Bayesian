from __future__ import annotations

from pathlib import Path

from bayes_infer.cli import main
from bayes_infer.multi_d import generic_protocol, nmr_prep


def write_generic_data(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_dir.joinpath("exp-generic.dat").write_text(
        "\n".join([
            "noe_1 1.0 0.1",
            "noe_2 2.0 0.1",
            "distance_1 3.0 0.1",
            "pre_1 4.0 0.1",
            "r2-J1JNCA 5.0 0.1",
        ]) + "\n"
    )
    data_dir.joinpath("models-generic.dat").write_text(
        "\n".join(f"model_{i:02d} 0.0 0.0 0.0 0.0 0.0" for i in range(10)) + "\n"
    )
    return data_dir


def test_generic_data_ids_and_models(tmp_path):
    data_dir = write_generic_data(tmp_path)
    assert generic_protocol.count_models(data_dir) == 10
    assert generic_protocol.read_data_ids_from_exp(data_dir) == [
        "noe_1", "noe_2", "distance_1", "pre_1", "r2-J1JNCA"
    ]


def test_theta_scan_writes_default_log_weight_commands(tmp_path):
    data_dir = write_generic_data(tmp_path).resolve()
    out_dir = tmp_path / "scan"
    generic_protocol.theta_scan(data_dir, out_dir, [1000.0], run=False)
    commands = (out_dir / "theta_scan_commands.tsv").read_text()
    assert "local-log-weights" in commands
    assert "bayes-infer multi-d run" in commands
    assert "--backend" not in commands
    assert (out_dir / "theta_000000" / "thetas.dat").read_text().strip() == "1000"


def test_nmr_prep_direct_cli_uses_shared_theta_default():
    parser = nmr_prep.build_arg_parser()
    make_args = parser.parse_args([
        "make-data",
        "--raw-xls", "raw.xls",
        "--r1rho-xls", "r1rho.xls",
        "--structures-dir", "structures",
    ])
    run_args = parser.parse_args(["write-run", "--data-dir", "BioEN_Files"])
    assert nmr_prep.DEFAULT_THETA == 3000.0
    assert generic_protocol.DEFAULT_THETA == nmr_prep.DEFAULT_THETA
    assert make_args.default_theta == nmr_prep.DEFAULT_THETA
    assert run_args.theta == nmr_prep.DEFAULT_THETA


def test_cli_multi_d_make_data_has_no_drop_missing_option(capsys):
    try:
        main(["multi-d", "make-data", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    text = capsys.readouterr().out
    assert "--keep-incomplete-models" in text
    assert "--drop-missing" not in text


def test_cli_multi_d_run_has_no_backend_option(capsys):
    try:
        main(["multi-d", "run", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    text = capsys.readouterr().out
    assert "--data-dir" in text
    assert "--theta" in text
    assert "--out-dir" in text
    assert "--backend" not in text
