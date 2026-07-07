from __future__ import annotations

import json
from pathlib import Path


def test_mean_var_plot_lcurve_from_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from bayes_infer.plotting.lcurve import mean_var_plot_lcurve

    log = tmp_path / "theta_closed.log"
    log.write_text(
        "1.0e+02 1.0e-04 1.0e+01 6.0e+00 4.0e+00\n"
        "1.0e+01 2.0e-03 2.0e+00 1.5e+00 5.0e-01\n"
    )

    class Args:
        scan_root = None
        theta_logs = [str(log)]
        labels = ["closed"]
        output = str(tmp_path / "lcurve.png")
        table = str(tmp_path / "lcurve.tsv")
        chi2 = "total"
        xscale = "linear"
        yscale = "linear"
        annotate = True
        max_annotations = 10
        title = None

    mean_var_plot_lcurve(Args())
    assert (tmp_path / "lcurve.png").exists()
    table = (tmp_path / "lcurve.tsv").read_text()
    assert "series\ttheta\tS_KL\tchi2" in table
    assert "closed" in table


def test_mean_var_plot_lcurve_from_scan_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from bayes_infer.plotting.lcurve import mean_var_plot_lcurve

    for i in range(2):
        d = tmp_path / "scan" / f"theta_{i:06d}"
        d.mkdir(parents=True)
        (d / "theta_closed.log").write_text(
            f"{100-i:.6g} 1e-4 10 6 4\n"
            f"{1+i:.6g} {0.01*(i+1):.6g} {2-i*0.1:.6g} 1.5 0.5\n"
        )

    class Args:
        scan_root = str(tmp_path / "scan")
        stems = ["closed"]
        theta_logs = None
        labels = None
        output = str(tmp_path / "lcurve_scan.png")
        table = str(tmp_path / "lcurve_scan.tsv")
        chi2 = "total"
        xscale = "linear"
        yscale = "linear"
        annotate = False
        max_annotations = 10
        title = None

    mean_var_plot_lcurve(Args())
    assert (tmp_path / "lcurve_scan.png").exists()
    rows = (tmp_path / "lcurve_scan.tsv").read_text().strip().splitlines()
    assert len(rows) == 3  # header + two final scan points


def test_multi_d_plot_lcurve_from_summaries(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from bayes_infer.plotting.lcurve import multi_d_plot_lcurve

    scan = tmp_path / "multi_scan"
    for i, theta in enumerate([1000.0, 100.0]):
        d = scan / f"theta_{i:06d}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "theta": theta,
            "chi2_half_sum": 1.0 + i,
            "chi2_full_sum": 2.0 + 2*i,
            "entropy_skl": 0.01 * (i + 1),
        }))

    class Args:
        scan_dir = str(scan)
        output = str(tmp_path / "lcurve_multi.png")
        table = str(tmp_path / "lcurve_multi.tsv")
        chi2 = "full"
        xscale = "linear"
        yscale = "linear"
        annotate = False
        max_annotations = 10
        title = None

    multi_d_plot_lcurve(Args())
    assert (tmp_path / "lcurve_multi.png").exists()
    table = (tmp_path / "lcurve_multi.tsv").read_text()
    assert "multi-d" in table
    assert "1000" in table


def test_mean_var_plot_chi2_theta_from_logs(tmp_path):
    from bayes_infer.cli import main
    log = tmp_path / "theta_closed.log"
    log.write_text("1.0 0.5 2.0 1.2 0.8\n2.0 0.3 1.0 0.6 0.4\n")
    out = tmp_path / "chi2_theta.png"
    table = tmp_path / "chi2_theta.tsv"
    assert main([
        "mean-var", "plot-chi2-theta",
        "--theta-logs", str(log),
        "--labels", "closed",
        "--output", str(out),
        "--table", str(table),
        "--target-chi2", "1.2",
    ]) == 0
    assert out.exists() and out.stat().st_size > 0
    assert table.exists() and "closed" in table.read_text()


def test_multi_d_plot_chi2_theta_from_summary(tmp_path):
    from bayes_infer.cli import main
    import json
    d0 = tmp_path / "scan" / "theta_000000"
    d1 = tmp_path / "scan" / "theta_000001"
    d0.mkdir(parents=True)
    d1.mkdir(parents=True)
    (d0 / "summary.json").write_text(json.dumps({"theta": 10.0, "chi2_half_sum": 3.0, "chi2_full_sum": 6.0, "entropy_skl": 0.2}))
    (d1 / "summary.json").write_text(json.dumps({"theta": 100.0, "chi2_half_sum": 2.0, "chi2_full_sum": 4.0, "entropy_skl": 0.1}))
    out = tmp_path / "multi_chi2_theta.png"
    assert main([
        "multi-d", "plot-chi2-theta",
        "--scan-dir", str(tmp_path / "scan"),
        "--output", str(out),
        "--target-chi2", "4.5",
    ]) == 0
    assert out.exists() and out.stat().st_size > 0
