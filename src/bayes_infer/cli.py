"""Command-line interface for bayes-infer."""
from __future__ import annotations

import argparse
from pathlib import Path

from .mean_var import mean_variance as mean_var_fit
from .multi_d import generic_protocol as multi_d_generic
from .plotting import lcurve as lcurve_plot


def _add_mean_var_commands(sub: argparse._SubParsersAction) -> None:
    mean_var = sub.add_parser(
        "mean-var",
        aliases=["mean_var"],
        help="Mean/variance ensemble optimizer",
    )
    mean_var_sub = mean_var.add_subparsers(dest="cmd", required=True)

    run = mean_var_sub.add_parser("run", help="Fit one mean/variance input file")
    run.add_argument("input")
    run.add_argument("-o", "--output", default="output_mean_var.dat")
    run.add_argument("--theta-log", default=None)
    run.add_argument("--ntheta", type=int, default=100)
    run.add_argument("--pow", type=float, default=1.189207115002721)
    run.add_argument("--niter", type=int, default=100)
    run.add_argument("--gtol", type=float, default=1e-8)
    run.add_argument("--verbose", action="store_true")

    batch = mean_var_sub.add_parser("batch", help="Fit multiple mean/variance input files")
    batch.add_argument("inputs", nargs="+")
    batch.add_argument("--output-dir", default=".")
    batch.add_argument("--ntheta", type=int, default=100)
    batch.add_argument("--pow", type=float, default=1.189207115002721)
    batch.add_argument("--niter", type=int, default=100)
    batch.add_argument("--gtol", type=float, default=1e-8)
    batch.add_argument("--verbose", action="store_true")

    scan = mean_var_sub.add_parser("theta-scan", help="Scan target theta values for one or more mean/variance input templates")
    scan.add_argument("--templates", nargs="+", required=True, help="template input files whose first theta line will be replaced")
    scan.add_argument("--start", nargs="+", type=float, required=True, help="first target theta value for each template")
    scan.add_argument("--stop", nargs="+", type=float, required=True, help="last target theta value for each template")
    scan.add_argument("-n", "--num-theta", dest="num_theta", type=int, default=100, help="number of target theta points to scan")
    scan.add_argument("--spacing", choices=["linear", "log"], default="linear", help="spacing of target theta values")
    scan.add_argument("--output-root", dest="output_root", default="mean_var_theta_scan", help="root directory for theta_000000 folders")
    scan.add_argument("--ntheta", type=int, default=100, help="number of internal ramp-down steps for each target theta fit")
    scan.add_argument("--pow", type=float, default=1.189207115002721, help="internal theta-ramp scale factor for each target theta fit")
    scan.add_argument("--niter", type=int, default=100)
    scan.add_argument("--gtol", type=float, default=1e-8)
    scan.add_argument("--verbose", action="store_true")

    plot = mean_var_sub.add_parser("plot-lcurve", help="Plot L-curve from mean-var theta logs or a mean-var theta-scan directory")
    src = plot.add_mutually_exclusive_group(required=True)
    src.add_argument("--scan-root", help="mean_var_theta_scan directory containing theta_*/theta_*.log files")
    src.add_argument("--theta-logs", nargs="+", help="one or more theta log files, e.g. theta_closed.log theta_open.log")
    plot.add_argument("--stems", nargs="+", help="when using --scan-root, plot only these stems, e.g. closed open")
    plot.add_argument("--labels", nargs="+", help="labels for --theta-logs")
    plot.add_argument("--output", default="lcurve_mean_var.png", help="output plot path")
    plot.add_argument("--table", default=None, help="optional TSV table of plotted points")
    plot.add_argument("--chi2", choices=["total", "mean", "variance"], default="total", help="which chi-squared term to plot")
    plot.add_argument("--xscale", choices=["linear", "log"], default="linear")
    plot.add_argument("--yscale", choices=["linear", "log"], default="linear")
    plot.add_argument("--annotate", action="store_true", help="annotate points with theta values")
    plot.add_argument("--max-annotations", type=int, default=50, help="maximum approximate number of theta labels")
    plot.add_argument("--title", default=None)


    plot_theta = mean_var_sub.add_parser("plot-chi2-theta", help="Plot chi-squared versus theta from mean-var theta logs or a theta-scan directory")
    src = plot_theta.add_mutually_exclusive_group(required=True)
    src.add_argument("--scan-root", help="mean_var_theta_scan directory containing theta_*/theta_*.log files")
    src.add_argument("--theta-logs", nargs="+", help="one or more theta log files, e.g. theta_closed.log theta_open.log")
    plot_theta.add_argument("--stems", nargs="+", help="when using --scan-root, plot only these stems, e.g. closed open")
    plot_theta.add_argument("--labels", nargs="+", help="labels for --theta-logs")
    plot_theta.add_argument("--output", default="chi2_vs_theta_mean_var.png", help="output plot path")
    plot_theta.add_argument("--table", default=None, help="optional TSV table of plotted points")
    plot_theta.add_argument("--chi2", choices=["total", "mean", "variance"], default="total", help="which chi-squared term to plot")
    plot_theta.add_argument("--xscale", choices=["linear", "log"], default="linear")
    plot_theta.add_argument("--yscale", choices=["linear", "log"], default="linear")
    plot_theta.add_argument("--annotate", action="store_true", help="annotate points with theta values")
    plot_theta.add_argument("--max-annotations", type=int, default=50, help="maximum approximate number of theta labels")
    plot_theta.add_argument("--target-chi2", type=float, default=None, help="print the theta whose chi2 is closest to this target")
    plot_theta.add_argument("--chi2-cutoff", type=float, default=None, help="highlight/mark points whose chi2 is below this cutoff")
    plot_theta.add_argument("--title", default=None)


def _add_multi_d_commands(sub: argparse._SubParsersAction) -> None:
    multi_d = sub.add_parser(
        "multi-d",
        aliases=["multi_d"],
        help="Multi-dimensional generic protocol",
    )
    multi_d_sub = multi_d.add_subparsers(dest="cmd", required=True)

    make = multi_d_sub.add_parser("make-data", help="Prepare input files from NMR-PRE data")
    make.add_argument("--raw-xls", required=True)
    make.add_argument("--r1rho-xls", required=True)
    make.add_argument("--structures-dir", required=True)
    make.add_argument("--out-dir", required=True)
    make.add_argument("--theta", type=float, default=multi_d_generic.DEFAULT_THETA)
    make.add_argument("--skip-column", type=int, nargs="*", default=[8])
    make.add_argument("--finite-lower", type=float, default=multi_d_generic.prep.FINITE_LOWER_DEFAULT)
    make.add_argument("--finite-upper", type=float, default=multi_d_generic.prep.FINITE_UPPER_DEFAULT)
    make.add_argument("--default-noise", type=float, default=multi_d_generic.prep.DEFAULT_NOISE_DEFAULT)
    make.add_argument("--broadened-distance", type=float, default=multi_d_generic.prep.BROADENED_DISTANCE_DEFAULT)
    make.add_argument("--broadened-noise", type=float, default=multi_d_generic.prep.BROADENED_NOISE_DEFAULT)
    make.add_argument("--long-distance", type=float, default=multi_d_generic.prep.LONG_DISTANCE_DEFAULT)
    make.add_argument("--long-distance-noise", type=float, default=multi_d_generic.prep.LONG_DISTANCE_NOISE_DEFAULT)
    make.add_argument("--expected-last-residue", type=int, default=multi_d_generic.prep.EXPECTED_LAST_RESIDUE_DEFAULT)
    make.add_argument("--atom-mode", choices=["nmr-index", "atom-name"], default="nmr-index")
    make.add_argument("--keep-incomplete-models", action="store_true")
    make.add_argument("--bioen-exe", default="bioen")

    run = multi_d_sub.add_parser("run", help="Run one theta through the default local log-weight backend")
    run.add_argument("--data-dir", required=True)
    run.add_argument("--theta", type=float, required=True)
    run.add_argument("--out-dir", required=True)

    scan = multi_d_sub.add_parser("theta-scan", help="Create and run a theta scan through the default local log-weight backend")
    scan.add_argument("--data-dir", required=True)
    scan.add_argument("--output-root", dest="output_root", required=True, help="root directory for theta_000000 folders")
    scan.add_argument("--start", dest="start", type=float, required=True, help="first target theta value")
    scan.add_argument("--stop", dest="stop", type=float, required=True, help="last target theta value")
    scan.add_argument("-n", "--num-theta", dest="num_theta", type=int, default=80)
    scan.add_argument("--spacing", choices=["log", "linear"], default="log")
    scan.add_argument("--no-run", dest="run", action="store_false", help="Only create theta folders and commands; do not run optimizations")
    scan.set_defaults(run=True)

    plot = multi_d_sub.add_parser("plot-lcurve", help="Plot L-curve from a completed multi-d theta-scan directory")
    plot.add_argument("--scan-dir", required=True, help="theta-scan directory containing theta_*/summary.json files")
    plot.add_argument("--output", default="lcurve_multi_d.png", help="output plot path")
    plot.add_argument("--table", default=None, help="optional TSV table of plotted points")
    plot.add_argument("--chi2", choices=["full", "half"], default="full", help="plot full chi2 or half chi2")
    plot.add_argument("--xscale", choices=["linear", "log"], default="linear")
    plot.add_argument("--yscale", choices=["linear", "log"], default="linear")
    plot.add_argument("--annotate", action="store_true", help="annotate points with theta values")
    plot.add_argument("--max-annotations", type=int, default=50, help="maximum approximate number of theta labels")
    plot.add_argument("--title", default=None)


    plot_theta = multi_d_sub.add_parser("plot-chi2-theta", help="Plot chi-squared versus theta from a completed multi-d theta-scan directory")
    plot_theta.add_argument("--scan-dir", required=True, help="theta-scan directory containing theta_*/summary.json files")
    plot_theta.add_argument("--output", default="chi2_vs_theta_multi_d.png", help="output plot path")
    plot_theta.add_argument("--table", default=None, help="optional TSV table of plotted points")
    plot_theta.add_argument("--chi2", choices=["full", "half"], default="full", help="plot full chi2 or half chi2")
    plot_theta.add_argument("--xscale", choices=["linear", "log"], default="linear")
    plot_theta.add_argument("--yscale", choices=["linear", "log"], default="linear")
    plot_theta.add_argument("--annotate", action="store_true", help="annotate points with theta values")
    plot_theta.add_argument("--max-annotations", type=int, default=50, help="maximum approximate number of theta labels")
    plot_theta.add_argument("--target-chi2", type=float, default=None, help="print the theta whose chi2 is closest to this target")
    plot_theta.add_argument("--chi2-cutoff", type=float, default=None, help="highlight/mark points whose chi2 is below this cutoff")
    plot_theta.add_argument("--title", default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayes-infer",
        description="Bayesian ensemble-inference workflows for mean-var and multi-d data.",
    )
    sub = parser.add_subparsers(dest="system", required=True)
    _add_mean_var_commands(sub)
    _add_multi_d_commands(sub)

    args = parser.parse_args(argv)
    system = args.system.replace("_", "-")

    if system == "mean-var":
        if args.cmd == "run":
            mean_var_fit.run_one(args)
        elif args.cmd == "batch":
            mean_var_fit.batch_run(args)
        elif args.cmd == "theta-scan":
            mean_var_fit.theta_scan(args)
        elif args.cmd == "plot-lcurve":
            lcurve_plot.mean_var_plot_lcurve(args)
        elif args.cmd == "plot-chi2-theta":
            lcurve_plot.mean_var_plot_chi2_theta(args)
        return 0

    if system == "multi-d":
        if args.cmd == "make-data":
            multi_d_generic.make_data(
                Path(args.raw_xls),
                Path(args.r1rho_xls),
                Path(args.structures_dir),
                Path(args.out_dir),
                theta=args.theta,
                skip_column=args.skip_column,
                finite_lower=args.finite_lower,
                finite_upper=args.finite_upper,
                default_noise=args.default_noise,
                broadened_distance=args.broadened_distance,
                broadened_noise=args.broadened_noise,
                long_distance=args.long_distance,
                long_distance_noise=args.long_distance_noise,
                expected_last_residue=args.expected_last_residue,
                atom_mode=args.atom_mode,
                keep_incomplete_models=args.keep_incomplete_models,
                bioen_exe=args.bioen_exe,
            )
        elif args.cmd == "run":
            multi_d_generic.run_one(Path(args.data_dir), args.theta, Path(args.out_dir))
        elif args.cmd == "theta-scan":
            vals = multi_d_generic.generate_theta_values(args.start, args.stop, args.num_theta, args.spacing)
            multi_d_generic.theta_scan(Path(args.data_dir), Path(args.output_root), vals, run=args.run)
        elif args.cmd == "plot-lcurve":
            lcurve_plot.multi_d_plot_lcurve(args)
        elif args.cmd == "plot-chi2-theta":
            lcurve_plot.multi_d_plot_chi2_theta(args)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
