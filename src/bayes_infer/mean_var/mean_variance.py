#!/usr/bin/env python3
"""
Mean/variance ensemble optimizer.

Input format:
    line 1: theta
    line 2: experimental mean, SE(mean)
    line 3: experimental variance, SE(variance)
    remaining lines: prior_weight_i observable_i

The optimizer refines normalized ensemble weights w_i using mean and variance restraints.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, TextIO

import numpy as np


@dataclass
class MeanVarInput:
    theta: float
    y_obs: float
    sigma: float
    var_obs: float
    sigma_var: float
    prior: np.ndarray
    y: np.ndarray


@dataclass
class Moments:
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    s0: float
    s1: float
    s2: float
    s3: float
    s4: float
    w: np.ndarray
    log_w_over_p: np.ndarray


@dataclass
class HessianResult:
    xl: float
    xlf: float
    xlg: float
    xlff: float
    xlfg: float
    xlgg: float
    det: float
    moments: Moments


@dataclass
class FitResult:
    f: float
    g: float
    theta: float
    weights: np.ndarray
    log_w_over_p: np.ndarray
    chi2: float
    chi2_mean: float
    chi2_variance: float
    entropy: float
    det: float
    theta_trace: list[tuple[float, float, float, float, float]]
    input_data: MeanVarInput


def read_input(path: str | os.PathLike[str] | TextIO) -> MeanVarInput:
    """Read mean/variance input from a file path or text stream."""
    should_close = False
    if hasattr(path, "read"):
        handle = path  # type: ignore[assignment]
    else:
        handle = open(path, "r", encoding="utf-8")
        should_close = True

    try:
        lines = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]
    finally:
        if should_close:
            handle.close()

    if len(lines) < 4:
        raise ValueError("Input must contain theta, mean/SE, variance/SEV, and at least one prior/observable row.")

    theta = float(lines[0].split()[0])
    y_obs, sigma = map(float, lines[1].split()[:2])
    var_obs, sigma_var = map(float, lines[2].split()[:2])

    rows = []
    for lineno, line in enumerate(lines[3:], start=4):
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Line {lineno} must contain prior and observable values: {line!r}")
        rows.append((float(fields[0]), float(fields[1])))

    arr = np.asarray(rows, dtype=np.float64)
    prior = arr[:, 0]
    y = arr[:, 1]

    if np.any(prior <= 0.0):
        bad = int(np.nonzero(prior <= 0.0)[0][0]) + 1
        raise ValueError(f"Input error: prior({bad}) <= 0: {prior[bad - 1]}")
    if sigma <= 0.0 or sigma_var <= 0.0:
        raise ValueError("Mean SE and variance SE must both be positive.")

    prior = prior / np.sum(prior)
    return MeanVarInput(theta, y_obs, sigma, var_obs, sigma_var, prior, y)


def setup_weights(f: float, g: float, prior: np.ndarray, y: np.ndarray) -> Moments:
    """Compute normalized weights, moments, and entropy-weighted moments."""
    log_factor = (f + g * y) * y
    max_log = float(np.max(log_factor))
    shifted = log_factor - max_log
    unnormalized = prior * np.exp(shifted)
    total = float(np.sum(unnormalized))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Weight normalization failed; try a larger theta or inspect input values.")

    w = unnormalized / total
    log_w_over_p = shifted - math.log(total)

    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    y5 = y4 * y
    y6 = y3 * y3

    c1 = float(np.sum(w * y))
    c2 = float(np.sum(w * y2))
    c3 = float(np.sum(w * y3))
    c4 = float(np.sum(w * y4))
    c5 = float(np.sum(w * y5))
    c6 = float(np.sum(w * y6))

    wxl = w * log_w_over_p
    s0 = float(np.sum(wxl))
    s1 = float(np.sum(wxl * y))
    s2 = float(np.sum(wxl * y2))
    s3 = float(np.sum(wxl * y3))
    s4 = float(np.sum(wxl * y4))

    return Moments(c1, c2, c3, c4, c5, c6, s0, s1, s2, s3, s4, w, log_w_over_p)


def hessian(f: float, g: float, data: MeanVarInput, theta: float) -> HessianResult:
    """Return log-posterior, gradient, Hessian, and determinant."""
    m = setup_weights(f, g, data.prior, data.y)
    c1, c2, c3, c4, c5, c6 = m.c1, m.c2, m.c3, m.c4, m.c5, m.c6
    s0, s1, s2, s3, s4 = m.s0, m.s1, m.s2, m.s3, m.s4
    YObs = data.y_obs
    sig2 = data.sigma ** 2
    VarObs = data.var_obs
    varsig2 = data.sigma_var ** 2

    xl = 0.5 * (YObs - c1) ** 2 / sig2 + 0.5 * (VarObs + c1 ** 2 - c2) ** 2 / varsig2 + theta * s0

    xlf = ((YObs - c1) * (c1 ** 2 - c2)) / sig2 - ((VarObs + c1 ** 2 - c2) * (2 * c1 ** 3 - 3 * c1 * c2 + c3)) / varsig2 + theta * (-(c1 * s0) + s1)

    xlg = ((-YObs + c1) * (-(c1 * c2) + c3)) / sig2 - ((VarObs + c1 ** 2 - c2) * (2 * c1 ** 2 * c2 - c2 ** 2 - 2 * c1 * c3 + c4)) / varsig2 + theta * (-(c2 * s0) + s2)

    xlff = (
        10 * sig2 * c1 ** 6
        + (3 * VarObs * sig2 + varsig2) * c2 ** 2
        - 3 * sig2 * c2 ** 3
        + 3 * c1 ** 4 * (2 * VarObs * sig2 + varsig2 - 10 * sig2 * c2)
        - varsig2 * YObs * c3
        + sig2 * c3 ** 2
        + c1 ** 3 * (-2 * varsig2 * YObs + 8 * sig2 * c3)
        - VarObs * sig2 * c4
        + sig2 * c2 * (theta * varsig2 + c4 - theta * varsig2 * s0)
        + c1 ** 2 * (-(12 * VarObs * sig2 + 5 * varsig2) * c2 + 24 * sig2 * c2 ** 2 - sig2 * (theta * varsig2 + c4) + 2 * sig2 * theta * varsig2 * s0)
        + c1 * ((4 * VarObs * sig2 + varsig2) * c3 + c2 * (3 * varsig2 * YObs - 10 * sig2 * c3) - 2 * sig2 * theta * varsig2 * s1)
        + sig2 * theta * varsig2 * s2
    ) / (sig2 * varsig2)

    xlfg = (
        10 * sig2 * c1 ** 5 * c2
        + sig2 * theta * varsig2 * c3
        - 10 * sig2 * c1 ** 4 * c3
        + c2 ** 2 * (varsig2 * YObs - 5 * sig2 * c3)
        - varsig2 * YObs * c4
        + sig2 * c3 * c4
        + c1 ** 3 * (c2 * (6 * VarObs * sig2 + 3 * varsig2 - 20 * sig2 * c2) + 5 * sig2 * c4)
        - VarObs * sig2 * c5
        - c1 ** 2 * (3 * (2 * VarObs * sig2 + varsig2) * c3 + 2 * c2 * (varsig2 * YObs - 9 * sig2 * c3) + sig2 * c5)
        - sig2 * theta * varsig2 * c3 * s0
        + c2 * ((4 * VarObs * sig2 + varsig2) * c3 + sig2 * (c5 - theta * varsig2 * s1))
        + c1 * (-2 * (3 * VarObs * sig2 + varsig2) * c2 ** 2 + 9 * sig2 * c2 ** 3 + 2 * varsig2 * YObs * c3 - 2 * sig2 * c3 ** 2 + (3 * VarObs * sig2 + varsig2) * c4 + sig2 * c2 * (-6 * c4 + theta * varsig2 * (-1 + 2 * s0)) - sig2 * theta * varsig2 * s2)
        + sig2 * theta * varsig2 * s3
    ) / (sig2 * varsig2)

    xlgg = (
        (2 * YObs * c2 * c3 + c3 ** 2 + c1 ** 2 * (3 * c2 ** 2 - c4) - YObs * c5 + c1 * (-2 * c2 * (YObs * c2 + 2 * c3) + YObs * c4 + c5)) / sig2
        + ((2 * c1 ** 2 * c2 - c2 ** 2 - 2 * c1 * c3 + c4) ** 2 + (VarObs + c1 ** 2 - c2) * (-2 * c2 ** 3 + 2 * c3 ** 2 + c1 ** 2 * (6 * c2 ** 2 - 2 * c4) + 3 * c2 * c4 + 2 * c1 * (-4 * c2 * c3 + c5) - c6)) / varsig2
        + theta * (-(c4 * (-1 + s0)) + c2 ** 2 * (-1 + 2 * s0) - 2 * c2 * s2 + s4)
    )

    det = xlff * xlgg - xlfg ** 2
    return HessianResult(float(xl), float(xlf), float(xlg), float(xlff), float(xlfg), float(xlgg), float(det), m)


def postanalysis(f: float, g: float, data: MeanVarInput) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    m = setup_weights(f, g, data.prior, data.y)
    chi2_mean = (data.y_obs - m.c1) ** 2 / (data.sigma ** 2)
    chi2_variance = (data.var_obs + m.c1 ** 2 - m.c2) ** 2 / (data.sigma_var ** 2)
    chi2 = chi2_mean + chi2_variance
    return float(chi2), float(chi2_mean), float(chi2_variance), float(m.s0), m.w, m.log_w_over_p


def fit(
    data: MeanVarInput,
    ntheta: int = 25,
    pow_factor: float = 1.5,
    niter: int = 100,
    gtol: float = 1e-8,
    stderr: Optional[TextIO] = None,
) -> FitResult:
    """Run the theta ramp-down Newton optimization."""
    f = 0.0
    g = 0.0
    final_h: Optional[HessianResult] = None
    theta_trace: list[tuple[float, float, float, float, float]] = []

    for itheta in range(ntheta, -1, -1):
        theta = data.theta * (pow_factor ** itheta)
        xlf = 1.0e9
        xlg = 1.0e9
        iteration = 1
        h = None
        while iteration <= niter and abs(xlf) + abs(xlg) > gtol:
            h = hessian(f, g, data, theta)
            xlf, xlg = h.xlf, h.xlg
            grad_norm = abs(xlf) + abs(xlg)
            if grad_norm <= gtol:
                break

            if np.isfinite(h.det) and abs(h.det) > np.finfo(float).tiny:
                df = (h.xlg * h.xlfg - h.xlf * h.xlgg) / h.det
                dg = (h.xlf * h.xlfg - h.xlg * h.xlff) / h.det
            else:
                # The standard 2x2 Newton update divides by the Hessian determinant.
                # During dense theta scans, some intermediate continuation steps can
                # produce a singular or nearly singular Hessian even though the target
                # theta fit is otherwise recoverable. In that case, use a small
                # ridge-regularized Newton solve, equivalent to adding lambda*I to the
                # Hessian, only for this ill-conditioned step.
                H = np.array([[h.xlff, h.xlfg], [h.xlfg, h.xlgg]], dtype=np.float64)
                grad = np.array([h.xlf, h.xlg], dtype=np.float64)
                scale = max(float(np.max(np.abs(H))), 1.0)
                step = None
                last_error: Exception | None = None
                for lam in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
                    try:
                        candidate = np.linalg.solve(H + (lam * scale) * np.eye(2), -grad)
                    except np.linalg.LinAlgError as exc:
                        last_error = exc
                        continue
                    if np.all(np.isfinite(candidate)):
                        step = candidate
                        break
                if step is None:
                    raise FloatingPointError(
                        f"Invalid Hessian determinant at theta={theta}: det={h.det}"
                    ) from last_error
                df, dg = float(step[0]), float(step[1])
                if stderr is not None:
                    print(
                        f"warning: regularized singular Hessian at theta={theta:.6g}; det={h.det:.4e}",
                        file=stderr,
                    )

            f += df
            g += dg
            iteration += 1

        h = hessian(f, g, data, theta) if h is None else hessian(f, g, data, theta)
        final_h = h
        final_grad_norm = abs(h.xlf) + abs(h.xlg)
        if final_grad_norm > gtol and iteration > niter:
            warnings.warn(
                (
                    "mean-var optimizer reached --niter before convergence "
                    f"at theta={theta:.6g}; gradient_norm={final_grad_norm:.6g}, "
                    f"gtol={gtol:.6g}"
                ),
                RuntimeWarning,
            )
        chi2, chi2m, chi2v, entropy, _, _ = postanalysis(f, g, data)
        theta_trace.append((theta, entropy, chi2, chi2m, chi2v))

        if stderr is not None:
            print(
                f"iterations:{iteration:8d} {theta:12.4e} {f:12.4e} {g:12.4e} "
                f"{h.xl:12.4e} {h.xlf:12.4e} {h.xlg:12.4e} "
                f"{h.xlff:12.4e} {h.xlfg:12.4e} {h.xlgg:12.4e} {h.det:12.4e}",
                file=stderr,
            )

    if final_h is None:
        raise RuntimeError("Optimizer did not run.")

    chi2, chi2m, chi2v, entropy, weights, log_w_over_p = postanalysis(f, g, data)
    return FitResult(f, g, data.theta, weights, log_w_over_p, chi2, chi2m, chi2v, entropy, final_h.det, theta_trace, data)


def write_output(result: FitResult, out: TextIO) -> None:
    d = result.input_data
    print("# mean/variance ensemble refinement", file=out)
    print("#", file=out)
    print(f"# prior weight factor theta={d.theta:12.5e}", file=out)
    print(f"# mean={d.y_obs:12.4e} +-{d.sigma:12.5e}", file=out)
    print(f"# variance={d.var_obs:12.4e}+-{d.sigma_var:12.5e}", file=out)
    print(f"# number of points n={len(d.y):8d}", file=out)
    if result.det < 0.0:
        print(f"# warning: negative determinant of Hessian{result.det:12.4e}", file=out)
    print("# ---------------------------------", file=out)
    print(f"# chi2={result.chi2:18.10e}", file=out)
    print(f"# chi2(mean)={result.chi2_mean:18.10e}", file=out)
    print(f"# chi2(variance)={result.chi2_variance:18.10e}", file=out)
    print(f"# entropy={result.entropy:18.10e}", file=out)
    print("# ---------------------------------", file=out)
    print("# p_i", file=out)
    print("# -----------------------------------", file=out)
    for wi in result.weights:
        print(f"{wi:12.5e}", file=out)


def write_theta_log(result: FitResult, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for theta, entropy, chi2, chi2m, chi2v in result.theta_trace:
            print(f"{theta:12.4e} {entropy:12.4e} {chi2:12.4e} {chi2m:12.4e} {chi2v:12.4e}", file=out)


def run_one(args: argparse.Namespace) -> None:
    data = read_input(args.input if args.input != "-" else sys.stdin)
    result = fit(data, ntheta=args.ntheta, pow_factor=args.pow, niter=args.niter, gtol=args.gtol, stderr=sys.stderr if args.verbose else None)

    if args.output == "-":
        write_output(result, sys.stdout)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out:
            write_output(result, out)

    if args.theta_log:
        theta_log_path = Path(args.theta_log)
        theta_log_path.parent.mkdir(parents=True, exist_ok=True)
        write_theta_log(result, theta_log_path)


def batch_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            print(f"WARNING: missing input file: {path}", file=sys.stderr)
            continue
        stem = path.name[:-6] if path.name.endswith(".input") else path.stem
        output_path = output_dir / f"output_{stem}.dat"
        log_path = output_dir / f"theta_{stem}.log"
        data = read_input(path)
        result = fit(data, ntheta=args.ntheta, pow_factor=args.pow, niter=args.niter, gtol=args.gtol, stderr=sys.stderr if args.verbose else None)
        with open(output_path, "w", encoding="utf-8") as out:
            write_output(result, out)
        write_theta_log(result, log_path)
        print(f"wrote {output_path}")


def replace_first_line(src: Path, dst: Path, value: float) -> None:
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    if not lines:
        raise ValueError(f"Empty template input: {src}")
    lines[0] = f"{value}\n"
    dst.write_text("".join(lines), encoding="utf-8")


def theta_scan(args: argparse.Namespace) -> None:
    """Scan target theta values by rewriting line 1 of template inputs and fitting each point.

    This is a target-theta scan.  For each scan point, the first line of each
    template input is replaced by the scanned target theta value; each target
    theta is then fitted using the usual internal mean-var ramp controlled by
    --ntheta and --pow.
    """
    templates = [Path(p) for p in args.templates]
    missing = [str(p) for p in templates if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing template input(s): " + ", ".join(missing))

    if len(templates) != len(args.start) or len(templates) != len(args.stop):
        raise ValueError("--templates, --start, and --stop must have the same length.")
    num_theta = int(getattr(args, "num_theta", getattr(args, "n", 100)))
    if num_theta <= 0:
        raise ValueError("--num-theta/-n must be positive.")

    if getattr(args, "spacing", "linear") == "log":
        if any(v <= 0 for v in args.start) or any(v <= 0 for v in args.stop):
            raise ValueError("Log-spaced theta scans require positive --start and --stop values.")
        ranges = [np.geomspace(a, b, num_theta) for a, b in zip(args.start, args.stop)]
    else:
        ranges = [np.linspace(a, b, num_theta) for a, b in zip(args.start, args.stop)]

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    summary_path = root / "theta_scan_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as summary:
        summary.write("scan_index\tfolder\ttemplate\ttarget_theta\toutput_file\ttheta_log\n")
        for i in range(num_theta):
            folder = root / f"theta_{i:06d}"
            folder.mkdir(parents=True, exist_ok=True)
            scan_inputs = []
            rows = []
            for template, values in zip(templates, ranges):
                dst = folder / template.name
                target_theta = float(values[i])
                replace_first_line(template, dst, target_theta)
                scan_inputs.append(str(dst))
                stem = dst.name[:-6] if dst.name.endswith(".input") else dst.stem
                rows.append((template, target_theta, folder / f"output_{stem}.dat", folder / f"theta_{stem}.log"))

            ns = argparse.Namespace(
                inputs=scan_inputs,
                output_dir=str(folder),
                ntheta=args.ntheta,
                pow=args.pow,
                niter=args.niter,
                gtol=args.gtol,
                verbose=args.verbose,
            )
            batch_run(ns)
            for template, target_theta, output_file, theta_log in rows:
                summary.write(
                    f"{i}\t{folder}\t{template}\t{target_theta:.16g}\t{output_file}\t{theta_log}\n"
                )
    print(f"wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mean/variance Ensemble refinement.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ntheta", type=int, default=100, help="number of theta ramp-down steps; default 100")
    common.add_argument("--pow", type=float, default=1.189207115002721, help="theta geometric scale factor; default 1.189207115002721")
    common.add_argument("--niter", type=int, default=100, help="maximum Newton iterations per theta")
    common.add_argument("--gtol", type=float, default=1e-8, help="gradient tolerance")
    common.add_argument("--verbose", action="store_true", help="print iteration diagnostics to stderr")

    p_run = sub.add_parser("run", parents=[common], help="run one input file")
    p_run.add_argument("input", help="input file path, or '-' for stdin")
    p_run.add_argument("-o", "--output", default="-", help="output file path, or '-' for stdout")
    p_run.add_argument("--theta-log", help="optional theta trace log path")
    p_run.set_defaults(func=run_one)

    p_batch = sub.add_parser("batch", parents=[common], help="run multiple mean/variance input files")
    p_batch.add_argument("inputs", nargs="+", help="input files, e.g. closed.input open.input")
    p_batch.add_argument("--output-dir", default=".", help="directory for output_*.dat and theta_*.log")
    p_batch.set_defaults(func=batch_run)

    p_scan = sub.add_parser("theta-scan", parents=[common], help="scan target theta values")
    p_scan.add_argument("--templates", nargs="+", default=["closed.input", "open.input"], help="template input files")
    p_scan.add_argument("--start", nargs="+", type=float, default=[0.01, 0.1], help="first theta value for each template")
    p_scan.add_argument("--stop", nargs="+", type=float, default=[10.0, 10.0], help="last theta value for each template")
    p_scan.add_argument("-n", "--num-theta", dest="n", type=int, default=20000, help="number of target theta points")
    p_scan.add_argument("--spacing", choices=["linear", "log"], default="linear", help="spacing for target theta values")
    p_scan.add_argument("--output-root", default="mean_var_theta_scan", help="root directory for theta_000000 folders")
    p_scan.set_defaults(func=theta_scan)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
