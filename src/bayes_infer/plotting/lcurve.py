"""L-curve plotting utilities for bayes-infer theta scans."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LCurvePoint:
    label: str
    theta: float
    entropy_skl: float
    chi2_total: float
    chi2_mean: float | None = None
    chi2_variance: float | None = None
    source: str = ""


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - message depends on environment
        raise RuntimeError(
            "matplotlib is required for L-curve plotting. Install with `pip install matplotlib`."
        ) from exc
    return plt


def read_mean_var_theta_log(path: Path | str, label: str | None = None) -> list[LCurvePoint]:
    """Read a mean-var theta log.

    Expected columns:
        theta_internal, S_KL, chi2_total, chi2_mean, chi2_variance
    """
    path = Path(path)
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 5:
        raise ValueError(f"{path} must have at least 5 columns: theta S_KL chi2 chi2_mean chi2_variance")
    name = label or path.stem
    return [
        LCurvePoint(
            label=name,
            theta=float(row[0]),
            entropy_skl=float(row[1]),
            chi2_total=float(row[2]),
            chi2_mean=float(row[3]),
            chi2_variance=float(row[4]),
            source=str(path),
        )
        for row in data
    ]


def _mean_var_logs_from_scan_root(scan_root: Path, stems: Sequence[str] | None = None) -> dict[str, list[LCurvePoint]]:
    """Collect final points from theta_*/theta_<stem>.log files."""
    scan_root = Path(scan_root)
    if not scan_root.exists():
        raise FileNotFoundError(scan_root)
    theta_dirs = sorted(p for p in scan_root.glob("theta_*" ) if p.is_dir())
    if not theta_dirs:
        raise FileNotFoundError(f"No theta_* directories found in {scan_root}")

    if stems:
        wanted = [s[:-4] if s.endswith(".log") else s for s in stems]
        # Normalize likely user inputs: closed, theta_closed, closed.input.
        normalized = []
        for s in wanted:
            if s.endswith(".input"):
                s = s[:-6]
            if s.startswith("theta_"):
                s = s[6:]
            normalized.append(s)
        log_names = [f"theta_{s}.log" for s in normalized]
    else:
        discovered = sorted({p.name for d in theta_dirs for p in d.glob("theta_*.log")})
        if not discovered:
            raise FileNotFoundError(f"No theta_*.log files found below {scan_root}")
        log_names = discovered

    series: dict[str, list[LCurvePoint]] = {}
    for log_name in log_names:
        label = log_name
        if label.startswith("theta_"):
            label = label[6:]
        if label.endswith(".log"):
            label = label[:-4]
        points: list[LCurvePoint] = []
        for d in theta_dirs:
            log_path = d / log_name
            if not log_path.exists():
                continue
            rows = read_mean_var_theta_log(log_path, label=label)
            if rows:
                points.append(rows[-1])
        if points:
            series[label] = points
    if not series:
        raise FileNotFoundError("No matching mean-var theta logs found.")
    return series


def mean_var_lcurve_from_logs(logs: Sequence[Path | str], labels: Sequence[str] | None = None) -> dict[str, list[LCurvePoint]]:
    if labels is not None and len(labels) != len(logs):
        raise ValueError("--labels must have the same length as --theta-logs")
    series: dict[str, list[LCurvePoint]] = {}
    for i, log in enumerate(logs):
        label = labels[i] if labels else Path(log).stem
        series[label] = read_mean_var_theta_log(log, label=label)
    return series


def read_multi_d_theta_scan(scan_dir: Path | str, chi2: str = "full") -> dict[str, list[LCurvePoint]]:
    """Read multi-d theta scan results from theta_*/summary.json files."""
    scan_dir = Path(scan_dir)
    if not scan_dir.exists():
        raise FileNotFoundError(scan_dir)
    points: list[LCurvePoint] = []
    for theta_dir in sorted(p for p in scan_dir.glob("theta_*" ) if p.is_dir()):
        summary_file = theta_dir / "summary.json"
        if not summary_file.exists():
            continue
        summary = json.loads(summary_file.read_text())
        theta = float(summary.get("theta"))
        half = float(summary.get("chi2_half_sum"))
        full = float(summary.get("chi2_full_sum", 2.0 * half))
        entropy = float(summary.get("entropy_skl"))
        points.append(
            LCurvePoint(
                label="multi-d",
                theta=theta,
                entropy_skl=entropy,
                chi2_total=full if chi2 == "full" else half,
                source=str(summary_file),
            )
        )
    if not points:
        raise FileNotFoundError(f"No theta_*/summary.json files found in {scan_dir}. Run theta-scan with --run first.")
    points.sort(key=lambda p: p.theta)
    return {"multi-d": points}


def write_lcurve_table(series: dict[str, Sequence[LCurvePoint]], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["series", "theta", "S_KL", "chi2", "chi2_mean", "chi2_variance", "source"])
        for label, points in series.items():
            for p in points:
                writer.writerow([
                    label,
                    f"{p.theta:.16g}",
                    f"{p.entropy_skl:.16g}",
                    f"{p.chi2_total:.16g}",
                    "" if p.chi2_mean is None else f"{p.chi2_mean:.16g}",
                    "" if p.chi2_variance is None else f"{p.chi2_variance:.16g}",
                    p.source,
                ])


def plot_lcurve(
    series: dict[str, Sequence[LCurvePoint]],
    out_path: Path | str,
    *,
    title: str = "L-curve",
    chi2_column: str = "total",
    xscale: str = "linear",
    yscale: str = "linear",
    annotate: bool = False,
    max_annotations: int = 50,
) -> None:
    """Plot S_KL versus chi2 for one or more L-curve series."""
    plt = _require_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def select_chi2(p: LCurvePoint) -> float:
        if chi2_column == "mean":
            if p.chi2_mean is None:
                raise ValueError("chi2_mean is unavailable for this L-curve series")
            return p.chi2_mean
        if chi2_column == "variance":
            if p.chi2_variance is None:
                raise ValueError("chi2_variance is unavailable for this L-curve series")
            return p.chi2_variance
        return p.chi2_total

    plt.figure(figsize=(6.5, 4.5))
    for label, points in series.items():
        xs = [p.entropy_skl for p in points]
        ys = [select_chi2(p) for p in points]
        plt.plot(xs, ys, marker="o", linestyle="-", label=label)
        if annotate:
            step = max(1, int(np.ceil(len(points) / max_annotations)))
            for p in list(points)[::step]:
                plt.annotate(f"{p.theta:.2g}", (p.entropy_skl, select_chi2(p)), fontsize=7)

    plt.xlabel("Relative entropy (S_KL)", fontsize=12)
    ylabel = "Chi-squared"
    if chi2_column == "mean":
        ylabel = "Chi-squared, mean term"
    elif chi2_column == "variance":
        ylabel = "Chi-squared, variance term"
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



def plot_chi2_vs_theta(
    series: dict[str, Sequence[LCurvePoint]],
    out_path: Path | str,
    *,
    title: str = "Chi-squared versus theta",
    chi2_column: str = "total",
    xscale: str = "linear",
    yscale: str = "linear",
    annotate: bool = False,
    max_annotations: int = 50,
    target_chi2: float | None = None,
    chi2_cutoff: float | None = None,
) -> list[tuple[str, LCurvePoint, float]]:
    """Plot chi2 versus theta for one or more theta-scan series.

    Returns closest points to an optional target chi-squared value and
    highlights points below an optional cutoff.
    """
    plt = _require_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def select_chi2(p: LCurvePoint) -> float:
        if chi2_column == "mean":
            if p.chi2_mean is None:
                raise ValueError("chi2_mean is unavailable for this theta series")
            return p.chi2_mean
        if chi2_column == "variance":
            if p.chi2_variance is None:
                raise ValueError("chi2_variance is unavailable for this theta series")
            return p.chi2_variance
        return p.chi2_total

    plt.figure(figsize=(6.5, 4.5))
    closest: list[tuple[str, LCurvePoint, float]] = []
    for label, points_in in series.items():
        points = sorted([p for p in points_in if np.isfinite(p.theta) and np.isfinite(select_chi2(p))], key=lambda p: p.theta)
        if not points:
            continue
        xs = [p.theta for p in points]
        ys = [select_chi2(p) for p in points]
        plt.plot(xs, ys, marker="o", linestyle="-", label=label)
        if annotate:
            step = max(1, int(np.ceil(len(points) / max_annotations)))
            for p in points[::step]:
                plt.annotate(f"{p.theta:.2g}", (p.theta, select_chi2(p)), fontsize=7)
        if chi2_cutoff is not None:
            below = [p for p in points if select_chi2(p) < chi2_cutoff]
            if below:
                # Add a light marker overlay for the accepted region without changing colors manually.
                plt.plot([p.theta for p in below], [select_chi2(p) for p in below], linestyle="", marker="x", label=f"{label} < {chi2_cutoff:g}")
        if target_chi2 is not None:
            best = min(points, key=lambda p: abs(select_chi2(p) - target_chi2))
            closest.append((label, best, abs(select_chi2(best) - target_chi2)))

    plt.xlabel("Theta", fontsize=12)
    ylabel = "Chi-squared"
    if chi2_column == "mean":
        ylabel = "Chi-squared, mean term"
    elif chi2_column == "variance":
        ylabel = "Chi-squared, variance term"
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return closest


def mean_var_plot_chi2_theta(args) -> None:
    if getattr(args, "scan_root", None):
        series = _mean_var_logs_from_scan_root(Path(args.scan_root), stems=args.stems)
    elif getattr(args, "theta_logs", None):
        series = mean_var_lcurve_from_logs(args.theta_logs, labels=args.labels)
    else:
        raise ValueError("Provide either --scan-root or --theta-logs")
    if args.table:
        write_lcurve_table(series, args.table)
    closest = plot_chi2_vs_theta(
        series,
        args.output,
        title=args.title or "mean-var chi-squared versus theta",
        chi2_column=args.chi2,
        xscale=args.xscale,
        yscale=args.yscale,
        annotate=args.annotate,
        max_annotations=args.max_annotations,
        target_chi2=args.target_chi2,
        chi2_cutoff=args.chi2_cutoff,
    )
    print(f"wrote {args.output}")
    if args.table:
        print(f"wrote {args.table}")
    for label, point, delta in closest:
        if args.chi2 == "mean":
            selected_chi2 = point.chi2_mean
        elif args.chi2 == "variance":
            selected_chi2 = point.chi2_variance
        else:
            selected_chi2 = point.chi2_total
        if selected_chi2 is None:
            raise ValueError(f"chi2_{args.chi2} is unavailable for {label}")
        print(
            f"[{label}] closest {args.chi2} chi2 to {args.target_chi2:g}: "
            f"theta={point.theta:.12g}, chi2={selected_chi2:.12g}, abs_delta={delta:.3g}"
        )


def multi_d_plot_chi2_theta(args) -> None:
    series = read_multi_d_theta_scan(args.scan_dir, chi2=args.chi2)
    if args.table:
        write_lcurve_table(series, args.table)
    closest = plot_chi2_vs_theta(
        series,
        args.output,
        title=args.title or "multi-d chi-squared versus theta",
        chi2_column="total",
        xscale=args.xscale,
        yscale=args.yscale,
        annotate=args.annotate,
        max_annotations=args.max_annotations,
        target_chi2=args.target_chi2,
        chi2_cutoff=args.chi2_cutoff,
    )
    print(f"wrote {args.output}")
    if args.table:
        print(f"wrote {args.table}")
    for label, point, delta in closest:
        print(
            f"[{label}] closest chi2 to {args.target_chi2:g}: "
            f"theta={point.theta:.12g}, chi2={point.chi2_total:.12g}, abs_delta={delta:.3g}"
        )


def mean_var_plot_lcurve(args) -> None:
    if getattr(args, "scan_root", None):
        series = _mean_var_logs_from_scan_root(Path(args.scan_root), stems=args.stems)
    elif getattr(args, "theta_logs", None):
        series = mean_var_lcurve_from_logs(args.theta_logs, labels=args.labels)
    else:
        raise ValueError("Provide either --scan-root or --theta-logs")
    if args.table:
        write_lcurve_table(series, args.table)
    plot_lcurve(
        series,
        args.output,
        title=args.title or "mean-var L-curve",
        chi2_column=args.chi2,
        xscale=args.xscale,
        yscale=args.yscale,
        annotate=args.annotate,
        max_annotations=args.max_annotations,
    )
    print(f"wrote {args.output}")
    if args.table:
        print(f"wrote {args.table}")


def multi_d_plot_lcurve(args) -> None:
    series = read_multi_d_theta_scan(args.scan_dir, chi2=args.chi2)
    if args.table:
        write_lcurve_table(series, args.table)
    plot_lcurve(
        series,
        args.output,
        title=args.title or "multi-d L-curve",
        chi2_column="total",
        xscale=args.xscale,
        yscale=args.yscale,
        annotate=args.annotate,
        max_annotations=args.max_annotations,
    )
    print(f"wrote {args.output}")
    if args.table:
        print(f"wrote {args.table}")
