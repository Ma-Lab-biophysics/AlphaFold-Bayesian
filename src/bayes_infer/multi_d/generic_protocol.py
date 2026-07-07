"""multi-dimensional inference protocol

Data preparation follows the NMR/PRE data-preparation script in ``nmr_prep``.  The optimization step uses the local default log-weights backend.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Sequence

from . import nmr_prep as prep
from .backends.bioen_log_weights import LocalBioEnLogWeightsResult, run_local_generic

DEFAULT_THETA = prep.DEFAULT_THETA
DEFAULT_MULTI_D_INPUT_DIR = prep.DEFAULT_MULTI_D_INPUT_DIR


def read_data_ids_from_exp(data_dir: Path) -> list[str]:
    exp_file = data_dir / "exp-generic.dat"
    if not exp_file.exists():
        raise FileNotFoundError(f"Missing {exp_file}")
    ids: list[str] = []
    with exp_file.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split()[0])
    return ids


def count_models(data_dir: Path) -> int:
    models = data_dir / "models-generic.dat"
    if not models.exists():
        raise FileNotFoundError(f"Missing {models}")
    with models.open() as fh:
        return sum(1 for line in fh if line.strip() and not line.lstrip().startswith("#"))


def generate_theta_values(start: float, stop: float, n: int, spacing: str = "log") -> list[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return [float(start)]
    if spacing == "linear":
        return [float(start + (stop - start) * i / (n - 1)) for i in range(n)]
    if spacing == "log":
        if start <= 0 or stop <= 0:
            raise ValueError("log spacing requires positive start/stop")
        a, b = math.log10(start), math.log10(stop)
        return [float(10 ** (a + (b - a) * i / (n - 1))) for i in range(n)]
    raise ValueError(f"Unknown theta spacing: {spacing}")


def make_data(raw_xls: Path, r1rho_xls: Path, structures_dir: Path, out_dir: Path, **kwargs) -> Path:
    """Make generic multi-d input files using the NMR/PRE preparation code."""
    class Args:
        pass

    args = Args()
    args.raw_xls = str(raw_xls)
    args.r1rho_xls = str(r1rho_xls)
    args.structures_dir = str(structures_dir)
    args.out_dir = str(out_dir)
    args.skip_column = list(kwargs.get("skip_column", (8,)))
    args.finite_lower = float(kwargs.get("finite_lower", prep.FINITE_LOWER_DEFAULT))
    args.finite_upper = float(kwargs.get("finite_upper", prep.FINITE_UPPER_DEFAULT))
    args.default_noise = float(kwargs.get("default_noise", prep.DEFAULT_NOISE_DEFAULT))
    args.broadened_distance = float(kwargs.get("broadened_distance", prep.BROADENED_DISTANCE_DEFAULT))
    args.broadened_noise = float(kwargs.get("broadened_noise", prep.BROADENED_NOISE_DEFAULT))
    args.long_distance = float(kwargs.get("long_distance", prep.LONG_DISTANCE_DEFAULT))
    args.long_distance_noise = float(kwargs.get("long_distance_noise", prep.LONG_DISTANCE_NOISE_DEFAULT))
    args.expected_last_residue = int(kwargs.get("expected_last_residue", prep.EXPECTED_LAST_RESIDUE_DEFAULT))
    args.atom_mode = kwargs.get("atom_mode", "nmr-index")
    args.keep_incomplete_models = bool(kwargs.get("keep_incomplete_models", False))
    args.default_theta = float(kwargs.get("theta", DEFAULT_THETA))
    args.bioen_exe = kwargs.get("bioen_exe", "bioen")
    prep.cmd_make_data(args)
    return out_dir / DEFAULT_MULTI_D_INPUT_DIR


def run_one(
    data_dir: Path,
    theta: float,
    out_dir: Path,
    data_ids: Sequence[str] | None = None,
) -> LocalBioEnLogWeightsResult:
    """Run one theta using the default local log-weight backend."""
    data_dir = data_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = tuple(data_ids or read_data_ids_from_exp(data_dir))
    (out_dir / "backend.txt").write_text("local-log-weights\n")
    result = run_local_generic(data_dir, theta, out_dir, ids)
    if not result.converged:
        print(
            "WARNING: multi-d local optimizer did not report convergence; "
            f"warnflag={result.warnflag}. See {out_dir / 'summary.json'}.",
            file=sys.stderr,
        )
    return result


def theta_scan(
    data_dir: Path,
    out_dir: Path,
    theta_values: Sequence[float],
    run: bool = False,
) -> None:
    """Create a theta scan and optionally run the default local log-weight backend."""
    data_dir = data_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = tuple(read_data_ids_from_exp(data_dir))
    with (out_dir / "theta_scan_commands.tsv").open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["theta_index", "theta", "theta_dir", "backend", "command"])
        for i, theta in enumerate(theta_values):
            theta_dir = out_dir / f"theta_{i:06d}"
            theta_dir.mkdir(parents=True, exist_ok=True)
            theta_file = theta_dir / "thetas.dat"
            theta_file.write_text(f"{theta:.12g}\n")
            command = (
                "bayes-infer multi-d run "
                f"--data-dir {data_dir} --theta {theta:.12g} --out-dir {theta_dir}"
            )
            (theta_dir / "run_bayes_infer.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
            writer.writerow([i, theta, theta_dir, "local-log-weights", command])
            if run:
                result = run_local_generic(data_dir, theta, theta_dir, ids)
                if not result.converged:
                    print(
                        "WARNING: multi-d local optimizer did not report convergence "
                        f"for theta={theta:.12g}; warnflag={result.warnflag}. "
                        f"See {theta_dir / 'summary.json'}.",
                        file=sys.stderr,
                    )
