#!/usr/bin/env python3
"""
Use NMR-PRE data to prepare multi-dimensional observables.

This module converts NMR-PRE intensity ratios to distance restraints, writes
input files for the local multi-d backend, and writes optional official BioEn
run scripts for users who want to run BioEn separately.

The data-making workflow:
  1. parses probe residues from column names containing "AtC";
  2. parses query residues from row labels containing "N-H" or "?N-H";
  3. skips configured spreadsheet columns;
  4. converts finite intensity ratios with the PRE equations;
  5. assigns configured fallback distances for broadened and long-distance
     restraints; and
  6. simulates distances from PDB structures with either NMR/PRE atom-index
     offsets or atom-name lookup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from scipy.optimize import brentq
except ImportError:  # pragma: no cover
    brentq = None


# ---------------------------------------------------------------------------
# Constants for NMR/PRE restraint construction.
# ---------------------------------------------------------------------------
T_INEPT_DEFAULT = 9.0e-3
OMEGA_H_DEFAULT = 2.0 * 3.14 * 9.0e8
TAU_C_DEFAULT = 4.0e-9
K_CONST_DEFAULT = 1.23e-32
FINITE_LOWER_DEFAULT = 1.0e-5
FINITE_UPPER_DEFAULT = 0.95
BROADENED_DISTANCE_DEFAULT = 7.0
BROADENED_NOISE_DEFAULT = 5.0
LONG_DISTANCE_DEFAULT = 50.0
LONG_DISTANCE_NOISE_DEFAULT = 25.0
DEFAULT_NOISE_DEFAULT = 5.0
EXPECTED_LAST_RESIDUE_DEFAULT = 441
DEFAULT_THETA = 3000.0
DEFAULT_MULTI_D_INPUT_DIR = "multi_d_input"
LEGACY_MULTI_D_INPUT_DIR = "BioEN_Files"


@dataclass
class Restraint:
    data_id: str
    probe: int
    query: int
    measurement: float
    noise: float
    source_ratio: Optional[float]
    kind: str


@dataclass
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float


@dataclass
class ModelRecord:
    model_index: int
    pdb_file: str
    pdb_stem: str


# ---------------------------------------------------------------------------
# Small utilities.
# ---------------------------------------------------------------------------
def require_package_dependencies() -> None:
    missing = []
    if pd is None:
        missing.append("pandas/openpyxl")
    if brentq is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "Missing required package(s): " + ", ".join(missing) +
            ". Install them in the environment that runs this script."
        )


def parse_probe_from_column(column_name: str) -> Optional[int]:
    """Parse probe residue from column labels containing AtC<number>."""
    text = str(column_name)
    match = re.search(r"AtC(\d+)", text)
    if match:
        return int(match.group(1))
    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else None


def parse_query_from_row_label(label: object) -> Optional[int]:
    """Parse query residue from labels like 'A123N-H' or '?N-H'."""
    if label is None:
        return None
    text = str(label)
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"(\d+)\s*\??N-H", text)
    if match:
        return int(match.group(1))
    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else None


def read_r2_from_r1rho_table(r1rho_df, residue: int) -> float:
    """Read the R2/R1rho value for a residue, with a label-based fallback."""
    # Primary layout: residue rows in order, R2/R1rho values in the second column.
    if 1 <= residue <= len(r1rho_df):
        value = r1rho_df.iloc[residue - 1, 1]
        if pd.notna(value):
            return float(value)

    # Fallback if first column explicitly stores residue IDs.
    for row_i in range(len(r1rho_df)):
        label = str(r1rho_df.iloc[row_i, 0])
        nums = re.findall(r"\d+", label)
        if nums and int(nums[0]) == residue:
            value = r1rho_df.iloc[row_i, 1]
            if pd.notna(value):
                return float(value)

    raise ValueError(f"Could not find R2/R1rho value for residue {residue}")


def pre_ratio_to_distance_angstrom(
    i_ratio: float,
    r2: float,
    t_inept: float = T_INEPT_DEFAULT,
    omega_h: float = OMEGA_H_DEFAULT,
    tau_c: float = TAU_C_DEFAULT,
    k_const: float = K_CONST_DEFAULT,
) -> Optional[float]:
    """Convert Iox/Ired to an effective PRE distance in Angstrom.

    Eq. 1:
        Iox/Ired = R2 * exp(-R2sp*t) / (R2 + R2sp)
    Eq. 2:
        r = [K/R2sp * (4*tauc + 3*tauc/(1 + omega_h^2*tauc^2))]^(1/6)
    """
    if not (0.0 < i_ratio < 1.0) or r2 <= 0.0:
        return None

    def equation(r2sp: float) -> float:
        return r2 * math.exp(-r2sp * t_inept) / (r2 + r2sp) - i_ratio

    lo = 1e-12
    hi = 1.0
    while equation(hi) > 0.0 and hi < 1e12:
        hi *= 10.0
    if hi >= 1e12 and equation(hi) > 0.0:
        return None

    r2sp = brentq(equation, lo, hi, xtol=1e-12, maxiter=300)
    factor = 4.0 * tau_c + 3.0 * tau_c / (1.0 + omega_h * omega_h * tau_c * tau_c)
    r_cm = ((k_const / r2sp) * factor) ** (1.0 / 6.0)
    return r_cm * 1.0e8


def is_number(value: object) -> bool:
    try:
        float(value)
        return not math.isnan(float(value))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NMR/PRE multi-d restraint construction.
# ---------------------------------------------------------------------------
def build_restraints_nmr(
    raw_xls: Path,
    r1rho_xls: Path,
    skip_column_indices_1based: Sequence[int] = (8,),
    finite_lower: float = FINITE_LOWER_DEFAULT,
    finite_upper: float = FINITE_UPPER_DEFAULT,
    default_noise: float = DEFAULT_NOISE_DEFAULT,
    broadened_distance: float = BROADENED_DISTANCE_DEFAULT,
    broadened_noise: float = BROADENED_NOISE_DEFAULT,
    long_distance: float = LONG_DISTANCE_DEFAULT,
    long_distance_noise: float = LONG_DISTANCE_NOISE_DEFAULT,
) -> List[Restraint]:
    require_package_dependencies()

    raw_df = pd.read_excel(raw_xls)
    r1rho_df = pd.read_excel(r1rho_xls, header=None)

    restraints: List[Restraint] = []

    # The first column contains row labels, so data columns start at index 1.
    for col_0based in range(1, raw_df.shape[1]):
        col_1based = col_0based + 1
        if col_1based in set(skip_column_indices_1based):
            continue

        probe_column = raw_df.columns[col_0based]
        probe_residue = parse_probe_from_column(str(probe_column))
        if probe_residue is None:
            continue

        for row_i in range(raw_df.shape[0]):
            row_label = raw_df.iloc[row_i, 0]
            query_residue = parse_query_from_row_label(row_label)
            if query_residue is None:
                continue

            value = raw_df.iloc[row_i, col_0based]
            if not is_number(value):
                continue
            intensity_ratio = float(value)

            if probe_residue == query_residue:
                # Self-restraints are physically meaningless for this distance model.
                continue

            data_id = f"{probe_residue}_{query_residue}"

            if finite_lower < intensity_ratio < finite_upper:
                R2 = read_r2_from_r1rho_table(r1rho_df, query_residue)
                distance = pre_ratio_to_distance_angstrom(intensity_ratio, R2)
                if distance is None or not math.isfinite(distance) or distance == 0.0:
                    continue
                restraints.append(
                    Restraint(
                        data_id=data_id,
                        probe=probe_residue,
                        query=query_residue,
                        measurement=float(distance),
                        noise=default_noise,
                        source_ratio=intensity_ratio,
                        kind="finite_ratio",
                    )
                )
            elif intensity_ratio == 0.0:
                restraints.append(
                    Restraint(
                        data_id=data_id,
                        probe=probe_residue,
                        query=query_residue,
                        measurement=broadened_distance,
                        noise=broadened_noise,
                        source_ratio=intensity_ratio,
                        kind="strongly_broadened",
                    )
                )
            elif intensity_ratio > finite_upper:
                restraints.append(
                    Restraint(
                        data_id=data_id,
                        probe=probe_residue,
                        query=query_residue,
                        measurement=long_distance,
                        noise=long_distance_noise,
                        source_ratio=intensity_ratio,
                        kind="beyond_detection",
                    )
                )

    return restraints


# ---------------------------------------------------------------------------
# PDB reading and simulated distance generation.
# ---------------------------------------------------------------------------
def read_pdb_atoms_first_chain(pdb_path: Path) -> List[Atom]:
    atoms: List[Atom] = []
    first_chain: Optional[str] = None
    with pdb_path.open("r", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip() or " "
            if first_chain is None:
                first_chain = chain
            if chain != first_chain:
                break
            try:
                atom = Atom(
                    serial=int(line[6:11]),
                    name=line[12:16].strip(),
                    resname=line[17:20].strip(),
                    chain=chain,
                    resseq=int(line[22:26]),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                )
                atoms.append(atom)
            except Exception:
                continue
    return atoms


def residue_anchor_indices(atoms: Sequence[Atom]) -> Tuple[List[int], int]:
    """Return residue anchors used by the NMR/PRE atom-index distance mode.

    Residue 1 is anchored at its first atom. Later residues are anchored at the
    last atom of the previous residue. Missing residues receive -1.
    """
    if not atoms:
        return [], -1

    max_resseq = max(a.resseq for a in atoms)
    anchors = [-1] * (max_resseq + 1)

    first_resseq = atoms[0].resseq
    if 0 <= first_resseq <= max_resseq:
        anchors[first_resseq] = 0

    for idx in range(len(atoms) - 1):
        current_resseq = atoms[idx].resseq
        next_resseq = atoms[idx + 1].resseq
        if (
            current_resseq != next_resseq
            and 0 <= next_resseq <= max_resseq
            and anchors[next_resseq] == -1
        ):
            anchors[next_resseq] = idx

    return anchors, max_resseq


def distance(a: Atom, b: Atom) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def nmr_index_distance(atoms: Sequence[Atom], residue_anchor_indices: Sequence[int], probe: int, query: int) -> Optional[float]:
    """Distance using NMR/PRE atom-index offsets from residue anchors."""
    if probe >= len(residue_anchor_indices) or query >= len(residue_anchor_indices):
        return None
    probe_anchor = residue_anchor_indices[probe]
    query_anchor = residue_anchor_indices[query]
    if probe_anchor < 0 or query_anchor < 0:
        return None
    probe_idx = probe_anchor + 6
    query_idx = query_anchor + 2
    if probe_idx >= len(atoms) or query_idx >= len(atoms):
        return None
    return distance(atoms[query_idx], atoms[probe_idx])


def find_atom_by_resseq_and_names(atoms: Sequence[Atom], resseq: int, names: Sequence[str]) -> Optional[Atom]:
    wanted = {n.upper() for n in names}
    for atom in atoms:
        if atom.resseq == resseq and atom.name.upper() in wanted:
            return atom
    return None


def atom_name_distance(atoms: Sequence[Atom], probe: int, query: int) -> Optional[float]:
    """Safer optional mode: MTSL approximated by CB; amide by H/HN/N fallback."""
    probe_atom = find_atom_by_resseq_and_names(atoms, probe, ["CB", "CA"])
    query_atom = find_atom_by_resseq_and_names(atoms, query, ["H", "HN", "1H", "N"])
    if probe_atom is None or query_atom is None:
        return None
    return distance(query_atom, probe_atom)


def build_simulated_distances(
    structures_dir: Path,
    restraints: Sequence[Restraint],
    expected_last_residue: int = EXPECTED_LAST_RESIDUE_DEFAULT,
    atom_mode: str = "nmr-index",
    drop_incomplete_models: bool = True,
) -> Tuple[List[ModelRecord], Dict[str, List[float]], List[str]]:
    """Build sim-<ID>-generic.dat content from PDB structures.

    Returns:
      model_records: accepted model list
      sim_by_id: data_id -> list of simulated values, one per accepted model
      warnings: warning strings
    """
    pdb_files = sorted(structures_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {structures_dir}")

    data_ids = [r.data_id for r in restraints]
    sim_by_id: Dict[str, List[float]] = {data_id: [] for data_id in data_ids}
    model_records: List[ModelRecord] = []
    warnings: List[str] = []

    for pdb_file in pdb_files:
        atoms = read_pdb_atoms_first_chain(pdb_file)
        if not atoms:
            warnings.append(f"SKIP {pdb_file.name}: no ATOM records in first chain")
            continue

        if atoms[-1].resseq != expected_last_residue:
            warnings.append(
                f"SKIP {pdb_file.name}: last residue in first chain is {atoms[-1].resseq}, "
                f"expected {expected_last_residue}"
            )
            continue

        anchor_indices, _ = residue_anchor_indices(atoms)
        model_values: Dict[str, float] = {}
        incomplete = False

        for restraint in restraints:
            if atom_mode == "nmr-index":
                d = nmr_index_distance(atoms, anchor_indices, restraint.probe, restraint.query)
            elif atom_mode == "atom-name":
                d = atom_name_distance(atoms, restraint.probe, restraint.query)
            else:
                raise ValueError(f"Unknown atom mode: {atom_mode}")

            if d is None or not math.isfinite(d):
                incomplete = True
                model_values[restraint.data_id] = float("nan")
            else:
                model_values[restraint.data_id] = d

        if incomplete and drop_incomplete_models:
            warnings.append(f"SKIP {pdb_file.name}: missing atoms for at least one restraint")
            continue

        model_index = len(model_records)
        model_records.append(ModelRecord(model_index, pdb_file.name, pdb_file.stem))
        for data_id in data_ids:
            sim_by_id[data_id].append(model_values[data_id])

    if not model_records:
        raise RuntimeError("No PDB models survived filtering; check structure numbering/atom mode.")

    return model_records, sim_by_id, warnings


# ---------------------------------------------------------------------------
# Multi-dimensional file writing and command writing.
# ---------------------------------------------------------------------------
def write_bioen_generic_files(
    out_dir: Path,
    restraints: Sequence[Restraint],
    model_records: Sequence[ModelRecord],
    sim_by_id: Dict[str, Sequence[float]],
) -> None:
    input_dir = out_dir / DEFAULT_MULTI_D_INPUT_DIR
    input_dir.mkdir(parents=True, exist_ok=True)

    # exp-generic.dat uses the BioEn-compatible generic observable format.
    with (input_dir / "exp-generic.dat").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["#ID", "measurement", "noise"])
        for r in restraints:
            writer.writerow([r.data_id, f"{r.measurement:.10g}", f"{r.noise:.10g}"])

    # One simulated-data file for each restraint/data ID.
    for r in restraints:
        sim_file = input_dir / f"sim-{r.data_id}-generic.dat"
        with sim_file.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["#Simulated Measurements Sorted by Rank"])
            for value in sim_by_id[r.data_id]:
                writer.writerow([f"{float(value):.10g}"])

    # models-generic.dat uses zero-based model indices.
    with (input_dir / "models-generic.dat").open("w") as handle:
        for record in model_records:
            handle.write(f"{record.model_index}\n")

    # DataOrder.txt keeps a compact model-index to PDB-stem map.
    with (out_dir / "DataOrder.txt").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for record in model_records:
            writer.writerow([record.model_index, record.pdb_stem])
    with (out_dir / "DataOrder.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["model_index", "pdb_file", "pdb_stem"])
        for record in model_records:
            writer.writerow([record.model_index, record.pdb_file, record.pdb_stem])

    # Extra metadata
    with (out_dir / "restraints.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["data_id", "probe", "query", "measurement", "noise", "source_ratio", "kind"])
        for r in restraints:
            writer.writerow([r.data_id, r.probe, r.query, r.measurement, r.noise, r.source_ratio, r.kind])

    with (out_dir / "DataIds.txt").open("w") as handle:
        handle.write(",".join(r.data_id for r in restraints) + "\n")


def make_bioen_command(
    data_dir: Path,
    theta_file: Path,
    output_pkl: Path,
    data_ids: Sequence[str],
    number_of_models: int,
    bioen_exe: str = "bioen",
) -> List[str]:
    return [
        bioen_exe,
        "--number_of_models", str(number_of_models),
        "--models_list", str(data_dir / "models-generic.dat"),
        "--experiments", "generic",
        "--theta", str(theta_file),
        "--sim_path", str(data_dir),
        "--exp_path", str(data_dir),
        "--data_ids", ",".join(data_ids),
        "--output_pkl", str(output_pkl),
    ]


def shell_quote(text: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=,+\-]+", text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def write_run_script(
    script_path: Path,
    data_dir: Path,
    theta_file: Path,
    output_pkl: Path,
    data_ids: Sequence[str],
    number_of_models: int,
    bioen_exe: str = "bioen",
) -> None:
    cmd = make_bioen_command(data_dir, theta_file, output_pkl, data_ids, number_of_models, bioen_exe)
    with script_path.open("w") as handle:
        handle.write("#!/bin/bash\n")
        handle.write("# Optional helper script for running the BioEn program (third party) separately.\n")
        handle.write("# The default bayes-infer multi-d workflow uses the local backend.\n\n")
        handle.write('echo "Running optional BioEn program (third party)"\n\n')
        handle.write(f'path="{data_dir}"\n')
        handle.write(f'flex_ids="{",".join(data_ids)}"\n\n')
        handle.write(" \\\n    ".join(shell_quote(x) for x in cmd) + "\n")
    script_path.chmod(0o755)


def read_data_ids_from_exp(data_dir: Path) -> List[str]:
    exp_file = data_dir / "exp-generic.dat"
    if not exp_file.exists():
        raise FileNotFoundError(f"Missing {exp_file}")
    data_ids: List[str] = []
    with exp_file.open() as handle:
        reader = csv.reader(handle, delimiter="\t")
        header_seen = False
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#") and not header_seen:
                header_seen = True
                continue
            if row[0].startswith("#"):
                continue
            data_ids.append(row[0].strip())
    return data_ids


def count_models(data_dir: Path) -> int:
    models = data_dir / "models-generic.dat"
    if not models.exists():
        raise FileNotFoundError(f"Missing {models}")
    with models.open() as handle:
        return sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))


def write_default_theta_file(out_dir: Path, theta: float = DEFAULT_THETA) -> Path:
    theta_file = out_dir / "thetas.dat"
    theta_file.write_text(f"{theta:.12g}\n")
    return theta_file


# ---------------------------------------------------------------------------
# Optional official BioEn command helpers.
# ---------------------------------------------------------------------------
def generate_theta_values(start: float, stop: float, n: int, spacing: str) -> List[float]:
    if n <= 0:
        raise ValueError("num-theta must be positive")
    if n == 1:
        return [start]
    if spacing == "linear":
        return [start + (stop - start) * i / (n - 1) for i in range(n)]
    if spacing == "log":
        if start <= 0.0 or stop <= 0.0:
            raise ValueError("log spacing requires positive start and stop")
        log_start = math.log10(start)
        log_stop = math.log10(stop)
        return [10.0 ** (log_start + (log_stop - log_start) * i / (n - 1)) for i in range(n)]
    raise ValueError(f"Unknown spacing: {spacing}")


def run_subprocess(cmd: Sequence[str], cwd: Path, log_file: Path) -> int:
    with log_file.open("w") as log:
        log.write("COMMAND:\n")
        log.write(" ".join(shell_quote(x) for x in cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def theta_scan(
    data_dir: Path,
    out_dir: Path,
    theta_values: Sequence[float],
    run_bioen: bool = False,
    bioen_exe: str = "bioen",
) -> None:
    data_dir = data_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_ids = read_data_ids_from_exp(data_dir)
    n_models = count_models(data_dir)

    bioen_path = shutil.which(bioen_exe) if os.path.sep not in bioen_exe else bioen_exe
    if run_bioen and bioen_path is None:
        raise RuntimeError(
            f"Cannot find BioEn executable '{bioen_exe}' on PATH. "
            "Activate the BioEn environment or provide --bioen-exe."
        )

    master_script = out_dir / "run_all_theta.sh"
    summary = out_dir / "theta_scan_commands.tsv"

    with master_script.open("w") as master, summary.open("w", newline="") as summary_handle:
        master.write("#!/bin/bash\nset -e\n\n")
        writer = csv.writer(summary_handle, delimiter="\t")
        writer.writerow(["theta_index", "theta", "theta_dir", "command"])

        for i, theta in enumerate(theta_values):
            theta_dir = out_dir / f"theta_{i:06d}"
            theta_dir.mkdir(parents=True, exist_ok=True)
            theta_file = theta_dir / "thetas.dat"
            theta_file.write_text(f"{theta:.12g}\n")

            target_data = data_dir

            output_pkl = theta_dir / "bioen_result.pkl"
            run_script = theta_dir / "run_bioen.sh"
            cmd = make_bioen_command(target_data, theta_file, output_pkl, data_ids, n_models, bioen_exe)
            write_run_script(run_script, target_data, theta_file, output_pkl, data_ids, n_models, bioen_exe)

            command_text = " ".join(shell_quote(x) for x in cmd)
            writer.writerow([i, f"{theta:.12g}", str(theta_dir), command_text])
            master.write(f"echo 'Running theta {i}: {theta:.12g}'\n")
            master.write(f"bash {shell_quote(str(run_script))}\n\n")

            if run_bioen:
                log_file = theta_dir / "bioen.log"
                code = run_subprocess(cmd, cwd=theta_dir, log_file=log_file)
                if code != 0:
                    raise RuntimeError(f"BioEn failed for theta index {i}; see {log_file}")

    master_script.chmod(0o755)


# ---------------------------------------------------------------------------
# CLI commands.
# ---------------------------------------------------------------------------
def cmd_make_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    restraints = build_restraints_nmr(
        raw_xls=Path(args.raw_xls),
        r1rho_xls=Path(args.r1rho_xls),
        skip_column_indices_1based=tuple(args.skip_column),
        finite_lower=args.finite_lower,
        finite_upper=args.finite_upper,
        default_noise=args.default_noise,
        broadened_distance=args.broadened_distance,
        broadened_noise=args.broadened_noise,
        long_distance=args.long_distance,
        long_distance_noise=args.long_distance_noise,
    )

    if args.atom_mode == "nmr-index":
        print(
            "WARNING: --atom-mode nmr-index uses fixed atom-index offsets and assumes "
            "the PDB atom order matches the NMR/PRE dataset. If atom order differs, "
            "use --atom-mode atom-name.",
            file=sys.stderr,
        )

    model_records, sim_by_id, warnings = build_simulated_distances(
        structures_dir=Path(args.structures_dir),
        restraints=restraints,
        expected_last_residue=args.expected_last_residue,
        atom_mode=args.atom_mode,
        drop_incomplete_models=not args.keep_incomplete_models,
    )

    write_bioen_generic_files(out_dir, restraints, model_records, sim_by_id)
    theta_file = write_default_theta_file(out_dir, args.default_theta)

    data_dir = out_dir / DEFAULT_MULTI_D_INPUT_DIR
    data_ids = [r.data_id for r in restraints]
    write_run_script(
        script_path=out_dir / "run_bioen.sh",
        data_dir=Path(DEFAULT_MULTI_D_INPUT_DIR),
        theta_file=Path("thetas.dat"),
        output_pkl=Path("bioen_result.pkl"),
        data_ids=data_ids,
        number_of_models=len(model_records),
        bioen_exe=args.bioen_exe,
    )

    metadata = {
        "raw_xls": str(args.raw_xls),
        "r1rho_xls": str(args.r1rho_xls),
        "structures_dir": str(args.structures_dir),
        "out_dir": str(out_dir),
        "n_restraints": len(restraints),
        "n_models": len(model_records),
        "atom_mode": args.atom_mode,
        "expected_last_residue": args.expected_last_residue,
        "skip_column_indices_1based": list(args.skip_column),
        "constants": {
            "t_inept": T_INEPT_DEFAULT,
            "omega_h": OMEGA_H_DEFAULT,
            "tau_c": TAU_C_DEFAULT,
            "K": K_CONST_DEFAULT,
            "finite_lower": args.finite_lower,
            "finite_upper": args.finite_upper,
            "default_noise": args.default_noise,
            "broadened_distance": args.broadened_distance,
            "broadened_noise": args.broadened_noise,
            "long_distance": args.long_distance,
            "long_distance_noise": args.long_distance_noise,
        },
        "theta_file": str(theta_file),
    }
    (out_dir / "core_protocol_metadata.json").write_text(json.dumps(metadata, indent=2))

    if warnings:
        (out_dir / "warnings.log").write_text("\n".join(warnings) + "\n")

    print(f"Wrote generic multi-d input files to: {data_dir}")
    print("These files use a BioEn-compatible generic observable format.")
    print(f"Wrote optional official BioEn run script: {out_dir / 'run_bioen.sh'}")
    print(f"Number of restraints: {len(restraints)}")
    print(f"Number of accepted models: {len(model_records)}")
    if warnings:
        print(f"Warnings written to: {out_dir / 'warnings.log'}")


def cmd_write_run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).resolve()
    data_ids = read_data_ids_from_exp(data_dir)
    n_models = count_models(data_dir)
    theta_file = Path(args.theta_file).resolve()
    if not theta_file.exists():
        theta_file.write_text(f"{args.theta:.12g}\n")
    output_pkl = Path(args.output_pkl).resolve()
    script_path = Path(args.script).resolve()
    write_run_script(script_path, data_dir, theta_file, output_pkl, data_ids, n_models, args.bioen_exe)
    print(f"Wrote optional official BioEn run script: {script_path}")


def cmd_theta_scan(args: argparse.Namespace) -> None:
    theta_values = generate_theta_values(args.start, args.stop, args.num_theta, args.spacing)
    theta_scan(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.output_root),
        theta_values=theta_values,
        run_bioen=args.run,
        bioen_exe=args.bioen_exe,
    )
    print(f"Wrote optional official BioEn theta scan to: {Path(args.output_root).resolve()}")
    print(f"Theta count: {len(theta_values)}")
    if not args.run:
        print("Official BioEn was not run. Use run_all_theta.sh or rerun with --run.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NMR-PRE data maker with optional official BioEn run-script helpers."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_make = sub.add_parser("make-data", help="Build multi-dimensional input files from NMR/PRE spreadsheets and PDBs")
    p_make.add_argument("--raw-xls", required=True, help="Raw PRE intensity-ratio spreadsheet, e.g. wt_tau_values_MTSL.xls")
    p_make.add_argument("--r1rho-xls", required=True, help="R1rho/R2 spreadsheet, e.g. R1rho_Values.xls")
    p_make.add_argument("--structures-dir", required=True, help="Directory containing PDB models")
    p_make.add_argument("--out-dir", default="output_multi_d", help="Output directory")
    p_make.add_argument("--skip-column", type=int, nargs="*", default=[8], help="1-based raw spreadsheet column indices to skip; default: 8")
    p_make.add_argument("--expected-last-residue", type=int, default=EXPECTED_LAST_RESIDUE_DEFAULT, help="Expected last residue in first chain; default: 441")
    p_make.add_argument("--atom-mode", choices=["nmr-index", "atom-name"], default="nmr-index", help="Distance atom selection mode; default uses NMR/PRE atom-index offsets")
    p_make.add_argument("--keep-incomplete-models", action="store_true", help="Keep models with missing simulated distances as NaN; not recommended")
    p_make.add_argument("--finite-lower", type=float, default=FINITE_LOWER_DEFAULT)
    p_make.add_argument("--finite-upper", type=float, default=FINITE_UPPER_DEFAULT)
    p_make.add_argument("--default-noise", type=float, default=DEFAULT_NOISE_DEFAULT)
    p_make.add_argument("--broadened-distance", type=float, default=BROADENED_DISTANCE_DEFAULT)
    p_make.add_argument("--broadened-noise", type=float, default=BROADENED_NOISE_DEFAULT)
    p_make.add_argument("--long-distance", type=float, default=LONG_DISTANCE_DEFAULT)
    p_make.add_argument("--long-distance-noise", type=float, default=LONG_DISTANCE_NOISE_DEFAULT)
    p_make.add_argument("--default-theta", type=float, default=DEFAULT_THETA, help="Single default theta written to thetas.dat")
    p_make.add_argument("--bioen-exe", default="bioen", help="BioEn executable name/path used for optional run_bioen.sh")
    p_make.set_defaults(func=cmd_make_data)

    p_run = sub.add_parser("write-run", help="Write an optional official BioEn run script for an existing generic multi-d input directory")
    p_run.add_argument("--data-dir", required=True, help="Generic multi-d input directory; old BioEN_Files folders are also supported")
    p_run.add_argument("--theta-file", default="thetas.dat", help="Theta file; created if missing")
    p_run.add_argument("--theta", type=float, default=DEFAULT_THETA, help="Theta used if theta-file must be created")
    p_run.add_argument("--output-pkl", default="bioen_result.pkl")
    p_run.add_argument("--script", default="run_bioen.sh")
    p_run.add_argument("--bioen-exe", default="bioen")
    p_run.set_defaults(func=cmd_write_run)

    p_scan = sub.add_parser("theta-scan", help="Create and optionally run official BioEn jobs for a theta scan")
    p_scan.add_argument("--data-dir", required=True, help="Generic multi-d input directory; old BioEN_Files folders are also supported")
    p_scan.add_argument("--output-root", default="theta_scan", help="Theta scan output directory")
    p_scan.add_argument("--start", type=float, default=1e-2)
    p_scan.add_argument("--stop", type=float, default=1e5)
    p_scan.add_argument("--num-theta", type=int, default=80)
    p_scan.add_argument("--spacing", choices=["log", "linear"], default="log")
    p_scan.add_argument("--run", action="store_true", help="Run the optional official BioEn command for each theta")
    p_scan.add_argument("--bioen-exe", default="bioen")
    p_scan.set_defaults(func=cmd_theta_scan)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
