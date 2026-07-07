#!/usr/bin/env python3
"""Run AvTraj on ./Structures/frame_*.pdb and write FRET efficiencies.

Expected input layout:
  ./Structures/frame_0.pdb
  ./Structures/frame_1.pdb
  ...

The first frame, ./Structures/frame_0.pdb, is used as the topology file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import avtraj as avt
import mdtraj as md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AvTraj on PDB frames and write FRET efficiencies."
    )
    parser.add_argument(
        "--structures-dir",
        default="Structures",
        help="Directory containing frame_0.pdb, frame_1.pdb, ... Default: Structures",
    )
    parser.add_argument(
        "--labeling-file",
        default="labeling.fps.json",
        help="AvTraj/ChiSurf/Olga-style labeling JSON file. Default: labeling.fps.json",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=51.0,
        help="Forster radius R0 used to convert rDAE to FRET efficiency. Default: 51.0",
    )
    parser.add_argument(
        "--w0",
        type=float,
        default=0.0001,
        help="Initial weight provided in fret.input. Default: 0.0001",
    )
    parser.add_argument(
        "--output-prefix",
        default="avtraj",
        help="Prefix for intermediate output files. Default: avtraj",
    )
    parser.add_argument(
        "--input-bayes",
        default="fret.input",
        help="Mean-var bayes-infer input file. Default: fret.input",
    )
    parser.add_argument(
        "--save-av",
        action="store_true",
        help="Save accessible-volume XYZ files for the first frame.",
    )
    return parser.parse_args()


def frame_number(path: Path) -> int:
    """Return the integer suffix in frame_<N>.pdb."""
    match = re.fullmatch(r"frame_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected frame filename: {path.name}; expected frame_<N>.pdb")
    return int(match.group(1))


def load_structures(structures_dir: Path) -> md.Trajectory:
    """Load Structures/frame_*.pdb as a trajectory using frame_0.pdb as topology."""
    frame_files = sorted(structures_dir.glob("frame_*.pdb"), key=frame_number)
    if not frame_files:
        raise FileNotFoundError(
            f"No PDB frames found in {structures_dir.resolve()} matching frame_*.pdb"
        )

    topology_file = structures_dir / "frame_0.pdb"
    if not topology_file.exists():
        raise FileNotFoundError(
            f"Topology file not found: {topology_file}. Expected Structures/frame_0.pdb."
        )

    traj = md.load([str(p) for p in frame_files], top=str(topology_file))
    print(f"Loaded {traj.n_frames} frames from {structures_dir} using topology {topology_file}")
    return traj


def make_av_trajectories(traj: md.Trajectory, save_av: bool = False) -> None:
    """Build donor and acceptor accessible volumes for the UvrD example."""
    av_parameters_donor = {
        "simulation_type": "AV1",
        "linker_length": 21.0,
        "linker_width": 2.0,
        "radius1": 3.5,
        "simulation_grid_resolution": 1.0,
        "residue_seq_number": 473,
        "atom_name": "CA",
        "chain_identifier": "A",
        "contact_volume_thickness": 0,
        "contact_volume_trapped_fraction": 0.8,
    }

    av_traj_donor = avt.AVTrajectory(traj, av_parameters=av_parameters_donor, name="473")

    av_parameters_acceptor = {
        "simulation_type": "AV1",
        "linker_length": 21.0,
        "linker_width": 2.0,
        "radius1": 7.5,
        "simulation_grid_resolution": 1.0,
        "residue_seq_number": 100,
        "atom_name": "CA",
        "chain_identifier": "A",
        "contact_volume_thickness": 0,
        "contact_volume_trapped_fraction": 0.8,
    }

    av_traj_acceptor = avt.AVTrajectory(traj, av_parameters=av_parameters_acceptor, name="100")

    if save_av:
        av_traj_donor[0].save_av()
        av_traj_acceptor[0].save_av()


def as_float_list(value) -> List[float]:
    """Convert scalar/list/array-like values to a flat list of finite floats."""
    if isinstance(value, (str, bytes)):
        raw_values: Iterable = [value]
    elif isinstance(value, Iterable):
        raw_values = value
    else:
        raw_values = [value]

    floats: List[float] = []
    for raw in raw_values:
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite rDAE value found: {raw}")
        floats.append(val)
    return floats


def extract_rdae_from_object(obj) -> List[float]:
    """Extract rDAE values directly from an AvTraj object/dict, with text fallback."""
    if isinstance(obj, dict) and "rDAE" in obj:
        return as_float_list(obj["rDAE"])

    text = repr(obj)
    matches = re.findall(r"['\"]rDAE['\"]\s*:\s*\[([^\]]+)\]", text, flags=re.S)
    values: List[float] = []
    for match in matches:
        for token in re.split(r"[,\s]+", match.strip()):
            if token:
                values.extend(as_float_list(token))
    return values


def compute_av_distances(traj: md.Trajectory, labeling_file: Path) -> List[float]:
    """Compute AvTraj distance trajectory and return one rDAE value per frame/model."""
    if not labeling_file.exists():
        raise FileNotFoundError(f"Labeling file not found: {labeling_file}")

    with labeling_file.open("r", encoding="utf-8") as handle:
        labeling = json.load(handle)

    av_dist = avt.AvDistanceTrajectory(traj, labeling)

    rdae_values: List[float] = []
    for i in range(len(av_dist)):
        values = extract_rdae_from_object(av_dist[i])
        if not values:
            raise ValueError(f"Could not find rDAE value in AvTraj output for frame {i}: {av_dist[i]!r}")
        if len(values) != 1:
            raise ValueError(
                f"Expected one rDAE value for frame {i}, but found {len(values)} values: {values}. "
                "If you have multiple observables, write one output file per observable."
            )
        rdae_values.append(values[0])

    if len(rdae_values) != traj.n_frames:
        raise ValueError(
            f"Number of rDAE values ({len(rdae_values)}) does not match number of frames ({traj.n_frames})."
        )
    return rdae_values


def compute_fret_efficiency(r_da: float, r0: float) -> float:
    """Convert donor-acceptor distance rDA to FRET efficiency."""
    return 1.0 / (1.0 + (r_da / r0) ** 6)


def write_fret_input(values: Sequence[float], path: Path, r0: float, w0: float) -> None:
    """Write one row per model: w0 E_model."""
    with path.open("w", encoding="utf-8") as handle:
        for r_da in values:
            e_model = compute_fret_efficiency(r_da, r0)
            handle.write(f"{w0:.8g} {e_model:.10f}\n")


def main() -> None:
    args = parse_args()

    structures_dir = Path(args.structures_dir)
    labeling_file = Path(args.labeling_file)
    numeric_values_file = Path(f"{args.output_prefix}_rDAE_values.dat")
    input_bayes = Path(args.input_bayes)

    traj = load_structures(structures_dir)
    make_av_trajectories(traj, save_av=args.save_av)
    rdae_values = compute_av_distances(traj, labeling_file)

    numeric_values_file.write_text("".join(f"{v:.8f}\n" for v in rdae_values), encoding="utf-8")
    write_fret_input(rdae_values, input_bayes, r0=args.r0, w0=args.w0)

    print(f"Extracted {len(rdae_values)} rDAE values")
    print(f"Wrote numeric rDAE values to {numeric_values_file}")
    print(f"Wrote bayes-infer mean-var input file to {input_bayes}")


if __name__ == "__main__":
    main()
