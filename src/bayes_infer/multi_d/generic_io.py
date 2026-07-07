"""Generic multi-dimensional input/output utilities.

This module implements the narrow generic-data reader needed by bayes-infer.
Its file conventions use BioEn-compatible generic observable files:

    exp-generic.dat
    models-generic.dat
    sim-<DATA_ID>-generic.dat

The reader intentionally keeps the data model simple: experimental values and
simulated values are loaded as raw arrays, then scaled by the per-restraint
experimental noise before optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GenericData:
    """Container for generic multi-dimensional restraints.

    Attributes
    ----------
    data_dir:
        Directory containing generic multi-d input files.
    data_ids:
        Restraint/observable identifiers in the order used for fitting.
    model_ids:
        Model identifiers from ``models-generic.dat``.
    exp:
        Experimental values, shape ``(M,)``.
    sigma:
        Experimental/model uncertainty values, shape ``(M,)``.
    sim:
        Simulated observable matrix, shape ``(M, N)`` where ``M`` is the number
        of restraints and ``N`` is the number of models.
    """

    data_dir: Path
    data_ids: tuple[str, ...]
    model_ids: tuple[int, ...]
    exp: np.ndarray
    sigma: np.ndarray
    sim: np.ndarray

    @property
    def nrestraints(self) -> int:
        return int(self.exp.shape[0])

    @property
    def nmodels(self) -> int:
        return int(self.sim.shape[1])

    @property
    def exp_scaled(self) -> np.ndarray:
        return self.exp / self.sigma

    @property
    def sim_scaled(self) -> np.ndarray:
        return self.sim / self.sigma[:, None]


def _noncomment_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    out: list[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def read_models(data_dir: Path) -> tuple[int, ...]:
    """Read model identifiers from ``models-generic.dat``."""
    values: list[int] = []
    for line in _noncomment_lines(data_dir / "models-generic.dat"):
        token = line.split()[0]
        try:
            values.append(int(token))
        except ValueError:
            # Keep compatibility with model lists that use one model filename per
            # line by mapping them onto 0-based positions.
            values.append(len(values))
    if not values:
        raise ValueError(f"No models found in {data_dir / 'models-generic.dat'}")
    return tuple(values)


def read_exp_records(data_dir: Path) -> dict[str, tuple[float, float]]:
    """Read ``exp-generic.dat`` into an ordered dict-like mapping."""
    records: dict[str, tuple[float, float]] = {}
    for line in _noncomment_lines(data_dir / "exp-generic.dat"):
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Malformed exp-generic.dat line: {line!r}")
        data_id = parts[0]
        if data_id in records:
            raise ValueError(f"Duplicate experimental data ID: {data_id}")
        value = float(parts[1])
        sigma = float(parts[2])
        if sigma <= 0:
            raise ValueError(f"Non-positive uncertainty for {data_id}: {sigma}")
        records[data_id] = (value, sigma)
    if not records:
        raise ValueError(f"No restraints found in {data_dir / 'exp-generic.dat'}")
    return records


def resolve_data_ids(data_dir: Path, data_ids: Sequence[str] | str | None = None) -> tuple[str, ...]:
    """Resolve requested IDs against ``exp-generic.dat``.

    ``None`` and ``'all'`` use the experimental-file order.  Prefix-tolerant
    data ID resolution supports generic examples when the prefix is unique.
    """
    records = read_exp_records(data_dir)
    available = tuple(records.keys())
    if data_ids is None:
        return available
    if isinstance(data_ids, str):
        requested = tuple(x.strip() for x in data_ids.split(",") if x.strip())
    else:
        requested = tuple(data_ids)
    if len(requested) == 1 and requested[0] == "all":
        return available

    resolved: list[str] = []
    for data_id in requested:
        if data_id in records:
            resolved.append(data_id)
            continue
        matches = [x for x in available if x.startswith(data_id) or data_id.startswith(x)]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        raise KeyError(f"Requested data ID {data_id!r} not found in exp-generic.dat")
    return tuple(resolved)


def read_sim_values(data_dir: Path, data_id: str, nmodels: int) -> np.ndarray:
    """Read one ``sim-<ID>-generic.dat`` file."""
    path = data_dir / f"sim-{data_id}-generic.dat"
    values = np.genfromtxt(path, comments="#", dtype=float)
    values = np.atleast_1d(values).astype(float)
    if values.shape[0] != nmodels:
        raise RuntimeError(
            f"Number of simulated values in {path} ({values.shape[0]}) does not match "
            f"number of models ({nmodels})."
        )
    return values


def load_generic_data(data_dir: Path | str, data_ids: Sequence[str] | str | None = None) -> GenericData:
    """Load a generic multi-dimensional dataset."""
    data_dir = Path(data_dir).resolve()
    model_ids = read_models(data_dir)
    records = read_exp_records(data_dir)
    ids = resolve_data_ids(data_dir, data_ids)
    exp = np.array([records[x][0] for x in ids], dtype=float)
    sigma = np.array([records[x][1] for x in ids], dtype=float)
    sim = np.vstack([read_sim_values(data_dir, x, len(model_ids)) for x in ids])
    return GenericData(data_dir=data_dir, data_ids=ids, model_ids=model_ids, exp=exp, sigma=sigma, sim=sim)
