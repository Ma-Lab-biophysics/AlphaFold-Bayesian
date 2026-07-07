"""Small BioEn-compatible math utilities for the local multi-d backend.

The formulas mirror the pure-Python functions in BioEn's ``bioen.optimize.common``
module, restricted to the generic-observable use case needed here.
"""
from __future__ import annotations

import numpy as np


def chi_sqr_term(w: np.ndarray, y_tilde: np.ndarray, y_exp_tilde: np.ndarray) -> float:
    """Return BioEn's chi-square term: 0.5 * ||<y/sigma>_w - Y/sigma||^2."""
    w_col = np.asarray(w, dtype=float).reshape(-1, 1)
    y_tilde = np.asarray(y_tilde, dtype=float)
    y_exp_tilde = np.asarray(y_exp_tilde, dtype=float).reshape(1, -1)
    v = (y_tilde @ w_col).T - y_exp_tilde
    return float(0.5 * (v @ v.T)[0, 0])


def get_ave(w: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return weighted average over models for an M x N observable matrix."""
    w_col = np.asarray(w, dtype=float).reshape(-1, 1)
    return (np.asarray(y, dtype=float) @ w_col)[:, 0]


def relative_entropy(w_ref: np.ndarray, w: np.ndarray) -> float:
    """Return S_KL = sum_i w_i log(w_i / w_ref_i)."""
    w = np.asarray(w, dtype=float).reshape(-1)
    w_ref = np.asarray(w_ref, dtype=float).reshape(-1)
    mask = w > 0.0
    return float(np.sum(w[mask] * np.log(w[mask] / w_ref[mask])))
