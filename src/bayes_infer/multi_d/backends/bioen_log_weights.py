"""Local BioEn-compatible log-weight optimizer for multi-d generic data.

This is the default multi-d backend for bayes-infer.  It follows BioEn's
standard/default log-weight formulation for generic observables, but keeps the
implementation narrow and pure Python/NumPy/SciPy for portability.

Objective convention
--------------------
The minimized objective is

    theta * S_KL(w || w0) + 0.5 * sum_alpha ((<x_alpha>_w - X_alpha)/sigma_alpha)^2

where ``w`` is represented by unconstrained log-weight variables and normalized
with a stable softmax transform.  The text output reports both the half-chi2
term and the full chi2 sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import json
import warnings

import numpy as np
import scipy.optimize as sopt

from ..generic_io import GenericData, load_generic_data
from . import bioen_common as common


@dataclass(frozen=True)
class LocalBioEnLogWeightsResult:
    theta: float
    weights: np.ndarray
    log_weights: np.ndarray
    fit: np.ndarray
    chi2: float
    entropy: float
    objective_initial: float
    objective_final: float
    nrestraints: int
    nmodels: int
    converged: bool
    warnflag: int
    task: str

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.nrestraints, 1)


def uniform_weights(nmodels: int) -> np.ndarray:
    return np.full((nmodels, 1), 1.0 / float(nmodels), dtype=float)


def weights_from_log_weights(log_weights: np.ndarray, w0: np.ndarray | None = None) -> np.ndarray:
    """Return normalized weights from log-weight variables.

    ``log_weights`` are interpreted as additive log factors relative to ``w0``.
    With uniform ``w0`` this is equivalent to the standard softmax.
    """
    z = np.asarray(log_weights, dtype=float).reshape(-1)
    if w0 is None:
        base = np.zeros_like(z)
    else:
        w0_flat = np.asarray(w0, dtype=float).reshape(-1)
        if w0_flat.shape != z.shape:
            raise ValueError("w0 and log_weights must have the same length")
        if np.any(w0_flat <= 0):
            raise ValueError("w0 must be strictly positive for log-weight optimization")
        base = np.log(w0_flat)
    a = base + z
    a = a - np.max(a)
    w = np.exp(a)
    w /= np.sum(w)
    return w.reshape(-1, 1)


def bioen_log_posterior_base(
    log_weights: np.ndarray,
    w0: np.ndarray,
    y_tilde: np.ndarray,
    y_exp_tilde: np.ndarray,
    theta: float,
) -> float:
    w = weights_from_log_weights(log_weights, w0)
    s_kl = common.relative_entropy(w0, w)
    chi2 = common.chi_sqr_term(w, y_tilde, y_exp_tilde)
    return float(theta * s_kl + chi2)


def grad_bioen_log_posterior_base(
    log_weights: np.ndarray,
    w0: np.ndarray,
    y_tilde: np.ndarray,
    y_exp_tilde: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Analytic gradient with respect to unconstrained log-weight variables."""
    w = weights_from_log_weights(log_weights, w0)  # N x 1
    y_ave = common.get_ave(w, y_tilde)  # M
    residual = y_ave - np.asarray(y_exp_tilde, dtype=float).reshape(-1)  # M
    # d/dw_i of the half-chi2 term is sum_alpha residual_alpha * y_alpha_i
    chi_grad_w = np.asarray(y_tilde, dtype=float).T @ residual  # N
    w_flat = w.reshape(-1)
    w0_flat = np.asarray(w0, dtype=float).reshape(-1)
    entropy_grad_w = np.log(w_flat / w0_flat) + 1.0
    grad_w = theta * entropy_grad_w + chi_grad_w
    # softmax Jacobian: dF/dz_i = w_i * (dF/dw_i - sum_j w_j dF/dw_j)
    centered = grad_w - float(np.dot(w_flat, grad_w))
    return w_flat * centered


def find_optimum_log_weights(
    w0: np.ndarray,
    y: np.ndarray,
    y_tilde: np.ndarray,
    y_exp_tilde: np.ndarray,
    theta: float,
    log_weights_init: np.ndarray | None = None,
    *,
    pgtol: float = 1.0e-3,
    max_iterations: int = 5000,
    verbose: bool = False,
) -> LocalBioEnLogWeightsResult:
    """Optimize generic BioEn weights using the default log-weight parameterization."""
    m, n = y_tilde.shape
    w0 = np.asarray(w0, dtype=float).reshape(n, 1)
    if log_weights_init is None:
        log_weights_init = np.zeros(n, dtype=float)
    else:
        log_weights_init = np.asarray(log_weights_init, dtype=float).reshape(n)

    f_initial = bioen_log_posterior_base(log_weights_init, w0, y_tilde, y_exp_tilde, theta)
    x_opt, f_final, grad_final, hess_inv, func_calls, grad_calls, warnflag = sopt.fmin_bfgs(
        bioen_log_posterior_base,
        log_weights_init,
        args=(w0, y_tilde, y_exp_tilde, theta),
        fprime=grad_bioen_log_posterior_base,
        gtol=pgtol,
        maxiter=max_iterations,
        full_output=True,
        disp=verbose,
    )
    info = {
        "warnflag": int(warnflag),
        "task": "scipy fmin_bfgs",
        "funcalls": int(func_calls),
        "gradcalls": int(grad_calls),
    }
    weights = weights_from_log_weights(x_opt, w0)
    fit = common.get_ave(weights, y)
    entropy = common.relative_entropy(w0, weights)
    chi2 = common.chi_sqr_term(weights, y_tilde, y_exp_tilde)
    return LocalBioEnLogWeightsResult(
        theta=float(theta),
        weights=weights.reshape(-1),
        log_weights=np.asarray(x_opt, dtype=float).reshape(-1),
        fit=fit.reshape(-1),
        chi2=float(chi2),
        entropy=float(entropy),
        objective_initial=float(f_initial),
        objective_final=float(f_final),
        nrestraints=m,
        nmodels=n,
        converged=int(info.get("warnflag", 1)) == 0,
        warnflag=int(info.get("warnflag", 1)),
        task=str(info.get("task", "")),
    )


def run_local_generic(
    data_dir: Path | str,
    theta: float,
    out_dir: Path | str,
    data_ids: Sequence[str] | str | None = None,
    *,
    pgtol: float = 1.0e-3,
    max_iterations: int = 5000,
    verbose: bool = False,
) -> LocalBioEnLogWeightsResult:
    """Load generic files, run the local default backend, and write outputs."""
    data = load_generic_data(data_dir, data_ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w0 = uniform_weights(data.nmodels)
    result = find_optimum_log_weights(
        w0=w0,
        y=data.sim,
        y_tilde=data.sim_scaled,
        y_exp_tilde=data.exp_scaled,
        theta=theta,
        pgtol=pgtol,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    write_local_outputs(data, result, out_dir)
    return result


def write_local_outputs(data: GenericData, result: LocalBioEnLogWeightsResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "weights.dat").open("w") as fh:
        for model_id, weight in zip(data.model_ids, result.weights):
            fh.write(f"{model_id} {weight:.16e}\n")
    with (out_dir / "log_weights.dat").open("w") as fh:
        for model_id, value in zip(data.model_ids, result.log_weights):
            fh.write(f"{model_id} {value:.16e}\n")
    with (out_dir / "fit.tsv").open("w") as fh:
        fh.write("data_id\texperiment\tsigma\tcalculated\tresidual_over_sigma\n")
        for data_id, exp, sigma, calc in zip(data.data_ids, data.exp, data.sigma, result.fit):
            fh.write(f"{data_id}\t{exp:.16g}\t{sigma:.16g}\t{calc:.16g}\t{(calc-exp)/sigma:.16g}\n")
    top = np.argsort(result.weights)[::-1][:50]
    model_arr = np.asarray(data.model_ids)
    with (out_dir / "top50_weights.dat").open("w") as fh:
        for idx in top:
            fh.write(f"{model_arr[idx]} {result.weights[idx]:.16e}\n")
    summary = {
        "backend": "local-log-weights",
        "theta": result.theta,
        "chi2_half_sum": result.chi2,
        "chi2_full_sum": 2.0 * result.chi2,
        "reduced_chi2_half_sum": result.reduced_chi2,
        "entropy_skl": result.entropy,
        "objective_initial": result.objective_initial,
        "objective_final": result.objective_final,
        "nrestraints": result.nrestraints,
        "nmodels": result.nmodels,
        "converged": result.converged,
        "warnflag": result.warnflag,
        "task": result.task,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        warnings.warn(
            f"Could not write result.pkl because pandas is unavailable: {exc}",
            RuntimeWarning,
        )
        return

    output = {
        result.theta: {
            "optimization_method": "log-weights",
            "optimization_algorithm": "bfgs",
            "optimization_minimizer": "scipy",
            "w0": uniform_weights(data.nmodels),
            "winit": uniform_weights(data.nmodels),
            "wopt": result.weights.reshape(-1, 1),
            "sim_init": data.sim @ uniform_weights(data.nmodels),
            "sim_wopt": result.fit.reshape(-1, 1),
            "S": result.entropy,
            "chi2": result.chi2,
            "theta": result.theta,
            "nrestraints": result.nrestraints,
            "nmodels": result.nmodels,
            "nmodels_list": data.model_ids,
            "exp": {"generic": dict(zip(data.data_ids, data.exp))},
            "exp_err": {"generic": dict(zip(data.data_ids, data.sigma))},
        }
    }
    pd.DataFrame.from_dict(output).to_pickle(out_dir / "result.pkl")
