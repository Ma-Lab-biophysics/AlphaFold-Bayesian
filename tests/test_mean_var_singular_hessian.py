import numpy as np

from bayes_infer.mean_var.mean_variance import MeanVarInput, fit


def test_mean_var_handles_singular_hessian_when_gradient_is_zero():
    data = MeanVarInput(
        theta=1.0,
        y_obs=0.5,
        sigma=0.1,
        var_obs=0.0,
        sigma_var=0.1,
        prior=np.ones(5) / 5,
        y=np.full(5, 0.5),
    )
    result = fit(data, ntheta=2, pow_factor=1.5, niter=5)
    assert np.isfinite(result.chi2)
    assert np.all(result.weights > 0)
    assert np.isclose(np.sum(result.weights), 1.0)
