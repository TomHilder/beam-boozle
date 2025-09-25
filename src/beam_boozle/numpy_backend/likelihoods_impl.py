import numpy as np
from numpy.typing import ArrayLike


def slow_log_likelihood(
    data: ArrayLike,
    model: ArrayLike,
    covariance_inverse: ArrayLike,
) -> np.ndarray:
    """
    Exact Gaussian log-likelihood using the full inverse covariance matrix.

    Notes
    -----
    - Intended for validation or small problems (computationally expensive).
    - `data` and `model` are flattened before use and must have the same size.
      After flattening, their length must match the dimension of `covariance_inverse`.

    Parameters
    ----------
    data : ArrayLike
        Observed data (any array-like; broadcast-compatible with `model`).
    model : ArrayLike
        Model prediction (same shape as `data` after broadcasting).
    covariance_inverse : ArrayLike
        Inverse covariance matrix with shape (N, N).

    Returns
    -------
    np.ndarray
        Scalar NumPy array with the quadratic log-likelihood term:
        -0.5 * rᵀ C⁻¹ r
        (excludes the log-determinant constant).
    """
    residual_1d = (np.asarray(data) - np.asarray(model)).reshape(-1)
    cov_inv = np.asarray(covariance_inverse)
    return -0.5 * residual_1d @ (cov_inv @ residual_1d)


def fast_log_likelihood(
    data: ArrayLike,
    model: ArrayLike,
    psd_inverse: ArrayLike,
) -> np.ndarray:
    """
    Approximate Gaussian log-likelihood using FFTs under stationarity,
    where the covariance is diagonal in the Fourier basis.

    Assumptions
    -----------
    - Stationary noise (circulant covariance).
    - Periodic boundary conditions.
    - `psd_inverse` matches the FFT grid and conventions of `np.fft.fft2`
      used here (unnormalized transform).

    Parameters
    ----------
    data : ArrayLike
        Observed 2D data array.
    model : ArrayLike
        Model prediction with the same shape as `data`.
    psd_inverse : ArrayLike
        Inverse power spectral density evaluated on the FFT grid.

    Returns
    -------
    np.ndarray
        Scalar NumPy array with the quadratic term in Fourier space:
        -0.5 * (1/(H*W)) * Σ_k |R_k|^2 * S_k^{-1}
        (excludes the log-determinant constant).
    """
    residual = np.asarray(data) - np.asarray(model)
    psd_inv = np.asarray(psd_inverse)

    residual_ft = np.fft.fft2(residual)  # Unnormalized FFT
    n_pixels = residual.size  # H * W

    quadratic_form = np.real(np.sum(np.conj(residual_ft) * (residual_ft * psd_inv))) / n_pixels
    return -0.5 * quadratic_form
