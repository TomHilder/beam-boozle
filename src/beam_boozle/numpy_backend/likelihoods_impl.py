import numpy as np
from numpy.typing import ArrayLike


def slow_log_likelihood(
    data: ArrayLike,
    model: ArrayLike,
    covariance_inverse: ArrayLike,
) -> np.ndarray:
    """
    NumPy-backend multivariate Gaussian log-likelihood using the dense inverse covariance matrix.
    Very expensive and slow, intended for validation on small problems. Neglects normalisation
    constant.

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
        -0.5 * R.T @ Cinv @ R
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
    NumPy-backend fast multivariate Gaussian log-likelihood using FFTs and inverse PSD. Neglects
    normalisation constant. Assumes stationary noise covariance.

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
        Scalar NumPy array with the quadratic log-likelihood term:
        -0.5 * R.T @ Cinv @ R
        (excludes the log-determinant constant).
    """
    residual = np.asarray(data) - np.asarray(model)
    psd_inv = np.asarray(psd_inverse)

    residual_ft = np.fft.fft2(residual)  # Unnormalized FFT
    n_pixels = residual.size  # H * W

    quadratic_form = np.real(np.sum(np.conj(residual_ft) * (residual_ft * psd_inv))) / n_pixels
    return -0.5 * quadratic_form
