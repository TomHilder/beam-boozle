import jax.numpy as jnp
from numpy.typing import ArrayLike


def slow_log_likelihood(
    data: ArrayLike,
    model: ArrayLike,
    covariance_inverse: ArrayLike,
) -> jnp.ndarray:
    """
    JAX-backend multivariate Gaussian log-likelihood using the dense inverse covariance matrix.
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
    jnp.ndarray
        Scalar JAX array with the quadratic log-likelihood term:
        -0.5 * R.T @ Cinv @ R
        (excludes the log-determinant constant).
    """
    # Input safety
    residual_1d = (jnp.asarray(data) - jnp.asarray(model)).reshape(-1)
    cov_inv = jnp.asarray(covariance_inverse)
    # Calc
    return -0.5 * residual_1d @ (cov_inv @ residual_1d)


def fast_log_likelihood(
    data: ArrayLike,
    model: ArrayLike,
    psd_inverse: ArrayLike,
) -> jnp.ndarray:
    """
    JAX-backend fast multivariate Gaussian log-likelihood using FFTs and inverse PSD. Neglects
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
    jnp.ndarray
        Scalar JAX array with the quadratic log-likelihood term:
        -0.5 * R.T @ Cinv @ R
        (excludes the log-determinant constant).
    """
    # Input safety
    residual = jnp.asarray(data) - jnp.asarray(model)
    psd_inv = jnp.asarray(psd_inverse)
    # FFT the residual; covariance is diagonal in Fourier space
    residual_ft = jnp.fft.fft2(residual)
    n_pixels = residual.size  # H * W
    # Calculation equivalent to R.T @ Cinv @ R; also divide by n_pixels to normalise the FFT
    quadratic_form = jnp.real(jnp.sum(jnp.conj(residual_ft) * (residual_ft * psd_inv))) / n_pixels
    return -0.5 * quadratic_form
