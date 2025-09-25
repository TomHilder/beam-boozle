from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

SafetyMode = Literal["ridge", "pinv"]


def _ensure_stack(images: ArrayLike) -> np.ndarray:
    """
    Ensure input is a stack of 2D images.

    Parameters
    ----------
    images : ArrayLike
        Either a single 2D array (H, W) or a stack (R, H, W).

    Returns
    -------
    stack : np.ndarray
        Array of shape (R, H, W) with dtype promoted to float.
    """
    arr = np.asarray(images, dtype=float)
    if arr.ndim == 2:
        return arr[0:1, ...]  # (1, H, W) without copying when possible
    if arr.ndim == 3:
        return arr
    raise ValueError(f"`images` must be 2D or 3D (got shape {arr.shape}).")


def estimate_noise_acf_masked(
    images: ArrayLike,
    mask: Optional[ArrayLike] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """
    Estimate the autocovariance function (ACF) for stationary noise. Keeps the
    variance scale in the DC mode. Supports a binary mask.

    Parameters
    ----------
    images : ArrayLike
        Noise-dominated data, shape (R, H, W) or (H, W).
    mask : ArrayLike, optional
        Binary mask (H, W): 1 = include, 0 = exclude. If None, all ones.
    eps : float, default 1e-12
        Small safety constant for divisions.

    Returns
    -------
    acf : np.ndarray
        Autocovariance, shape (H, W). DC at [0, 0], containing variance scale.
    var_hat : float
        Marginal variance.
    """
    X = _ensure_stack(images)  # (R, H, W)
    R, H, W = X.shape
    # Construct/validate mask
    if mask is None:
        M = np.ones((H, W), dtype=float)
    else:
        M = np.asarray(mask, dtype=float)
        if M.shape != (H, W):
            raise ValueError(f"`mask` shape {M.shape} must match image shape {(H, W)}.")
    # Check number of excluded pixels
    msum = float(np.sum(M))
    if msum <= 0:
        raise ValueError("`mask` excludes all pixels (sum == 0).")
    # Apply mask
    Y = X * M  # (R, H, W)
    # FFTs (unnormalised)
    A = np.fft.fft2(Y, axes=(-2, -1))  # (R, H, W)
    Mf = np.fft.fft2(M, axes=(-2, -1))  # (H, W)
    # Numerator: mean periodogram across realisations, then IFFT to lags
    num = np.fft.ifft2(np.mean(A * np.conj(A), axis=0), axes=(-2, -1)).real  # (H, W)
    # Denominator: valid-pair counts via autocorrelation of the mask
    den = np.fft.ifft2(Mf * np.conj(Mf), axes=(-2, -1)).real + eps  # (H, W)
    # Mask-corrected autocovariance
    acf = num / den  # (H, W)
    # Masked variance (second moment over included pixels, averaged over R)
    var_hat = float(np.sum(Y**2) / (msum * R + eps))
    # Enforce zero-lag exactly (keeps downstream PSD scaling consistent)
    acf[0, 0] = var_hat
    return acf, var_hat


def acf_to_psd(
    acf: ArrayLike,
    var_target: Optional[float] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert an autocovariance (ACF) to a nonnegative power spectral density (PSD)
    using the Wiener–Khinchin theorem.

    Parameters
    ----------
    acf : ArrayLike
        Autocovariance on circular lags, shape (H, W). DC at [0, 0].
    var_target : float, optional
        If provided, rescale PSD to satisfy Parseval:
            (1/(H*W)) * Σ_k PSD[k] = var_target
    eps : float, default 1e-12
        Small safety constant for normalisation.

    Returns
    -------
    psd : np.ndarray
        Real, nonnegative PSD, shape (H, W), DC at [0, 0].
    """
    # Calc PSD
    A = np.asarray(acf, dtype=float)
    H, W = A.shape[-2], A.shape[-1]
    psd = np.fft.fft2(A, axes=(-2, -1)).real
    # Optional re-scaling
    if var_target is not None:
        total = np.sum(psd)
        scale = (var_target * (H * W)) / (total + eps)
        psd = psd * scale
    # Clip tiny negatives from numerical round-off
    psd = np.maximum(psd, 0.0)
    return psd


def estimate_noise_psd_from_data(
    images: ArrayLike,
    mask: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the power spectral density (PSD) for stationary noise. Keeps the
    variance scale in the DC mode. Supports a binary mask.

    Parameters
    ----------
    images : ArrayLike
        Noise-dominated data, shape (R, H, W) or (H, W).
    mask : ArrayLike, optional
        Binary mask (H, W). If None, all ones.

    Returns
    -------
    psd : np.ndarray
        PSD (H, W), DC at [0, 0], scaled to the masked variance.
    acf : np.ndarray
        The intermediate ACF used to derive the PSD.
    """
    acf, var_hat = estimate_noise_acf_masked(images, mask=mask)
    psd = acf_to_psd(acf, var_target=var_hat)
    return psd, acf


def inverse_psd(psd: ArrayLike, mode: SafetyMode = "ridge", eps: float = 1e-4) -> np.ndarray:
    """
    Safe inverse PSD given the PSD. Uses either a ridge (mode="ridge") or pseudoinverse
    equivalent (mode="pinv") to guard against numerical instability. "ridge" is recommended
    for use in inference, either sampling or non-linear optimisation.

    Parameters
    ----------
    psd : ArrayLike
        PSD (H, W), DC at [0, 0].
    mode: SafetyMode
        Either "ridge" to use a small ridge (Tikhonov regularisation), or "pinv" for a pseudoinverse
        equivalent (relative eigenvalue floor) to ensure numerical stability.
    eps: float
        Numerical tolerance. Either ridge size, or relative tolerance for pinv.

    Returns
    -------
    inv_psd : np.ndarray
        Numerically safe inverse PSD.
    """
    psd = np.asarray(psd, dtype=float).real
    if mode == "ridge":
        return 1.0 / (psd + eps)
    elif mode == "pinv":
        tol = eps * np.max(psd)
        return np.where(psd > tol, 1.0 / psd, 0.0)


def estimate_noise_inv_psd_from_data(
    images: ArrayLike,
    mask: Optional[ArrayLike] = None,
    mode: SafetyMode = "ridge",
    eps: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the inverse power spectral density (PSD) for stationary noise. Keeps the
    variance scale in the DC mode. Supports a binary mask. Uses either a ridge (mode="ridge")
    or pseudoinverse equivalent (mode="pinv") to guard against numerical instability.
    "ridge" is recommended for use in inference, either sampling or non-linear optimisation.

    Parameters
    ----------
    images : ArrayLike
        Noise-dominated data, shape (R, H, W) or (H, W).
    mask : ArrayLike, optional
        Binary mask (H, W). If None, all ones.
    mode: SafetyMode
        Either "ridge" to use a small ridge (Tikhonov regularisation), or "pinv" for a pseudoinverse
        equivalent (relative eigenvalue floor) to ensure numerical stability.
    eps: float
        Numerical tolerance. Either ridge size, or relative tolerance for pinv.

    Returns
    -------
    inv_psd : np.ndarray
        Numerically safe inverse PSD.
    """
    return inverse_psd(psd=estimate_noise_psd_from_data(images, mask)[0], mode=mode, eps=eps)
