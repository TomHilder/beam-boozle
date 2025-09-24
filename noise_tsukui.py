import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit, random
from numpy.random import default_rng

# Set up JAX
jax.config.update("jax_enable_x64", True)

rng = default_rng(seed=0)

plt.style.use("mpl_drip.custom")
# Turn off latex rendering for faster plotting, set font to default
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "sans-serif"


def generate_correlated_noise_psf(N, correlation_length=3.0):
    """Generate a kernel for correlated noise (not just Gaussian)."""
    x = jnp.linspace(-N // 2, N // 2, N)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2)

    # Combination of Gaussian core and exponential tail
    gaussian_part = jnp.exp(-(R**2) / (2 * correlation_length**2))
    exp_part = jnp.exp(-R / (correlation_length * 3))
    psf = 0.7 * gaussian_part + 0.5 * exp_part

    # Add some asymmetry NOTE: turned off
    psf = psf * (1 + 0.0 * jnp.sin(2 * jnp.arctan2(Y, X + 1e-10)))

    # Normalize to unit sum
    psf = psf / jnp.sum(psf)
    return psf


def generate_noise_image(N, psf):
    """Generate correlated noise image by convolving white noise with PSF."""
    white_noise = rng.normal(size=(N, N))
    Hk = jnp.fft.fft2(jnp.fft.ifftshift(psf))
    noise_fft = jnp.fft.fft2(white_noise) * Hk
    noise = jnp.real(jnp.fft.ifft2(noise_fft))
    return noise


def true_noise_acf_psd_from_psf(psf, white_var=1.0, demean=False, eps=1e-12):
    """
    Given a real PSF/kernel `psf` (as from generate_correlated_noise_psf),
    return the *true* noise autocovariance (ACF) and power spectrum (PSD)
    for a field n = psf * w, with white noise variance = white_var.

    Conventions (matches your estimator):
      - Unnormalized FFTs (jnp.fft.fft2 / ifft2)
      - DC at [0,0]
      - PSD scaled so (1/N^2) * sum_k S(k) = Var(n)   (Parseval)
      - If demean=True, we zero the DC bin in S and then rescale to keep the total
        power equal to Var(n) * N^2 (this mimics per-image mean removal notch).

    Returns:
      acf_true : (N,N) real ACF
      psd_true : (N,N) nonnegative PSD (real), DC at [0,0]
      var_true : scalar variance of n in real space
    """
    psf = jnp.asarray(psf)
    N = psf.shape[-1]

    # True variance after convolution with psf (circular, stationary):
    # Var[n] = white_var * sum_x psf[x]^2
    var_true = white_var * jnp.sum(psf**2)

    # Shape of the spectrum from the kernel
    H = jnp.fft.fft2(jnp.fft.ifftshift(psf))  # complex transfer function
    S_shape = jnp.abs(H) ** 2  # |H|^2 (unscaled shape)

    # Scale PSD so that (1/N^2) * sum S = var_true  (Parseval consistency)
    scale = (var_true * (N * N)) / (jnp.sum(S_shape) + eps)
    S = S_shape * scale

    if demean:
        # mimic per-image mean removal (DC notch), then renormalize total power
        S = S.at[0, 0].set(0.0)
        S = S * ((var_true * (N * N)) / (jnp.sum(S) + eps))

    # ACF is IFFT of PSD
    C = jnp.real(jnp.fft.ifft2(S))

    # numerical hygiene
    S = jnp.maximum(S, 0.0)

    return C, S, var_true


def _ensure_stack(x):
    return x if x.ndim == 3 else x[jnp.newaxis, ...]


def estimate_noise_acf_masked(images, mask=None, demean=True, eps=1e-12):
    """
    Mask-aware, bias-corrected autocovariance estimator (ACF).
    images: (R, N, N) or (N, N) array of noise-dominated data
    mask:   (N, N) binary (1 noise-only, 0 exclude). If None, all ones.
    Returns:
        acf: (N, N), circular lags; DC at [0,0]
        var_hat: masked variance used to set the ACF scale (float)
    """
    X = _ensure_stack(images)
    R, N, _ = X.shape
    if mask is None:
        mask = jnp.ones((N, N), dtype=X.dtype)

    # masked mean per image (optional)
    if demean:
        wsum = jnp.sum(mask)
        mu = jnp.sum(X * mask, axis=(-1, -2)) / (wsum + eps)  # (R,)
        Xc = X - mu[:, None, None]
    else:
        Xc = X

    Y = Xc * mask  # apply mask

    # FFTs
    A = jnp.fft.fft2(Y, axes=(-2, -1))  # (R,N,N)
    M = jnp.fft.fft2(mask, axes=(-2, -1))  # (N,N)

    # Numerator: average periodogram in image space (via IFFT of |A|^2)
    num = jnp.fft.ifft2(jnp.mean(A * jnp.conj(A), axis=0), axes=(-2, -1)).real  # (N,N)

    # Denominator: number of valid pairs per lag (via IFFT of |M|^2)
    den = jnp.fft.ifft2(M * jnp.conj(M), axes=(-2, -1)).real + eps  # (N,N)

    acf = num / den  # unbiased (mask-corrected) autocovariance

    # masked variance for scaling sanity (use unbiased sample variance)
    # note: with large masks the simple second moment is fine; this is robust
    var_hat = jnp.sum((Y) ** 2) / (jnp.sum(mask) * R - (1 if demean else 0))

    # Optional: set zero-lag exactly to var_hat (helps numerical consistency)
    # acf = acf.at[0, 0].set(var_hat)

    return acf, var_hat


def acf_to_psd(acf, var_target=None):
    """
    Power spectrum from ACF (Wiener–Khinchin). DC at [0,0].
    If var_target is provided, enforce Parseval:
        (1/N^2) * sum_k S(k) = var_target
    """
    N = acf.shape[-1]
    S = jnp.fft.fft2(acf, axes=(-2, -1)).real  # small negatives may appear numerically
    if var_target is not None:
        scale = (var_target * (N * N)) / (jnp.sum(S) + 1e-12)
        S = S * scale
    # clip tiny negatives due to numerical roundoff
    S = jnp.maximum(S, 0.0)
    return S


def estimate_noise_psd_from_data(images, mask=None, demean=False):
    """
    Convenience wrapper: directly get PSD from data (ACF-first).
    Replaces your original estimate_noise_covariance_fft.
    Returns:
        psd: (N, N) with DC at [0,0]
        acf: (N, N) used internally (can ignore if not needed)
    """
    acf, var_hat = estimate_noise_acf_masked(images, mask=mask, demean=demean)
    psd = acf_to_psd(acf, var_target=var_hat)  # correct amplitude for likelihoods
    return psd, acf


N = 512
N_noise_images = 9


psf = generate_correlated_noise_psf(N, correlation_length=3.5)

noise_images = np.array([generate_noise_image(N, psf) for _ in range(N_noise_images)])


fig, ax = plt.subplots(3, 3, layout="compressed", dpi=100, figsize=[9, 9])
for i, ax_ in enumerate(ax.flatten()):
    ax_.imshow(noise_images[i, :, :])
    ax_.set_xticks([])
    ax_.set_yticks([])
plt.show()

power_spectrum_est = estimate_noise_psd_from_data(jnp.asarray(noise_images))[0]
Npix_side = N  # 2D image is N x N

# ---- Truth from your PSF ----
H = jnp.fft.fft2(jnp.fft.ifftshift(psf))  # complex transfer function
A_true = jnp.abs(H)  # |H|
S_shape = jnp.abs(H) ** 2  # |H|^2

# ---- Empirical from noise ----
S_emp = power_spectrum_est  # E[|FFT(noise)|^2]
A_emp = jnp.sqrt(S_emp) / Npix_side  # amplitude: sqrt(S)/N  (unnormalized FFT)

# ---- Put both on the same scale (match DC bin) ----
# DC is at [0,0] without shifts
dc_true = A_true[0, 0]
dc_emp = A_emp[0, 0]
A_emp_scaled = A_emp * (dc_true / (dc_emp + 1e-30))

# ---- Real-space covariances (autocorrelation) ----
C_true = jnp.real(jnp.fft.ifft2(S_shape))  # h ⋆ h (up to convention)
C_emp = jnp.real(jnp.fft.ifft2(S_emp))  # empirical covariance

# Normalize by zero-lag (variance) so shapes are comparable
var_true = C_true[0, 0]
var_emp = C_emp[0, 0]
C_true_n = C_true / (var_true + 1e-30)
C_emp_n = C_emp / (var_emp + 1e-30)

# ---- Plot ----
fig, ax = plt.subplots(2, 3, layout="compressed", dpi=100, figsize=[18, 9])

title_kwargs = dict(fontsize=12)

ax[0, 0].set_title("|H| (True, DC-matched)", **title_kwargs)
ax[0, 0].imshow(jnp.fft.fftshift(A_true), cmap="viridis")

ax[0, 1].set_title("sqrt(S_emp)/N (Measured, DC-matched)", **title_kwargs)
ax[0, 1].imshow(jnp.fft.fftshift(A_emp_scaled), cmap="viridis")

ax[0, 2].set_title("Difference (Measured - True)", **title_kwargs)
im = ax[0, 2].imshow(
    jnp.fft.fftshift(A_emp_scaled - A_true), cmap="RdBu", vmin=-0.1, vmax=0.1
)

ax[1, 0].set_title("IFFT(|H|^2) (True cov, norm @ 0-lag)", **title_kwargs)
ax[1, 0].imshow(jnp.fft.fftshift(C_true_n), cmap="viridis")

ax[1, 1].set_title("IFFT(S_emp) (Measured cov, norm @ 0-lag)", **title_kwargs)
ax[1, 1].imshow(jnp.fft.fftshift(C_emp_n), cmap="viridis")

ax[1, 2].set_title("Difference (Measured - True)", **title_kwargs)
im = ax[1, 2].imshow(
    jnp.fft.fftshift(C_emp_n - C_true_n), cmap="RdBu", vmin=-0.1, vmax=0.1
)

for axi in ax.flatten():
    axi.set_xticks([])
    axi.set_yticks([])

plt.show()
