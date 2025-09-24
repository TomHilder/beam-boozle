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


def estimate_noise_covariance_fft(noise_images):
    """Estimate noise power spectrum from multiple realizations."""
    # noise_images shape: (n_realizations, N, N)
    noise_fft = jnp.fft.fft2(noise_images)
    power_spectrum = jnp.mean(jnp.abs(noise_fft) ** 2, axis=0)
    return power_spectrum  # DC at [0,0]


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
power_spectrum_est = estimate_noise_covariance_fft(jnp.asarray(noise_images))
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
