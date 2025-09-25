import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit
from numpy.random import default_rng
from scipy.linalg import pinvh

from materialise_cov import full_cov_from_acf
from noise_tsukui import (
    estimate_noise_psd_from_data,
    generate_correlated_noise_psf,
    generate_noise_image,
    true_noise_acf_psd_from_psf,
)

jax.config.update("jax_enable_x64", True)
rng = default_rng(seed=0)
plt.style.use("mpl_drip.custom")

CORR_LENGTH = 0.5
N_IMAGE = 10
N_NOISE_INSTANCES = 400


A_MODEL = 7.0
X0_MODEL = N_IMAGE / 2
Y0_MODEL = N_IMAGE / 2
SIGMA_X_MODEL = N_IMAGE / 12
SIGMA_Y_MODEL = N_IMAGE / 6
ROTATION_MODEL = jnp.pi / 4.2


def generate_signal_model(N, amplitude, x0, y0, sigma_x, sigma_y, rotation):
    """Generate a 2D Gaussian signal model."""
    x = jnp.linspace(0, N - 1, N)
    y = jnp.linspace(0, N - 1, N)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Rotate coordinates
    cos_r = jnp.cos(rotation)
    sin_r = jnp.sin(rotation)
    Xc = X - x0
    Yc = Y - y0
    X_rot = cos_r * Xc - sin_r * Yc
    Y_rot = sin_r * Xc + cos_r * Yc

    model = amplitude * jnp.exp(-0.5 * (X_rot**2 / sigma_x**2 + Y_rot**2 / sigma_y**2))
    return model


@jit
def slow_log_likelihood(data, model, cov_inv):
    """Slow likelihood using full covariance matrix (validation only)."""
    residual = (data - model).reshape(-1)
    return -0.5 * residual @ (cov_inv @ residual)


@jit
def fast_log_likelihood(data, model, power_spectrum_inv):
    """
    Fast likelihood using FFT and diagonal covariance in Fourier domain.
    Assumes periodic boundary conditions and stationarity.
    """
    residual = data - model
    Rk = jnp.fft.fft2(residual)
    N2 = residual.size  # N*N
    # Quadratic form in Fourier basis; divide by N^2 for unnormalized FFT
    qform = jnp.real(jnp.sum(jnp.conj(Rk) * (Rk * power_spectrum_inv))) / N2
    return -0.5 * qform


if __name__ == "__main__":
    signal_true = generate_signal_model(
        N_IMAGE,
        A_MODEL,
        X0_MODEL,
        Y0_MODEL,
        SIGMA_X_MODEL,
        SIGMA_Y_MODEL,
        ROTATION_MODEL,
    )
    psf = generate_correlated_noise_psf(N_IMAGE, correlation_length=CORR_LENGTH)
    noise_images = jnp.array(
        [generate_noise_image(N_IMAGE, psf) for _ in range(N_NOISE_INSTANCES)]
    )
    data_images = signal_true + noise_images

    psd_est, acf_est = estimate_noise_psd_from_data(jnp.asarray(noise_images))
    acf_true, psd_true = true_noise_acf_psd_from_psf(psf, demean=False)[:2]
    cov_true = full_cov_from_acf(acf_true)
    cov_est = full_cov_from_acf(acf_est)

    # # Plot and compare the covariance matrices directly
    # vmax_cov = jnp.max(jnp.abs(cov_true))
    # imshow_kwargs = dict(cmap="RdBu", vmin=-vmax_cov, vmax=vmax_cov)
    # fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="compressed")
    # im0 = ax[0].imshow(cov_true, **imshow_kwargs)
    # ax[0].set_title("True Covariance")
    # im1 = ax[1].imshow(cov_est, **imshow_kwargs)
    # ax[1].set_title("Estimated Covariance")
    # im2 = ax[2].imshow(cov_true - cov_est, **imshow_kwargs)
    # ax[2].set_title("True - Estimated")
    # for a in ax:
    #     a.set_xticks([])
    #     a.set_yticks([])
    # plt.show()

    plot_vmax = 2.0 + jnp.max(jnp.abs(signal_true))
    fig, ax = plt.subplots(3, 3, figsize=(8, 8), layout="compressed")
    fig.suptitle("Example data images (signal + correlated noise)")
    for i, ax in enumerate(ax.flat):
        if i == 0:
            ax.text(
                0.5,
                0.9,
                "True Signal Model",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.imshow(signal_true, cmap="RdBu", vmin=-plot_vmax, vmax=plot_vmax)
        else:
            ax.imshow(data_images[i], cmap="RdBu", vmin=-plot_vmax, vmax=plot_vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

    # Numerical accuracy
    eps = 1e-3
    # mode = "pinv"
    mode = "ridge"  # will require larger eps

    if mode == "ridge":
        cov_inv_true = jnp.linalg.pinv(cov_true + eps * jnp.eye(cov_true.shape[0]))
        power_spectrum_inv = 1.0 / (psd_true + eps)
    elif mode == "pinv":
        cov_inv_true = pinvh(cov_true, rtol=eps)
        power_spectrum_inv = jnp.where(psd_true > eps, 1.0 / psd_true, 0.0)

    # Plot for checking the precision matrix didn't go crazy
    vmax_prec = jnp.max(jnp.abs(cov_inv_true))
    imshow_kwargs = dict(cmap="RdBu", vmin=-vmax_prec, vmax=vmax_prec)
    plt.figure(figsize=(5, 5), layout="compressed", dpi=100)
    plt.title("Precision Matrix (Direct inversion)")
    plt.imshow(cov_inv_true, **imshow_kwargs)
    plt.show()

    # === Check log-likelihood discrepancy across all synthetic images

    # First get signal that is not quite the true signal
    close_sigma = 0.1
    signal_close = generate_signal_model(
        N_IMAGE,
        A_MODEL * rng.normal(1.0, close_sigma),
        X0_MODEL * rng.normal(1.0, close_sigma),
        Y0_MODEL * rng.normal(1.0, close_sigma),
        SIGMA_X_MODEL * rng.normal(1.0, close_sigma),
        SIGMA_Y_MODEL * rng.normal(1.0, close_sigma),
        ROTATION_MODEL * rng.normal(1.0, close_sigma),
    )
    signal_far = generate_signal_model(
        N_IMAGE,
        A_MODEL * rng.normal(1.0, 3 * close_sigma),
        X0_MODEL * rng.normal(1.0, 3 * close_sigma),
        Y0_MODEL * rng.normal(1.0, 3 * close_sigma),
        SIGMA_X_MODEL * rng.normal(1.0, 3 * close_sigma),
        SIGMA_Y_MODEL * rng.normal(1.0, 3 * close_sigma),
        ROTATION_MODEL * rng.normal(1.0, 3 * close_sigma),
    )

    # Check log-likelihood values
    ll_slow_close = jnp.array(
        [slow_log_likelihood(data, signal_close, cov_inv_true) for data in data_images]
    )
    ll_slow_far = jnp.array(
        [slow_log_likelihood(data, signal_far, cov_inv_true) for data in data_images]
    )
    ll_fast_close = jnp.array(
        [
            fast_log_likelihood(data, signal_close, power_spectrum_inv)
            for data in data_images
        ]
    )
    ll_fast_far = jnp.array(
        [
            fast_log_likelihood(data, signal_far, power_spectrum_inv)
            for data in data_images
        ]
    )
    ll_diff_close = ll_slow_close - ll_fast_close
    ll_diff_far = ll_slow_far - ll_fast_far
    relative_error_close = jnp.abs(ll_diff_close) / (jnp.abs(ll_slow_close))
    relative_error_far = jnp.abs(ll_diff_far) / (jnp.abs(ll_slow_far))

    # Plot error distribution
    hist_kwargs = dict(bins=10, histtype="step", lw=2, density=True)
    plt.figure(figsize=(6, 4), layout="compressed", dpi=100)
    plt.hist(relative_error_close, label="Close", **hist_kwargs)
    plt.hist(relative_error_far, label="Far", **hist_kwargs)
    plt.xlabel("Relative Error (Log Likelihood)")
    # plt.ylabel("Frequency")
    plt.yticks([])
    plt.title("Log-Likelihood Relative Error Distribution")
    plt.legend()
    plt.show()

    # Check gradients are similar
    grad_slow = jax.grad(slow_log_likelihood, argnums=1)
    grad_fast = jax.grad(fast_log_likelihood, argnums=1)
    grad_slow_vals_close = jnp.array(
        [
            grad_slow(data, signal_close, cov_inv_true).reshape(-1)
            for data in data_images
        ]
    )
    grad_slow_vals_far = jnp.array(
        [grad_slow(data, signal_far, cov_inv_true).reshape(-1) for data in data_images]
    )
    grad_fast_vals_close = jnp.array(
        [
            grad_fast(data, signal_close, power_spectrum_inv).reshape(-1)
            for data in data_images
        ]
    )
    grad_fast_vals_far = jnp.array(
        [
            grad_fast(data, signal_far, power_spectrum_inv).reshape(-1)
            for data in data_images
        ]
    )
    grad_diff_close = grad_slow_vals_close - grad_fast_vals_close
    grad_diff_far = grad_slow_vals_far - grad_fast_vals_far

    def rel_vec_err(v1, v2):
        return jnp.linalg.norm(v1 - v2, axis=-1) / jnp.linalg.norm(v1, axis=-1)

    grad_relative_error_close = rel_vec_err(grad_slow_vals_close, grad_fast_vals_close)
    grad_relative_error_far = rel_vec_err(grad_slow_vals_far, grad_fast_vals_far)

    hist_kwargs = dict(bins=10, histtype="step", lw=2, density=True)
    plt.figure(figsize=(6, 4), layout="compressed", dpi=100)
    plt.hist(grad_relative_error_close, **hist_kwargs, label="Close")
    plt.hist(grad_relative_error_far, **hist_kwargs, label="Far")
    plt.xlabel("Relative Error (Gradient)")
    # plt.ylabel("Frequency")
    plt.yticks([])
    plt.title("Gradient Relative Error Distribution")
    plt.legend()
    plt.show()
