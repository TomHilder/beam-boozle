import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.linalg import pinvh

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
N_IMAGE = 42
N_NOISE_INSTANCES = 100


def full_cov_from_acf(acf):
    """Build full (N^2 x N^2) covariance from ACF, periodic."""
    N = acf.shape[0]
    ii = jnp.arange(N)
    I, J = jnp.meshgrid(ii, ii, indexing="ij")  # (N,N)
    coords = jnp.stack([I.ravel(), J.ravel()], axis=1)  # (N^2, 2)
    di = (coords[:, 0, None] - coords[None, :, 0]) % N  # (N^2, N^2)
    dj = (coords[:, 1, None] - coords[None, :, 1]) % N  # (N^2, N^2)
    cov = acf[di, dj]  # (N^2, N^2)
    return cov


def dense_precision_from_psd(psd, *, mode="ridge", eps=1e-8):
    """
    Build dense inverse covariance (precision) from PSD (DC at [0,0]).
    mode:
      - "ridge": inv_spec = 1/(psd + eps)  -> strict PD
      - "pinv" : inv_spec = 1/psd where psd > tol, else 0 (handles DC-notched PSD)
    """
    S = jnp.asarray(psd).real
    S = jnp.maximum(S, 0.0)  # project tiny negatives away

    if mode == "ridge":
        inv_spec = 1.0 / (S + eps)
    elif mode == "pinv":
        tol = eps * jnp.max(S)
        inv_spec = jnp.where(S > tol, 1.0 / S, 0.0)
    else:
        raise ValueError("mode must be 'ridge' or 'pinv'")

    acf_prec = jnp.fft.ifft2(inv_spec).real  # precision "kernel"
    return full_cov_from_acf(acf_prec)  # reuse your builder


if __name__ == "__main__":
    psf = generate_correlated_noise_psf(N_IMAGE, correlation_length=CORR_LENGTH)
    noise_images = jnp.array(
        [generate_noise_image(N_IMAGE, psf) for _ in range(N_NOISE_INSTANCES)]
    )
    psd_est, acf_est = estimate_noise_psd_from_data(jnp.asarray(noise_images))
    acf_true, psd_true = true_noise_acf_psd_from_psf(psf, demean=False)[:2]

    cov_true = full_cov_from_acf(acf_true)
    cov_est = full_cov_from_acf(acf_est)

    # Plot and compare the ACFs directly
    vmax_acf = jnp.max(jnp.abs(acf_true))
    imshow_kwargs = dict(cmap="RdBu", vmin=-vmax_acf, vmax=vmax_acf)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="compressed")
    im0 = ax[0].imshow(jnp.fft.fftshift(acf_true), **imshow_kwargs)
    ax[0].set_title("True ACF")
    im1 = ax[1].imshow(jnp.fft.fftshift(acf_est), **imshow_kwargs)
    ax[1].set_title("Estimated ACF")
    im2 = ax[2].imshow(jnp.fft.fftshift(acf_true - acf_est), **imshow_kwargs)
    ax[2].set_title("True - Estimated")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()

    # Plot and compare the covariance matrices directly
    vmax_cov = jnp.max(jnp.abs(cov_true))
    imshow_kwargs = dict(cmap="RdBu", vmin=-vmax_cov, vmax=vmax_cov)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="compressed")
    im0 = ax[0].imshow(cov_true, **imshow_kwargs)
    ax[0].set_title("True Covariance")
    im1 = ax[1].imshow(cov_est, **imshow_kwargs)
    ax[1].set_title("Estimated Covariance")
    im2 = ax[2].imshow(cov_true - cov_est, **imshow_kwargs)
    ax[2].set_title("True - Estimated")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()

    direct_inv = False if N_IMAGE > 64 else True
    # Plot and compare the precision from direct inversion and PSD-based (forget estimated)
    eps = 1e-8

    # Numerical accuracy mode
    mode = "pinv"
    # mode = "ridge" # will require larger eps

    if direct_inv:
        if mode == "ridge":
            cov_inv_true = jnp.linalg.pinv(cov_true + eps * jnp.eye(cov_true.shape[0]))
        elif mode == "pinv":
            cov_inv_true = pinvh(cov_true, rtol=eps)

    cov_inv_psd = dense_precision_from_psd(psd_true, mode=mode, eps=eps)

    # Plot
    vmax_prec = jnp.max(jnp.abs(cov_inv_true))
    imshow_kwargs = dict(cmap="RdBu", vmin=-vmax_prec, vmax=vmax_prec)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="compressed")
    ax[0].set_title("Precision (Direct inversion)")
    ax[1].set_title("Precision (PSD-based)")
    ax[2].set_title("Direct - PSD-based")
    im1 = ax[1].imshow(cov_inv_psd, **imshow_kwargs)
    if direct_inv:
        im0 = ax[0].imshow(cov_inv_true, **imshow_kwargs)
        im2 = ax[2].imshow(cov_inv_true - cov_inv_psd, **imshow_kwargs)
    else:
        im0 = ax[0].text(
            0.5,
            0.5,
            "N too large for direct inversion",
            ha="center",
            va="center",
            transform=ax[0].transAxes,
        )
        im2 = ax[2].text(
            0.5,
            0.5,
            "N too large for direct inversion",
            ha="center",
            va="center",
            transform=ax[2].transAxes,
        )
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()

    # cov_inv_true = jnp.linalg.pinv(cov_true + 1e-3 * jnp.eye(cov_true.shape[0]))
    # cov_inv_est = jnp.linalg.pinv(cov_est + 1e-3 * jnp.eye(cov_est.shape[0]))
    # vmax_prec = jnp.max(jnp.abs(cov_inv_true))
    # imshow_kwargs = dict(cmap="RdBu", vmin=-vmax_prec, vmax=vmax_prec)
    # fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="compressed")
    # im0 = ax[0].imshow(cov_inv_true, **imshow_kwargs)
    # ax[0].set_title("Precision (direct inversion)")
    # im1 = ax[1].imshow(cov_inv_est, **imshow_kwargs)
    # ax[1].set_title("Precision (from true PSD)")
    # im2 = ax[2].imshow(cov_inv_true - cov_inv_est, **imshow_kwargs)
    # ax[2].set_title("Direct - PSD-based")
    # for a in ax:
    #     a.set_xticks([])
    #     a.set_yticks([])
    # plt.show()
