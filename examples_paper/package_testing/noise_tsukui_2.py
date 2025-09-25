import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

# import beam_boozle.jax_backend as bb
import beam_boozle as bb
import beam_boozle.utils as bu

jax.config.update("jax_enable_x64", True)
rng = default_rng(seed=0)
plt.style.use("mpl_drip.custom")


def generate_correlated_noise_psf(N, correlation_length=3.0):
    x = jnp.linspace(-N // 2, N // 2, N)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2)
    gaussian_part = jnp.exp(-(R**2) / (2 * correlation_length**2))
    exp_part = jnp.exp(-R / (correlation_length * 3))
    psf = 0.7 * gaussian_part + 0.5 * exp_part
    psf = psf * (1 + 0.3 * jnp.sin(2 * jnp.arctan2(Y, X + 1e-10)))
    psf = psf / jnp.sqrt(jnp.sum(psf**2))
    return psf


def generate_noise_image(N, psf):
    white_noise = rng.normal(size=(N, N))
    Hk = jnp.fft.fft2(jnp.fft.ifftshift(psf))
    noise_fft = jnp.fft.fft2(white_noise) * Hk
    noise = jnp.real(jnp.fft.ifft2(noise_fft))
    return noise


def true_noise_acf_psd_from_psf(psf, white_var=1.0, demean=False, eps=1e-12):
    psf = jnp.asarray(psf)
    N = psf.shape[-1]
    var_true = white_var * jnp.sum(psf**2)
    H = jnp.fft.fft2(jnp.fft.ifftshift(psf))
    S_shape = jnp.abs(H) ** 2
    scale = (var_true * (N * N)) / (jnp.sum(S_shape) + eps)
    S = S_shape * scale
    if demean:
        S = S.at[0, 0].set(0.0)
        S = S * ((var_true * (N * N)) / (jnp.sum(S) + eps))
    C = jnp.real(jnp.fft.ifft2(S))
    S = jnp.maximum(S, 0.0)
    return C, S, var_true


if __name__ == "__main__":
    N = 46
    N_noise_images = 500

    psf = generate_correlated_noise_psf(N, correlation_length=2)
    noise_images = np.array([generate_noise_image(N, psf) for _ in range(N_noise_images)])
    psd_est, acf_est = bu.psd.estimate_noise_psd_from_data(jnp.asarray(noise_images))
    acf_true, psd_true = true_noise_acf_psd_from_psf(psf, demean=False)[:2]

    SHRINK_PLOTS = 0.6

    fig, ax = plt.subplots(1, 1, figsize=(SHRINK_PLOTS * 8, SHRINK_PLOTS * 8))
    ax.set_title("Example correlated noise instance")
    ax.imshow(noise_images[0], cmap="RdBu", vmin=-3, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    fig, ax = plt.subplots(
        2, 3, figsize=(SHRINK_PLOTS * 12, SHRINK_PLOTS * 8), layout="compressed"
    )

    psd_max = jnp.max(psd_true)
    acf_max = jnp.max(acf_true)

    imshow_psd_kwargs = dict(vmin=-psd_max, vmax=psd_max, cmap="RdBu")
    imshow_acf_kwargs = dict(vmin=-acf_max, vmax=acf_max, cmap="RdBu")

    ax[0, 0].imshow(jnp.fft.fftshift(psd_true), **imshow_psd_kwargs)
    ax[0, 1].imshow(jnp.fft.fftshift(psd_est), **imshow_psd_kwargs)
    ax[0, 2].imshow(jnp.fft.fftshift(jnp.abs(psd_true - psd_est)), **imshow_psd_kwargs)
    ax[1, 0].imshow(jnp.fft.fftshift(acf_true), **imshow_acf_kwargs)
    ax[1, 1].imshow(jnp.fft.fftshift(acf_est), **imshow_acf_kwargs)
    ax[1, 2].imshow(jnp.fft.fftshift(jnp.abs(acf_true - acf_est)), **imshow_acf_kwargs)

    ax[0, 0].set_title("True")
    ax[0, 1].set_title("Estimated")
    ax[0, 2].set_title("Absolute Difference")

    ax[0, 0].set_ylabel("PSD")
    ax[1, 0].set_ylabel("ACF")

    for a in ax.flatten():
        a.set_xticks([])
        a.set_yticks([])

    plt.show()
