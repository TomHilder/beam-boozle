import arviz as az
import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
from noise_tsukui_2 import generate_correlated_noise_psf, generate_noise_image
from numpy.random import default_rng
from numpyro import distributions as dist
from numpyro import infer

import beam_boozle.jax_backend as bb
import beam_boozle.utils as bu

jax.config.update("jax_enable_x64", True)
rng = default_rng(seed=0)
plt.style.use("mpl_drip.custom")
numpyro.set_host_device_count(2)

CORR_LENGTH = 1.5
N_IMAGE = 256
N_NOISE_INSTANCES = 5


A_MODEL = 4.0
X0_MODEL = N_IMAGE / 2
Y0_MODEL = N_IMAGE / 2
SIGMA_X_MODEL = N_IMAGE / 12
SIGMA_Y_MODEL = N_IMAGE / 6
ROTATION_MODEL = jnp.pi / 4.2


def generate_data_grid(N):
    x = jnp.linspace(0, N - 1, N)
    y = jnp.linspace(0, N - 1, N)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


def generate_signal_model(X, Y, amplitude, x0, y0, sigma_x, sigma_y, rotation):
    cos_r = jnp.cos(rotation)
    sin_r = jnp.sin(rotation)
    Xc = X - x0
    Yc = Y - y0
    X_rot = cos_r * Xc - sin_r * Yc
    Y_rot = sin_r * Xc + cos_r * Yc
    model = amplitude * jnp.exp(-0.5 * (X_rot**2 / sigma_x**2 + Y_rot**2 / sigma_y**2))
    return model


if __name__ == "__main__":
    X_true, Y_true = generate_data_grid(N_IMAGE)
    signal_true = generate_signal_model(
        X_true,
        Y_true,
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

    psd_inv = bu.estimate_noise_inv_psd_from_data(noise_images)
    rms_noise = jnp.std(noise_images)
    # cov_true = full_cov_from_acf(acf_true)
    # cov_est = full_cov_from_acf(acf_est)

    # Plot and compare the covariance matrices directly
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
    # a.set_xticks([])
    # a.set_yticks([])
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

    # Plot for checking the precision matrix didn't go crazy
    # vmax_prec = jnp.max(jnp.abs(cov_inv_true))
    # imshow_kwargs = dict(cmap="RdBu", vmin=-vmax_prec, vmax=vmax_prec)
    # plt.figure(figsize=(5, 5), layout="compressed", dpi=100)
    # plt.title("Precision Matrix (Direct inversion)")
    # plt.imshow(cov_inv_true, **imshow_kwargs)
    # plt.show()

    # Define NumPyro models
    def model_uncorr(x, y, data_vector, rms):
        # Model parameters
        a = numpyro.sample("a", dist.LogUniform(low=1e-2, high=1e2))
        x0 = numpyro.sample("x0", dist.Uniform(low=0, high=N_IMAGE))
        y0 = numpyro.sample("y0", dist.Uniform(low=0, high=N_IMAGE))
        sigma_x = numpyro.sample("sigma_x", dist.LogUniform(low=1e-2, high=1e2))
        sigma_y = numpyro.sample("sigma_y", dist.LogUniform(low=1e-2, high=1e2))
        angle = numpyro.sample("angle", dist.Uniform(low=0, high=jnp.pi / 2))
        # Generate model image
        model_image = generate_signal_model(
            X=x,
            Y=y,
            amplitude=a,
            x0=x0,
            y0=y0,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rotation=angle,
        )
        model_vector = model_image.flatten()
        # Data sampling statement
        numpyro.sample("data", dist.Normal(model_vector, rms), obs=data_vector)

    def model_corr(x, y, data_image, power_spectrum_inv):
        # Model parameters
        a = numpyro.sample("a", dist.LogUniform(low=1e-2, high=1e2))
        x0 = numpyro.sample("x0", dist.Uniform(low=0, high=N_IMAGE))
        y0 = numpyro.sample("y0", dist.Uniform(low=0, high=N_IMAGE))
        sigma_x = numpyro.sample("sigma_x", dist.LogUniform(low=1e-2, high=1e2))
        sigma_y = numpyro.sample("sigma_y", dist.LogUniform(low=1e-2, high=1e2))
        angle = numpyro.sample("angle", dist.Uniform(low=0, high=jnp.pi / 2))
        # Generate model image
        model_image = generate_signal_model(
            X=x,
            Y=y,
            amplitude=a,
            x0=x0,
            y0=y0,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            rotation=angle,
        )
        # Data sampling statement
        total_ll = bb.fast_log_likelihood(
            data_image,
            model_image,
            power_spectrum_inv,
        )
        numpyro.factor("custom_ll", total_ll)

    image_idx = 0  # which image to run inference on
    selected_image = data_images[image_idx]

    sampler_uncorr = infer.MCMC(
        infer.NUTS(model_uncorr),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )
    sampler_uncorr.run(
        jax.random.PRNGKey(0),
        X_true,
        Y_true,
        data_vector=selected_image.flatten(),
        rms=rms_noise,
    )

    sampler_corr = infer.MCMC(
        infer.NUTS(model_corr),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )
    sampler_corr.run(
        jax.random.PRNGKey(0),
        X_true,
        Y_true,
        data_image=selected_image,
        power_spectrum_inv=psd_inv,
    )

    inf_data_uncorr = az.from_numpyro(sampler_uncorr)
    inf_data_corr = az.from_numpyro(sampler_corr)
    print(az.summary(inf_data_uncorr))
    print(az.summary(inf_data_corr))

    true_params = [
        A_MODEL,
        X0_MODEL,
        Y0_MODEL,
        SIGMA_X_MODEL,
        SIGMA_Y_MODEL,
        ROTATION_MODEL,
    ]

    fig = plt.figure(figsize=[12, 12])
    corner.corner(
        inf_data_uncorr,
        var_names=["a", "x0", "y0", "sigma_x", "sigma_y", "angle"],
        truths=true_params,
        truth_color="grey",
        fig=fig,
        smooth=0.8,
        color="C0",
        alpha=0.5,
    )
    corner.corner(
        inf_data_corr,
        var_names=["a", "x0", "y0", "sigma_x", "sigma_y", "angle"],
        fig=fig,
        smooth=0.8,
        color="C1",
        alpha=0.5,
    )
    plt.show()
