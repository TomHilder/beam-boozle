# Beam-boozle

Another project with Fourier-accelerated Gaussian processes. This time, we solve the issue that it's hard to include correlations between pixels in Bayesian inference for radio interferometric images. This is because the matrix product R @ Cinv @ R.T is crazy expensive. Here, we (will) provide:

- A function to measure the noise structure from pure noise segments (empty channels)
- A super-fast _exact_ (to machine precision) likelihood that NEVER builds a matrix, and will never dominate your run time unless your images are absolutely huge
- Both NumPy and JAX backends. The gradients are also exact to machine precision.

The goal is to provide a drop-in solution for anyone who is fitting to synthesised images, such that they do not have to do basically any work, but can account for correlations. The companion paper will also explore why this is important, and **increasingly important** as model complexity increases (for example deep-learning and other non-parametric models).

## TODO (package stuff)

- [x] Fully switch to installable package + examples format
- [ ] Finish functionality:
  - [x] Likelihoods with both backends
  - [x] Noise segments -> inverse PSD estimator for users
  - [ ] Window function / padding type thing to kill periodic boundary issues (minor anyway)
  - [ ] Noise generation? Maybe a good check to check the ACF estimate is ok?
- [ ] Unit tests
- [ ] Clean up dependencies including Python versions to make it far easier to install for most people
- [ ] Automated PyPI
- [ ] Automated testing
- [ ] Simple Examples in README
- [ ] Tutorial notebooks for other cases/considerations
