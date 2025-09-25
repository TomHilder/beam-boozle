"""JAX backend for beam_boozle likelihoods. This is opt-in, with JAX as an extra dependency."""

try:
    from .likelihoods_impl import fast_log_likelihood, slow_log_likelihood
except Exception as e:
    raise ImportError(
        "beam_boozle.jax requires JAX. Install extras: `pip install beam-boozle[jax]`"
    ) from e

__all__ = ["slow_log_likelihood", "fast_log_likelihood"]
