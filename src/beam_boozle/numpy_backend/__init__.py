"""NumPy backend for beam_boozle likelihoods. This is the default backend."""

from .likelihoods_impl import fast_log_likelihood, slow_log_likelihood

__all__ = ["slow_log_likelihood", "fast_log_likelihood"]
