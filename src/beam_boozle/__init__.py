# Default public API uses NumPy to avoid optional deps
from .numpy_backend import (
    fast_log_likelihood,
    slow_log_likelihood,
)

__all__ = [
    "slow_log_likelihood",
    "fast_log_likelihood",
]
