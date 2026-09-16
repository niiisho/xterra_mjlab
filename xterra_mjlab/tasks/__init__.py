"""xTerra mjlab tasks. Importing this package registers the Gym-style tasks."""

from .velocity import config  # noqa: F401  (triggers task registration)
from .wheelie import config