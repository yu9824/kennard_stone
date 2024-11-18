import importlib.util

from ..logging import get_child_logger

_logger = get_child_logger(__name__)


class IgnoredArgumentWarning(Warning):
    """Warning used to ignore an argument."""

    ...


def is_installed(package_name: str) -> bool:
    """Check if the package is installed.

    Parameters
    ----------
    package_name : str
        package name like `sklearn`

    Returns
    -------
    bool
        if installed, True
    """
    return bool(importlib.util.find_spec(package_name))


def is_available_gpu() -> bool:
    if is_installed("torch"):
        import torch  # type: ignore

        _logger.debug(f"torch.cuda.is_available()={torch.cuda.is_available()}")
        _logger.debug(
            f"torch.backends.mps.is_available()={torch.backends.mps.is_available()}"
        )

        return torch.cuda.is_available() or torch.backends.mps.is_available()
    else:
        return False
