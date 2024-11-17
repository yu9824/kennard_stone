import importlib.util


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
