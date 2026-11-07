"""Compatibility helpers across scikit-learn versions."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from inspect import signature
from typing import Any


@cache
def _accepted_parameters(func: Callable[..., Any]) -> frozenset[str]:
    return frozenset(signature(func).parameters)


def finite_check_kwargs(
    func: Callable[..., Any], value: bool | str
) -> dict[str, bool | str]:
    """Build the keyword argument controlling the finite-value check.

    scikit-learn 1.6 renamed ``force_all_finite`` to ``ensure_all_finite``
    and 1.8 removed the old name. Pass the result as ``**kwargs`` to
    ``check_array`` or ``pairwise_distances``.

    Parameters
    ----------
    func : callable
        scikit-learn function accepting either parameter name.

    value : bool or "allow-nan"
        Value for the finite-value check.

    Returns
    -------
    dict
        ``{"ensure_all_finite": value}`` or ``{"force_all_finite": value}``,
        whichever ``func`` accepts (empty if neither).
    """
    params = _accepted_parameters(func)
    if "ensure_all_finite" in params:
        return {"ensure_all_finite": value}
    if "force_all_finite" in params:
        return {"force_all_finite": value}
    return {}
