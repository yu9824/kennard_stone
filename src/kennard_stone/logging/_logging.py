from __future__ import annotations

import importlib.util
import os
import re
import sys
from logging import (
    INFO,
    NOTSET,
    Formatter,
    Handler,
    Logger,
    StreamHandler,
    getLogger,
)
from types import TracebackType
from typing import TypeVar

HANDLER = TypeVar("HANDLER", bound=Handler)


def _color_supported() -> bool:
    """Return whether colored log output is supported."""
    if not importlib.util.find_spec("colorlog"):
        return False

    # NO_COLOR environment variable:
    if os.environ.get("NO_COLOR", None):
        return False

    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


_default_handler: StreamHandler | None = None
"""Default handler of the library root logger (None until configured)."""


def _get_root_logger_name() -> str:
    """Get the root logger name (the library name).

    Returns
    -------
    str
        root logger name (library name)
    """
    return __name__.split(".")[0]


def create_default_formatter() -> Formatter:
    """Create the default formatter.

    Returns
    -------
    Formatter
        default formatter
    """
    if _color_supported():
        from colorlog import ColoredFormatter

        return ColoredFormatter(
            "%(asctime)s - %(name)s:%(lineno)d%(log_color)s[%(levelname)s]%(reset)s - %(message)s"
        )
    else:
        return Formatter(
            "%(asctime)s - %(name)s:%(lineno)d[%(levelname)s] - %(message)s"
        )


default_formatter: Formatter = create_default_formatter()
"""Default formatter."""


def get_handler(
    handler: HANDLER, formatter: Formatter | None = None, level=NOTSET
) -> HANDLER:
    """Configure a handler with a formatter and a level.

    Parameters
    ----------
    handler : logging.Handler
        Handler to configure.
    formatter : logging.Formatter, default=None
        Formatter to set. If None, the default formatter is used.
    level : int, default=logging.NOTSET
        Level to set.

    Returns
    -------
    logging.Handler
        The given handler, configured.
    """
    handler.setLevel(level)
    handler.setFormatter(
        formatter if formatter else create_default_formatter()
    )
    return handler


def _create_default_handler() -> StreamHandler:
    return get_handler(StreamHandler())


def _configure_library_root_logger() -> None:
    global _default_handler

    if _default_handler:
        # This library has already configured the library root logger.
        return

    _default_handler = _create_default_handler()

    # Apply our default configuration to the library root logger.
    library_root_logger = get_root_logger()
    library_root_logger.addHandler(_default_handler)
    library_root_logger.setLevel(INFO)
    library_root_logger.propagate = False


def get_root_logger() -> Logger:
    """Get the root logger of this library.

    Returns
    -------
    Logger
        library root logger
    """
    _configure_library_root_logger()

    return getLogger(_get_root_logger_name())


def get_child_logger(name: str, propagate: bool = True) -> Logger:
    """Get a child logger of the library root logger.

    Parameters
    ----------
    name : str
        Name of the module, usually ``__name__``. It must be a submodule of
        kennard_stone or "__main__".
    propagate : bool, default=True
        Whether to propagate records to the handlers of the parent logger.

    Returns
    -------
    logging.Logger
        Child logger.

    Raises
    ------
    ValueError
        If ``name`` is neither a submodule of kennard_stone nor "__main__".
    """
    root_logger = get_root_logger()

    _result_logger = re.match(rf"{_get_root_logger_name()}\.(.+)", name)
    if _result_logger:
        child_logger = root_logger.getChild(_result_logger.group(1))
    elif name == "__main__":
        child_logger = root_logger.getChild(name)
    else:
        raise ValueError("You should use '__name__'.")

    child_logger.propagate = propagate
    return child_logger


def enable_default_handler() -> None:
    """Enable the default handler."""
    _configure_library_root_logger()

    assert _default_handler is not None
    get_root_logger().addHandler(_default_handler)


def disable_default_handler() -> None:
    """Disable the default handler."""
    _configure_library_root_logger()

    assert _default_handler is not None
    get_root_logger().removeHandler(_default_handler)


class catch_default_handler:
    """Disable the default handler temporarily, as a context manager.

    Examples
    --------
    >>> _logger = get_child_logger(__name__)
    >>> with catch_default_handler():
    ...     _logger.info("not logged")
    >>> _logger.info("logged")
    """

    def __enter__(self) -> None:
        disable_default_handler()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        enable_default_handler()
