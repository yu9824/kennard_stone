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
from typing import Optional, TypeVar

HANDLER = TypeVar("HANDLER", bound=Handler)


def _color_supported() -> bool:
    """Detection of color support."""
    if not importlib.util.find_spec("colorlog"):
        return False

    # NO_COLOR environment variable:
    if os.environ.get("NO_COLOR", None):
        return False

    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    else:
        return True


_default_handler: Optional[StreamHandler] = None
"""default root logger handler

if not configured, None
"""


def _get_root_logger_name() -> str:
    """get root logger name (library name)

    Returns
    -------
    str
        root logger name (library name)
    """
    return __name__.split(".")[0]


def create_default_formatter() -> Formatter:
    """create default formatter

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
"""default formatter"""


def get_handler(
    handler: HANDLER, formatter: Optional[Formatter] = None, level=NOTSET
) -> HANDLER:
    """configure handler in an easy api

    Parameters
    ----------
    handler : HANDLER

    formatter : Optional[Formatter], optional
        , by default None
    level : _type_, optional
        , by default NOTSET

    Returns
    -------
    HANDLER

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
    """get library root logger of this package

    Returns
    -------
    Logger
        library root logger
    """
    _configure_library_root_logger()

    return getLogger(_get_root_logger_name())


def get_child_logger(name: str, propagate: bool = True) -> Logger:
    """get logger

    Parameters
    ----------
    name : str
        You shold assign '__name__'

    propagate : bool
        propagate to parent handler or not, by default True

    Returns
    -------
    Logger
        child logger

    Raises
    ------
    ValueError

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
    """enable default handler"""
    _configure_library_root_logger()

    assert _default_handler is not None
    get_root_logger().addHandler(_default_handler)


def disable_default_handler() -> None:
    """disable default handler"""
    _configure_library_root_logger()

    assert _default_handler is not None
    get_root_logger().removeHandler(_default_handler)


class catch_default_handler:
    """catch default handler

    Example
    -------
    >>> _logger = get_child_logger(__name__)
    >>> with catch_default_handler():
    >>>    _logger.info("not log")
    >>> _logger.info("log")

    """

    def __enter__(self) -> None:
        disable_default_handler()

    def __exit__(
        self,
        exc_type: "Optional[type[Exception]]",
        exc_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> None:
        enable_default_handler()
