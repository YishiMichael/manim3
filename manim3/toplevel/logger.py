from __future__ import annotations


from types import TracebackType
from typing import Self

import rich.console

from .toplevel import Toplevel


class Logger:
    __slots__ = ("_console",)

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        self._console: rich.console.Console = rich.console.Console()

    def __enter__(
        self: Self
    ) -> None:
        Toplevel._logger = self

    def __exit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None
    ) -> None:
        Toplevel._logger = None
