from __future__ import annotations


from contextlib import contextmanager
from typing import (
    Iterator,
    Self
)

import moderngl
import numpy as np

from ...toplevel.toplevel import Toplevel
from .buffer import Buffer


class ReadOnlyBuffer(Buffer):
    __slots__ = ()

    @contextmanager
    def buffer(
        self: Self
    ) -> Iterator[moderngl.Buffer]:
        yield Toplevel.context.buffer(reserve=self._buffer_format_._nbytes_)

    def read(
        self: Self,
        buffer: moderngl.Buffer
    ) -> dict[str, np.ndarray]:
        return self._buffer_format_._read(buffer.read())
