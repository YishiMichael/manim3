from contextlib import contextmanager
from typing import Iterator

import moderngl
import numpy as np

from ...toplevel.toplevel import Toplevel
from .buffer import Buffer


class ReadOnlyBuffer(Buffer):
    __slots__ = ()

    @contextmanager
    def buffer(self) -> Iterator[moderngl.Buffer]:
        yield Toplevel.context.buffer(reserve=self._buffer_format_._nbytes_)
        #yield buffer
        #self._finalize_buffer(buffer)

    def read(
        self,
        buffer: moderngl.Buffer
    ) -> dict[str, np.ndarray]:
        return type(self)._read_from_bytes(
            data_bytes=buffer.read(),
            np_buffer=self._np_buffer_,
            np_buffer_pointers=self._np_buffer_pointers_
        )
