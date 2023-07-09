from contextlib import contextmanager
from typing import Iterator

import moderngl
import numpy as np

from .buffer import Buffer


class ReadOnlyBuffer(Buffer):
    __slots__ = ()

    @contextmanager
    def temporary_buffer(self) -> Iterator[moderngl.Buffer]:
        buffer = self._fetch_buffer()
        buffer.orphan(self._buffer_format_._nbytes_)
        yield buffer
        self._finalize_buffer(buffer)

    def read(
        self,
        buffer: moderngl.Buffer
    ) -> dict[str, np.ndarray]:
        return self._read_from_buffer(
            buffer=buffer,
            np_buffer=self._np_buffer_,
            np_buffer_pointers=self._np_buffer_pointers_
        )
