from __future__ import annotations


from typing import Self

import moderngl
import numpy as np

from ...lazy.lazy import Lazy
from ...toplevel.toplevel import Toplevel
from ..buffer_formats.buffer_format import BufferFormat
from .buffer import Buffer


class WriteOnlyBuffer(Buffer):
    __slots__ = ()

    @Lazy.variable()
    @staticmethod
    def _data_dict_() -> dict[str, np.ndarray]:
        return {}

    @Lazy.property()
    @staticmethod
    def _buffer_(
        data_dict: dict[str, np.ndarray],
        buffer_format: BufferFormat
    ) -> moderngl.Buffer:
        return Toplevel.context.buffer(data=buffer_format._write(data_dict))

    def write(
        self: Self,
        data_dict: dict[str, np.ndarray]
    ) -> None:
        self._data_dict_ = data_dict
