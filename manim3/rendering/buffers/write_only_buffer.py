import moderngl
import numpy as np

from ...lazy.lazy import Lazy
from .buffer import Buffer


class WriteOnlyBuffer(Buffer):
    __slots__ = ()

    @Lazy.property_external
    @classmethod
    def _buffer_(
        cls,
        np_buffer: np.ndarray,
        np_buffer_pointers: dict[str, tuple[np.ndarray, int]],
        data_dict: dict[str, np.ndarray]
    ) -> moderngl.Buffer:
        buffer = cls._fetch_buffer()
        cls._write_to_buffer(
            buffer=buffer,
            np_buffer=np_buffer,
            np_buffer_pointers=np_buffer_pointers,
            data_dict=data_dict
        )
        return buffer

    @_buffer_.finalizer
    @classmethod
    def _buffer_finalizer(
        cls,
        buffer: moderngl.Buffer
    ) -> None:
        cls._finalize_buffer(buffer)

    @Lazy.variable_external
    @classmethod
    def _data_dict_(cls) -> dict[str, np.ndarray]:
        return {}

    def write(
        self,
        data_dict: dict[str, np.ndarray]
    ) -> None:
        self._data_dict_ = data_dict

    def get_buffer(self) -> moderngl.Buffer:
        return self._buffer_
