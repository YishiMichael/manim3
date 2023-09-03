import moderngl
import numpy as np

from ...lazy.lazy import Lazy
from ...toplevel.toplevel import Toplevel
from .buffer import Buffer


class WriteOnlyBuffer(Buffer):
    __slots__ = ()

    @Lazy.variable_external
    @classmethod
    def _data_dict_(cls) -> dict[str, np.ndarray]:
        return {}

    @Lazy.property_external
    @classmethod
    def _buffer_(
        cls,
        np_buffer: np.ndarray,
        np_buffer_pointers: dict[str, tuple[np.ndarray, int]],
        data_dict: dict[str, np.ndarray]
    ) -> moderngl.Buffer:
        return Toplevel.context.buffer(data=cls._write_to_bytes(
            data_dict=data_dict,
            np_buffer=np_buffer,
            np_buffer_pointers=np_buffer_pointers
        ))

    #@_buffer_.finalizer
    #@classmethod
    #def _buffer_finalizer(
    #    cls,
    #    buffer: moderngl.Buffer
    #) -> None:
    #    cls._finalize_buffer(buffer)

    def write(
        self,
        data_dict: dict[str, np.ndarray]
    ) -> None:
        self._data_dict_ = data_dict
