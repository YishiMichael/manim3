#from __future__ import annotations


#import numpy as np

#from ...lazy.lazy import Lazy
#from ..buffer_formats.buffer_format import BufferFormat
#from .buffer import Buffer


#class ReadOnlyBuffer(Buffer):
#    __slots__ = ()

#    @Lazy.variable()
#    @staticmethod
#    def _data_bytes_() -> bytes:
#        return b""

#    @Lazy.property()
#    @staticmethod
#    def _data_dict_(
#        buffer_format: BufferFormat,
#        shape: tuple[int, ...],
#        data_bytes: bytes
#    ) -> dict[str, np.ndarray]:
#        return buffer_format.read(shape, data_bytes)

    #@contextmanager
    #def buffer(
    #    self: Self
    #) -> Iterator[moderngl.Buffer]:
    #    yield Toplevel.context.buffer(reserve=self._buffer_format_._nbytes_)

    #def read(
    #    self: Self,
    #    buffer: moderngl.Buffer
    #) -> dict[str, np.ndarray]:
    #    return self._buffer_format_._read(buffer.read())
