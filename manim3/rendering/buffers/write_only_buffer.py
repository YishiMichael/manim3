#from __future__ import annotations


#import numpy as np

#from ...lazy.lazy import Lazy
#from ..buffer_formats.buffer_format import BufferFormat
#from .buffer import Buffer


#class WriteOnlyBuffer(Buffer):
#    __slots__ = ()

#    @Lazy.variable()
#    @staticmethod
#    def _data_dict_() -> dict[str, np.ndarray]:
#        return {}

#    @Lazy.property()
#    @staticmethod
#    def _data_bytes_(
#        buffer_format: BufferFormat,
#        shape: tuple[int, ...],
#        data_dict: dict[str, np.ndarray]
#    ) -> bytes:
#        return buffer_format.write(shape, data_dict)

#    #@Lazy.property()
#    #@staticmethod
#    #def _buffer_(
#    #    data_dict: dict[str, np.ndarray],
#    #    buffer_format: BufferFormat
#    #) -> moderngl.Buffer:
#    #    print(buffer := Toplevel.context.buffer(data=buffer_format._write(data_dict)), buffer.glo)
#    #    return buffer

#    #def write(
#    #    self: Self,
#    #    data_dict: dict[str, np.ndarray]
#    #) -> None:
#    #    self._data_dict_ = data_dict
