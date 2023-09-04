import numpy as np

from ...lazy.lazy import Lazy
from ..buffer_formats.buffer_layout import BufferLayout
from .write_only_buffer import WriteOnlyBuffer


class AttributesBuffer(WriteOnlyBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        fields: list[str],
        num_vertex: int,
        array_lens: dict[str, int] | None = None,
        data: dict[str, np.ndarray]
    ) -> None:
        # Passing structs to an attribute is not allowed, so we eliminate the parameter `child_structs`.
        if array_lens is None:
            array_lens = {}
        super().__init__(
            field="__VertexStruct__ __vertex__[__NUM_VERTEX__]",
            child_structs={
                "__VertexStruct__": fields
            },
            array_lens={
                "__NUM_VERTEX__": num_vertex,
                **array_lens
            }
        )
        self.write(data)

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _layout_() -> BufferLayout:
        # Let's keep using std140 layout, hopefully giving a faster processing speed.
        return BufferLayout.STD140
