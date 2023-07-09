import numpy as np

from ...lazy.lazy import Lazy
from ..buffer_formats.buffer_layout import BufferLayout
from .write_only_buffer import WriteOnlyBuffer


class UniformBlockBuffer(WriteOnlyBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        fields: list[str],
        child_structs: dict[str, list[str]] | None = None,
        array_lens: dict[str, int] | None = None,
        data: dict[str, np.ndarray]
    ) -> None:
        if child_structs is None:
            child_structs = {}
        super().__init__(
            field=f"__UniformBlockStruct__ {name}",
            child_structs={
                "__UniformBlockStruct__": fields,
                **child_structs
            },
            array_lens=array_lens
        )
        self.write(data)

    @Lazy.property_hashable
    @classmethod
    def _layout_(cls) -> BufferLayout:
        return BufferLayout.STD140