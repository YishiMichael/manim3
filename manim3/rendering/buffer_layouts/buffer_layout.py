from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import (
    ClassVar,
    Never,
    Self
)

from ..buffer_formats.atomic_buffer_format import AtomicBufferFormat
from ..buffer_formats.buffer_format import BufferFormat
from ..buffer_formats.structured_buffer_format import StructuredBufferFormat


class BufferLayout(ABC):
    __slots__ = ()

    _GL_DTYPES: ClassVar[dict[str, tuple[str, int, tuple[int, ...]]]] = {
        "int":     ("i", 4, ()),
        "ivec2":   ("i", 4, (2,)),
        "ivec3":   ("i", 4, (3,)),
        "ivec4":   ("i", 4, (4,)),
        "uint":    ("u", 4, ()),
        "uvec2":   ("u", 4, (2,)),
        "uvec3":   ("u", 4, (3,)),
        "uvec4":   ("u", 4, (4,)),
        "float":   ("f", 4, ()),
        "vec2":    ("f", 4, (2,)),
        "vec3":    ("f", 4, (3,)),
        "vec4":    ("f", 4, (4,)),
        "mat2":    ("f", 4, (2, 2)),
        "mat2x3":  ("f", 4, (2, 3)),  # TODO: check order
        "mat2x4":  ("f", 4, (2, 4)),
        "mat3x2":  ("f", 4, (3, 2)),
        "mat3":    ("f", 4, (3, 3)),
        "mat3x4":  ("f", 4, (3, 4)),
        "mat4x2":  ("f", 4, (4, 2)),
        "mat4x3":  ("f", 4, (4, 3)),
        "mat4":    ("f", 4, (4, 4)),
        "double":  ("f", 8, ()),
        "dvec2":   ("f", 8, (2,)),
        "dvec3":   ("f", 8, (3,)),
        "dvec4":   ("f", 8, (4,)),
        "dmat2":   ("f", 8, (2, 2)),
        "dmat2x3": ("f", 8, (2, 3)),
        "dmat2x4": ("f", 8, (2, 4)),
        "dmat3x2": ("f", 8, (3, 2)),
        "dmat3":   ("f", 8, (3, 3)),
        "dmat3x4": ("f", 8, (3, 4)),
        "dmat4x2": ("f", 8, (4, 2)),
        "dmat4x3": ("f", 8, (4, 3)),
        "dmat4":   ("f", 8, (4, 4))
    }

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def get_atomic_buffer_format(
        cls: type[Self],
        name: str,
        shape: tuple[int, ...],
        gl_dtype_str: str
    ) -> AtomicBufferFormat:
        base_char, base_itemsize, base_shape = cls._GL_DTYPES[gl_dtype_str]
        assert len(base_shape) <= 2 and all(2 <= l <= 4 for l in base_shape)
        shape_dict = dict(enumerate(base_shape))
        col_len = shape_dict.get(0, 1)
        row_len = shape_dict.get(1, 1)
        col_padding = cls._get_atomic_col_padding(shape, col_len, row_len)
        base_alignment = cls._get_atomic_base_alignment(shape, col_len, row_len, base_itemsize)
        return AtomicBufferFormat(
            name=name,
            shape=shape,
            base_char=base_char,
            base_itemsize=base_itemsize,
            base_ndim=len(base_shape),
            row_len=row_len,
            col_len=col_len,
            col_padding=col_padding,
            itemsize=row_len * (col_len + col_padding) * base_itemsize,
            base_alignment=base_alignment
        )

    @classmethod
    def get_structured_buffer_format(
        cls: type[Self],
        name: str,
        shape: tuple[int, ...],
        children: tuple[BufferFormat, ...],
    ) -> StructuredBufferFormat:
        base_alignment = cls._get_structured_base_alignment()
        offsets: list[int] = []
        offset: int = 0
        for child in children:
            offset += (-offset) % child._base_alignment_
            offsets.append(offset)
            offset += child._nbytes_
        offset += (-offset) % base_alignment
        return StructuredBufferFormat(
            name=name,
            shape=shape,
            children=children,
            offsets=tuple(offsets),
            itemsize=offset,
            base_alignment=base_alignment
        )

    @classmethod
    @abstractmethod
    def _get_atomic_col_padding(
        cls: type[Self],
        shape: tuple[int, ...],
        col_len: int,
        row_len: int
    ) -> int:
        pass

    @classmethod
    @abstractmethod
    def _get_atomic_base_alignment(
        cls: type[Self],
        shape: tuple[int, ...],
        col_len: int,
        row_len: int,
        base_itemsize: int
    ) -> int:
        pass

    @classmethod
    @abstractmethod
    def _get_structured_base_alignment(
        cls: type[Self]
    ) -> int:
        pass
