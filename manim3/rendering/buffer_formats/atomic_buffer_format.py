from typing import ClassVar

import numpy as np

from ...lazy.lazy import Lazy
from .buffer_format import BufferFormat
from .buffer_layout import BufferLayout


class AtomicBufferFormat(BufferFormat):
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

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        gl_dtype_str: str,
        layout: BufferLayout
    ) -> None:
        base_char, base_itemsize, base_shape = type(self)._GL_DTYPES[gl_dtype_str]
        assert len(base_shape) <= 2 and all(2 <= l <= 4 for l in base_shape)
        shape_dict = dict(enumerate(base_shape))
        col_len = shape_dict.get(0, 1)
        row_len = shape_dict.get(1, 1)
        if layout == BufferLayout.STD140:
            col_padding = 0 if not shape and row_len == 1 else 4 - col_len
            base_alignment = (col_len if not shape and col_len <= 2 and row_len == 1 else 4) * base_itemsize
        else:
            col_padding = 0
            base_alignment = 1

        super().__init__(
            name=name,
            shape=shape
        )
        self._base_char_ = base_char
        self._base_itemsize_ = base_itemsize
        self._base_ndim_ = len(base_shape)
        self._row_len_ = row_len
        self._col_len_ = col_len
        self._col_padding_ = col_padding
        self._itemsize_ = row_len * (col_len + col_padding) * base_itemsize
        self._base_alignment_ = base_alignment

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _base_char_() -> str:
        return ""

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _base_itemsize_() -> int:
        return 0

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _base_ndim_() -> int:
        return 0

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _row_len_() -> int:
        return 0

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _col_len_() -> int:
        return 0

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _col_padding_() -> int:
        return 0

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _dtype_(
        base_char: str,
        base_itemsize: int,
        col_len: int,
        col_padding: int,
        row_len: int
    ) -> np.dtype:
        return np.dtype((np.dtype({
            "names": ["_"],
            "formats": [(np.dtype(f"{base_char}{base_itemsize}"), (col_len,))],
            "itemsize": (col_len + col_padding) * base_itemsize
        }), (row_len,)))

    @Lazy.property_collection(hasher=Lazy.naive_hasher)
    @staticmethod
    def _pointers_(
        base_ndim: int
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        return (((), base_ndim),)

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _format_str_(
        col_len: int,
        base_char: str,
        base_itemsize: int,
        col_padding: int,
        row_len: int,
        size: int
    ) -> str:
        row_components = [f"{col_len}{base_char}{base_itemsize}"]
        if col_padding:
            row_components.append(f"{col_padding}x{base_itemsize}")
        return " ".join(row_components * (row_len * size))
