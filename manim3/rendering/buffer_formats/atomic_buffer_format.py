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
        n_col = shape_dict.get(0, 1)
        n_row = shape_dict.get(1, 1)
        if layout == BufferLayout.STD140:
            n_col_pseudo = n_col if not shape and n_row == 1 else 4
            n_col_alignment = n_col if not shape and n_col <= 2 and n_row == 1 else 4
        else:
            n_col_pseudo = n_col
            n_col_alignment = n_col

        super().__init__(
            name=name,
            shape=shape
        )
        self._base_char_ = base_char
        self._base_itemsize_ = base_itemsize
        self._base_ndim_ = len(base_shape)
        self._n_row_ = n_row
        self._n_col_ = n_col
        self._n_col_pseudo_ = n_col_pseudo
        self._base_alignment_ = n_col_alignment * base_itemsize

    @Lazy.variable_hashable
    @classmethod
    def _base_char_(cls) -> str:
        return ""

    @Lazy.variable_hashable
    @classmethod
    def _base_itemsize_(cls) -> int:
        return 0

    @Lazy.variable_hashable
    @classmethod
    def _base_ndim_(cls) -> int:
        return 0

    @Lazy.variable_hashable
    @classmethod
    def _n_row_(cls) -> int:
        return 0

    @Lazy.variable_hashable
    @classmethod
    def _n_col_(cls) -> int:
        return 0

    @Lazy.variable_hashable
    @classmethod
    def _n_col_pseudo_(cls) -> int:
        return 0

    @Lazy.variable_hashable
    @classmethod
    def _base_alignment_(cls) -> int:
        return 0

    @Lazy.property_hashable
    @classmethod
    def _row_itemsize_(
        cls,
        n_col_pseudo: int,
        base_itemsize: int
    ) -> int:
        return n_col_pseudo * base_itemsize

    @Lazy.property_hashable
    @classmethod
    def _itemsize_(
        cls,
        n_row: int,
        row_itemsize: int
    ) -> int:
        return n_row * row_itemsize

    @Lazy.property_hashable
    @classmethod
    def _dtype_(
        cls,
        base_char: str,
        base_itemsize: int,
        n_col: int,
        row_itemsize: int,
        n_row: int
    ) -> np.dtype:
        return np.dtype((np.dtype({
            "names": ["_"],
            "formats": [(np.dtype(f"{base_char}{base_itemsize}"), (n_col,))],
            "itemsize": row_itemsize
        }), (n_row,)))
