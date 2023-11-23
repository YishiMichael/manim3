from __future__ import annotations


import functools
import operator
import re
from typing import (
    ClassVar,
    Self
)

import numpy as np

from ..constants.custom_typing import ShapeType
from ..lazy.lazy import Lazy
from ..lazy.lazy_object import LazyObject


class Field(LazyObject):
    __slots__ = ()

    # Note, numpy matrices are row-major, while OpenGL store matrices in column-major format.
    # One may need to transpose matrices before passing them to shaders.
    # In numpy, shape `(r, c)` stands for `r` rows and `c` columns.
    # In glsl, `matcxr` specifies a matrix with `r` rows and `c` columns.
    _GLSL_DTYPES: ClassVar[dict[str, np.dtype]] = {
        "int":     np.dtype("i4"),
        "ivec2":   np.dtype("2i4"),
        "ivec3":   np.dtype("3i4"),
        "ivec4":   np.dtype("4i4"),
        "uint":    np.dtype("u4"),
        "uvec2":   np.dtype("2u4"),
        "uvec3":   np.dtype("3u4"),
        "uvec4":   np.dtype("4u4"),
        "float":   np.dtype("f4"),
        "vec2":    np.dtype("2f4"),
        "vec3":    np.dtype("3f4"),
        "vec4":    np.dtype("4f4"),
        "mat2":    np.dtype("(2,2)f4"),
        "mat2x3":  np.dtype("(2,3)f4"),
        "mat2x4":  np.dtype("(2,4)f4"),
        "mat3x2":  np.dtype("(3,2)f4"),
        "mat3":    np.dtype("(3,3)f4"),
        "mat3x4":  np.dtype("(3,4)f4"),
        "mat4x2":  np.dtype("(4,2)f4"),
        "mat4x3":  np.dtype("(4,3)f4"),
        "mat4":    np.dtype("(4,4)f4"),
        "double":  np.dtype("f8"),
        "dvec2":   np.dtype("2f8"),
        "dvec3":   np.dtype("3f8"),
        "dvec4":   np.dtype("4f8"),
        "dmat2":   np.dtype("(2,2)f8"),
        "dmat2x3": np.dtype("(2,3)f8"),
        "dmat2x4": np.dtype("(2,4)f8"),
        "dmat3x2": np.dtype("(3,2)f8"),
        "dmat3":   np.dtype("(3,3)f8"),
        "dmat3x4": np.dtype("(3,4)f8"),
        "dmat4x2": np.dtype("(4,2)f8"),
        "dmat4x3": np.dtype("(4,3)f8"),
        "dmat4":   np.dtype("(4,4)f8")
    }

    def __init__(
        self: Self,
        *,
        name: str,
        shape: ShapeType,
        itemsize: int,
        base_alignment: int
    ) -> None:
        super().__init__()
        self._name_ = name
        self._shape_ = shape
        self._itemsize_ = itemsize
        self._base_alignment_ = base_alignment

    @Lazy.variable()
    @staticmethod
    def _name_() -> str:
        return ""

    @Lazy.variable()
    @staticmethod
    def _shape_() -> ShapeType:
        return ()

    @Lazy.variable()
    @staticmethod
    def _itemsize_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _base_alignment_() -> int:
        return 1

    @Lazy.property()
    @staticmethod
    def _size_(
        shape: ShapeType
    ) -> int:
        return functools.reduce(operator.mul, shape, 1)

    @Lazy.property()
    @staticmethod
    def _dtype_() -> np.dtype:
        return np.dtype("f4")

    @Lazy.property(plural=True)
    @staticmethod
    def _pointers_() -> tuple[tuple[tuple[str, ...], int], ...]:
        return ()

    @classmethod
    def _parse_field_declaration(
        cls: type[Self],
        field_declaration: str,
        array_lens_dict: dict[str, int]
    ) -> tuple[str, str, ShapeType]:
        pattern = re.compile(r"""
            (?P<dtype_str>\w+?)
            \s
            (?P<name>\w+?)
            (?P<shape>(\[\w+?\])*)
        """, flags=re.VERBOSE)
        match = pattern.fullmatch(field_declaration)
        assert match is not None

        dtype_str = match.group("dtype_str")
        name = match.group("name")
        shape = tuple(
            int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else array_lens_dict[s]
            for index_match in re.finditer(r"\[(\w+?)\]", match.group("shape"))
        )
        return dtype_str, name, shape

    @classmethod
    def parse_atomic_field(
        cls: type[Self],
        field_declaration: str,
        array_lens_dict: dict[str, int]
    ) -> AtomicField:
        dtype_str, name, shape = cls._parse_field_declaration(field_declaration, array_lens_dict)
        return AtomicField(
            name=name,
            shape=shape,
            dtype=cls._GLSL_DTYPES[dtype_str]
        )

    @classmethod
    def parse_field(
        cls: type[Self],
        field_declaration: str,
        struct_dict: dict[str, tuple[str, ...]],
        array_lens_dict: dict[str, int]
    ) -> Field:
        dtype_str, name, shape = cls._parse_field_declaration(field_declaration, array_lens_dict)
        if (dtype := cls._GLSL_DTYPES.get(dtype_str)) is not None:
            return AtomicField(
                name=name,
                shape=shape,
                dtype=dtype
            )
        return StructuredField(
            name=name,
            shape=shape,
            fields=tuple(
                cls.parse_field(
                    field_declaration=child_field_declaration,
                    struct_dict=struct_dict,
                    array_lens_dict=array_lens_dict
                )
                for child_field_declaration in struct_dict[dtype_str]
            )
        )

    def _get_np_buffer_and_pointers(
        self: Self,
        shape: ShapeType
    ) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, int]]]:

        def get_np_buffer_pointer(
            np_buffer: np.ndarray,
            name_chain: tuple[str, ...]
        ) -> np.ndarray:
            result = np_buffer
            for name in name_chain:
                result = result[name]
            return result["_"]

        np_buffer = np.zeros(shape, dtype=self._dtype_)
        np_buffer_pointers = {
            ".".join(name_chain): (get_np_buffer_pointer(np_buffer, name_chain), base_ndim)
            for name_chain, base_ndim in self._pointers_
        }
        return np_buffer, np_buffer_pointers

    def write(
        self: Self,
        shape: ShapeType,
        data_dict: dict[str, np.ndarray]
    ) -> bytes:
        np_buffer, np_buffer_pointers = self._get_np_buffer_and_pointers(shape)
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data = data_dict[key]
            if not np_buffer_pointer.size:
                assert not data.size
                continue
            data_expanded = np.expand_dims(data, axis=tuple(range(-2, -base_ndim)))
            assert np_buffer_pointer.shape == data_expanded.shape
            np_buffer_pointer[...] = data_expanded
        return np_buffer.tobytes()

    def read(
        self: Self,
        shape: ShapeType,
        data_bytes: bytes
    ) -> dict[str, np.ndarray]:
        data_dict: dict[str, np.ndarray] = {}
        np_buffer, np_buffer_pointers = self._get_np_buffer_and_pointers(shape)
        np_buffer[...] = np.frombuffer(data_bytes, dtype=np_buffer.dtype).reshape(np_buffer.shape)
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data_expanded = np_buffer_pointer[...]
            data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))
            data_dict[key] = data
        return data_dict


class AtomicField(Field):
    __slots__ = ()

    def __init__(
        self: Self,
        name: str,
        shape: ShapeType,
        dtype: np.dtype
    ) -> None:
        base_shape = dtype.shape
        base_char = dtype.base.char
        base_itemsize = dtype.base.itemsize
        assert len(base_shape) <= 2 and all(2 <= l <= 4 for l in base_shape)
        col_len, row_len, *_ = base_shape + (1, 1)
        col_padding = 0 if not shape and row_len == 1 else 4 - col_len
        base_alignment = (col_len if not shape and col_len <= 2 and row_len == 1 else 4) * base_itemsize

        super().__init__(
            name=name,
            shape=shape,
            itemsize=row_len * (col_len + col_padding) * base_itemsize,
            base_alignment=base_alignment
        )
        self._base_char_ = base_char
        self._base_itemsize_ = base_itemsize
        self._base_ndim_ = len(base_shape)
        self._row_len_ = row_len
        self._col_len_ = col_len
        self._col_padding_ = col_padding

    @Lazy.variable()
    @staticmethod
    def _base_char_() -> str:
        return ""

    @Lazy.variable()
    @staticmethod
    def _base_itemsize_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _base_ndim_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _row_len_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _col_len_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _col_padding_() -> int:
        return 0

    @Lazy.property()
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

    @Lazy.property(plural=True)
    @staticmethod
    def _pointers_(
        base_ndim: int
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        return (((), base_ndim),)


class StructuredField(Field):
    __slots__ = ()

    def __init__(
        self: Self,
        name: str,
        shape: ShapeType,
        fields: tuple[Field, ...]
    ) -> None:
        struct_base_alignment = 16
        next_base_alignments = (*(field._base_alignment_ for field in fields), struct_base_alignment)[1:]
        offsets: list[int] = []
        paddings: list[int] = []
        offset = 0
        for field, next_base_alignment in zip(fields, next_base_alignments, strict=True):
            offsets.append(offset)
            offset += field._itemsize_ * field._size_
            padding = (-offset) % next_base_alignment
            paddings.append(padding)
            offset += padding

        super().__init__(
            name=name,
            shape=shape,
            itemsize=offset,
            base_alignment=struct_base_alignment
        )
        self._fields_ = fields
        self._offsets_ = tuple(offsets)
        self._paddings_ = tuple(paddings)

    @Lazy.variable(plural=True)
    @staticmethod
    def _fields_() -> tuple[Field, ...]:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _offsets_() -> tuple[int, ...]:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _paddings_() -> tuple[int, ...]:
        return ()

    @Lazy.property()
    @staticmethod
    def _dtype_(
        fields: tuple[Field, ...],
        offsets: tuple[int, ...],
        itemsize: int
    ) -> np.dtype:
        return np.dtype({
            "names": tuple(field._name_ for field in fields),
            "formats": tuple((field._dtype_, field._shape_) for field in fields),
            "offsets": offsets,
            "itemsize": itemsize
        })

    @Lazy.property(plural=True)
    @staticmethod
    def _pointers_(
        fields: tuple[Field, ...]
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        return tuple(
            ((field._name_,) + name_chain, base_ndim)
            for field in fields
            for name_chain, base_ndim in field._pointers_
        )
