__all__ = [
    "AtomicBufferFormat",
    "AttributesBuffer",
    "BufferFormat",
    "StructuredBufferFormat",
    "IndexBuffer",
    "TextureIDBuffer",
    "TransformFeedbackBuffer",
    "UniformBlockBuffer"
]


from contextlib import contextmanager
from enum import Enum
from functools import reduce
import operator as op
import re
from typing import (
    ClassVar,
    Generator
)

import moderngl
import numpy as np

from ..custom_typing import VertexIndexType
from ..lazy.core import LazyObject
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..rendering.context import Context


class GLBufferLayout(Enum):
    PACKED = 0
    STD140 = 1


class BufferFormat(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...]
    ) -> None:
        super().__init__()
        self._name_ = name
        self._shape_ = shape

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _name_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _shape_(cls) -> tuple[int, ...]:
        return ()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _itemsize_(cls) -> int:
        # Implemented in subclasses.
        return 0

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _size_(
        cls,
        shape: tuple[int, ...]
    ) -> int:
        return reduce(op.mul, shape, 1)

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _nbytes_(
        cls,
        itemsize: int,
        size: int
    ) -> int:
        return itemsize * size

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _is_empty_(
        cls,
        size: int
    ) -> bool:
        return not size

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _dtype_(cls) -> np.dtype:
        # Implemented in subclasses.
        return np.dtype("f4")


class AtomicBufferFormat(BufferFormat):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        base_char: str,
        base_itemsize: int,
        base_ndim: int,
        n_col: int,
        n_row: int,
        row_itemsize_factor: int
    ) -> None:
        super().__init__(
            name=name,
            shape=shape
        )
        self._base_char_ = base_char
        self._base_itemsize_ = base_itemsize
        self._base_ndim_ = base_ndim
        self._n_col_ = n_col
        self._n_row_ = n_row
        self._row_itemsize_factor_ = row_itemsize_factor

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _base_char_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _base_itemsize_(cls) -> int:
        return 0

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _base_ndim_(cls) -> int:
        return 0

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _n_col_(cls) -> int:
        return 0

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _n_row_(cls) -> int:
        return 0

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _row_itemsize_factor_(cls) -> int:
        return 0

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _row_itemsize_(
        cls,
        row_itemsize_factor: int,
        base_itemsize: int
    ) -> int:
        return row_itemsize_factor * base_itemsize

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _itemsize_(
        cls,
        n_row: int,
        row_itemsize: int
    ) -> int:
        return n_row * row_itemsize

    @Lazy.property(LazyMode.UNWRAPPED)
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


class StructuredBufferFormat(BufferFormat):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        children: tuple[BufferFormat, ...],
        offsets: tuple[int, ...],
        itemsize: int
    ) -> None:
        super().__init__(
            name=name,
            shape=shape
        )
        self._children_.add(*children)
        self._offsets_ = offsets
        self._itemsize_ = itemsize

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _children_(cls) -> list[BufferFormat]:
        return []

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _offsets_(cls) -> tuple[int, ...]:
        return ()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _dtype_(
        cls,
        children__name: list[str],
        children__dtype: list[np.dtype],
        children__shape: list[tuple[int, ...]],
        offsets: tuple[int, ...],
        itemsize: int
    ) -> np.dtype:
        return np.dtype({
            "names": children__name,
            "formats": list(zip(children__dtype, children__shape, strict=True)),
            "offsets": list(offsets),
            "itemsize": itemsize
        })


class GLBuffer(LazyObject):
    __slots__ = ()

    _VACANT_BUFFERS: ClassVar[list[moderngl.Buffer]] = []
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
        field: str,
        child_structs: dict[str, list[str]] | None,
        array_lens: dict[str, int] | None
    ) -> None:
        super().__init__()
        self._field_ = field
        if child_structs is not None:
            self._child_struct_items_ = tuple(
                (name, tuple(child_struct_fields))
                for name, child_struct_fields in child_structs.items()
            )
        if array_lens is not None:
            self._array_len_items_ = tuple(array_lens.items())

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _field_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _child_struct_items_(cls) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return ()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _array_len_items_(cls) -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.PACKED

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _buffer_format_(
        cls,
        field: str,
        child_struct_items: tuple[tuple[str, tuple[str, ...]], ...],
        array_len_items: tuple[tuple[str, int], ...],
        layout: GLBufferLayout
    ) -> BufferFormat:

        def parse_field_str(
            field_str: str,
            array_lens_dict: dict[str, int]
        ) -> tuple[str, str, tuple[int, ...]]:
            pattern = re.compile(r"""
                (?P<dtype_str>\w+?)
                \s
                (?P<name>\w+?)
                (?P<shape>(\[\w+?\])*)
            """, flags=re.VERBOSE)
            match_obj = pattern.fullmatch(field_str)
            assert match_obj is not None
            dtype_str = match_obj.group("dtype_str")
            name = match_obj.group("name")
            shape = tuple(
                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else array_lens_dict[s]
                for index_match in re.finditer(r"\[(\w+?)\]", match_obj.group("shape"))
            )
            return (dtype_str, name, shape)

        def get_atomic_format(
            name: str,
            shape: tuple[int, ...],
            gl_dtype_str: str
        ) -> AtomicBufferFormat:
            base_char, base_itemsize, base_shape = cls._GL_DTYPES[gl_dtype_str]
            assert len(base_shape) <= 2 and all(2 <= l <= 4 for l in base_shape)
            shape_dict = dict(enumerate(base_shape))
            n_col = shape_dict.get(0, 1)
            n_row = shape_dict.get(1, 1)
            if layout == GLBufferLayout.STD140:
                row_itemsize_factor = n_col if not shape and n_col <= 2 and n_row == 1 else 4
            else:
                row_itemsize_factor = n_col
            return AtomicBufferFormat(
                name=name,
                shape=shape,
                base_char=base_char,
                base_itemsize=base_itemsize,
                base_ndim=len(base_shape),
                n_col=n_col,
                n_row=n_row,
                row_itemsize_factor=row_itemsize_factor
            )

        def get_structured_format(
            name: str,
            shape: tuple[int, ...],
            children: list[BufferFormat]
        ) -> StructuredBufferFormat:
            structured_base_alignment = 16
            offsets: list[int] = []
            offset: int = 0
            for child in children:
                if layout == GLBufferLayout.STD140:
                    if isinstance(child, AtomicBufferFormat):
                        base_alignment = child._row_itemsize_.value
                    elif isinstance(child, StructuredBufferFormat):
                        base_alignment = structured_base_alignment
                    else:
                        raise TypeError
                    offset += (-offset) % base_alignment
                offsets.append(offset)
                offset += child._nbytes_.value
            if layout == GLBufferLayout.STD140:
                offset += (-offset) % structured_base_alignment
            return StructuredBufferFormat(
                name=name,
                shape=shape,
                children=tuple(children),
                offsets=tuple(offsets),
                itemsize=offset
            )

        child_structs_dict = dict(child_struct_items)
        array_lens_dict = dict(array_len_items)

        def get_buffer_format(
            field: str
        ) -> BufferFormat:
            dtype_str, name, shape = parse_field_str(field, array_lens_dict)
            if (child_struct_fields := child_structs_dict.get(dtype_str)) is None:
                return get_atomic_format(
                    name=name,
                    shape=shape,
                    gl_dtype_str=dtype_str
                )
            return get_structured_format(
                name=name,
                shape=shape,
                children=[
                    get_buffer_format(child_struct_field)
                    for child_struct_field in child_struct_fields
                ]
            )

        return get_buffer_format(field)

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _np_buffer_(
        cls,
        buffer_format__shape: tuple[int, ...],
        buffer_format__dtype: np.dtype
    ) -> np.ndarray:
        return np.zeros(buffer_format__shape, dtype=buffer_format__dtype)

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _np_buffer_pointers_(
        cls,
        np_buffer: np.ndarray,
        _buffer_format_: BufferFormat
    ) -> dict[str, tuple[np.ndarray, int]]:

        def get_pointers(
            np_buffer_pointer: np.ndarray,
            buffer_format: BufferFormat,
            name_chain: tuple[str, ...]
        ) -> Generator[tuple[str, np.ndarray, int], None, None]:
            if isinstance(buffer_format, AtomicBufferFormat):
                yield ".".join(name_chain), np_buffer_pointer["_"], buffer_format._base_ndim_.value
            elif isinstance(buffer_format, StructuredBufferFormat):
                for child in buffer_format._children_:
                    name = child._name_.value
                    yield from get_pointers(
                        np_buffer_pointer[name],
                        child,
                        name_chain + (name,)
                    )

        return {
            key: (np_buffer_pointer, base_ndim)
            for key, np_buffer_pointer, base_ndim in get_pointers(np_buffer, _buffer_format_, ())
        }

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _np_buffer_pointer_keys_(
        cls,
        np_buffer_pointers: dict[str, np.ndarray]
    ) -> tuple[str, ...]:
        return tuple(np_buffer_pointers)

    @classmethod
    def _fetch_buffer(cls) -> moderngl.Buffer:
        if cls._VACANT_BUFFERS:
            return cls._VACANT_BUFFERS.pop()
        return Context.buffer()

    @classmethod
    def _finalize_buffer(
        cls,
        buffer: moderngl.Buffer
    ) -> None:
        cls._VACANT_BUFFERS.append(buffer)

    @classmethod
    def _write_to_buffer(
        cls,
        buffer: moderngl.Buffer,
        np_buffer: np.ndarray,
        np_buffer_pointers: dict[str, tuple[np.ndarray, int]],
        data_dict: dict[str, np.ndarray]
    ) -> None:
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data = data_dict[key]
            if not np_buffer_pointer.size:
                assert not data.size
                return
            data_expanded = np.expand_dims(data, axis=tuple(range(-2, -base_ndim)))
            assert np_buffer_pointer.shape == data_expanded.shape
            np_buffer_pointer[...] = data_expanded

        buffer.orphan(np_buffer.nbytes)
        buffer.write(np_buffer.tobytes())

    @classmethod
    def _read_from_buffer(
        cls,
        buffer: moderngl.Buffer,
        np_buffer: np.ndarray,
        np_buffer_pointers: dict[str, tuple[np.ndarray, int]]
    ) -> dict[str, np.ndarray]:
        data_dict: dict[str, np.ndarray] = {}
        np_buffer[...] = np.frombuffer(buffer.read(), dtype=np_buffer.dtype).reshape(np_buffer.shape)
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data_expanded = np_buffer_pointer[...]
            data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))
            data_dict[key] = data
        return data_dict


class GLWriteOnlyBuffer(GLBuffer):
    __slots__ = ()

    @Lazy.property(LazyMode.UNWRAPPED)
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

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _data_dict_(cls) -> dict[str, np.ndarray]:
        return {}

    def write(
        self,
        data_dict: dict[str, np.ndarray]
    ) -> None:
        self._data_dict_ = data_dict

    def get_buffer(self) -> moderngl.Buffer:
        return self._buffer_.value


class GLReadOnlyBuffer(GLBuffer):
    __slots__ = ()

    @contextmanager
    def temporary_buffer(self) -> Generator[moderngl.Buffer, None, None]:
        buffer = self._fetch_buffer()
        buffer.orphan(self._buffer_format_._nbytes_.value)
        yield buffer
        self._finalize_buffer(buffer)

    def read(
        self,
        buffer: moderngl.Buffer
    ) -> dict[str, np.ndarray]:
        return self._read_from_buffer(
            buffer=buffer,
            np_buffer=self._np_buffer_.value,
            np_buffer_pointers=self._np_buffer_pointers_.value
        )


class TextureIDBuffer(GLBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        field: str,
        array_lens: dict[str, int] | None = None
    ) -> None:
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        super().__init__(
            field=replaced_field,
            child_structs=None,
            array_lens=array_lens
        )


class UniformBlockBuffer(GLWriteOnlyBuffer):
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

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.STD140


class AttributesBuffer(GLWriteOnlyBuffer):
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

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        # Let's keep using std140 layout, hopefully giving a faster processing speed.
        return GLBufferLayout.STD140


class IndexBuffer(GLWriteOnlyBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        data: VertexIndexType | None
    ) -> None:
        data_len = 0 if data is None else len(data)
        super().__init__(
            field="uint __index__[__NUM_INDEX__]",
            child_structs={},
            array_lens={
                "__NUM_INDEX__": data_len
            }
        )
        if data is not None:
            self.write({
                "": data
            })
            self._omitted_ = False

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _omitted_(cls) -> bool:
        return True


class TransformFeedbackBuffer(GLReadOnlyBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        fields: list[str],
        child_structs: dict[str, list[str]] | None = None,
        num_vertex: int,
        array_lens: dict[str, int] | None = None
    ) -> None:
        # The interface should be similar to `AttributesBuffer`.
        if child_structs is None:
            child_structs = {}
        if array_lens is None:
            array_lens = {}
        super().__init__(
            field="__VertexStruct__ __vertex__[__NUM_VERTEX__]",
            child_structs={
                "__VertexStruct__": fields,
                **child_structs
            },
            array_lens={
                "__NUM_VERTEX__": num_vertex,
                **array_lens
            }
        )
