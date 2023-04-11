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
    #Any,
    #Callable,
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
        #dtype: np.dtype,
        shape: tuple[int, ...]
        #itemsize: int
        #base_alignment: int,
        #layout: GLBufferLayout
        #atomic_ndim: int | None
    ) -> None:
        super().__init__()
        self._name_ = name
        #self._dtype_ = dtype
        self._shape_ = shape
        #self._itemsize_ = itemsize
        #self._base_alignment_ = base_alignment
        #self._layout_ = layout
        #self._atomic_ndim_ = atomic_ndim

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _name_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _shape_(cls) -> tuple[int, ...]:
        return ()

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _layout_(cls) -> GLBufferLayout:
    #    return GLBufferLayout.PACKED

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _atomic_ndim_(cls) -> int | None:
    #    return None

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _dtype_and_base_alignment_(cls) -> tuple[np.dtype, int]:
    #    # Implemented in subclasses.
    #    return np.dtype("f4"), 0

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _dtype_(
    #    cls,
    #    dtype_and_base_alignment: tuple[np.dtype, int]
    #) -> np.dtype:
    #    dtype, _ = dtype_and_base_alignment
    #    return dtype

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _itemsize_(cls) -> int:
        # Implemented in subclasses.
        return 0

    #@Lazy.variable(LazyMode.UNWRAPPED)
    #@classmethod
    #def _base_alignment_(cls) -> int:
    #    # Implemented in subclasses.
    #    return 0

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

    #@Lazy.property(LazyMode.SHARED)
    #@classmethod
    #def _name_chains_(cls) -> tuple[tuple[str, ...], ...]:
    #    # Implemented in subclasses.
    #    return ()

    #@Lazy.property(LazyMode.SHARED)
    #@classmethod
    #def _names_(
    #    cls,
    #    name_chains: tuple[tuple[str, ...], ...]
    #) -> tuple[str, ...]:
    #    return tuple(".".join(name_chain) for name_chain in name_chains)

    #@Lazy.property(LazyMode.SHARED)
    #@classmethod
    #def _keys_(cls) -> tuple[str, ...]:
    #    return ("",)

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _dtype_(cls) -> np.dtype:
        # Implemented in subclasses.
        return np.dtype("f4")

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _traverse_items_(cls) -> tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...]:
    #    # Implemented in subclasses.
    #    return NotImplemented

    #def _traverse(
    #    self,
    #    callback: Callable[[np.ndarray, str, int], None],
    #    np_buffer: np.ndarray
    #) -> None:

    #    def traverse(
    #        callback: Callable[[np.ndarray, str, int], None],
    #        np_buffer_pointer: np.ndarray,
    #        buffer_format: BufferFormat,
    #        name_chain: tuple[str, ...]
    #    ) -> None:
    #        if isinstance(buffer_format, AtomicBufferFormat):
    #            callback(np_buffer_pointer["_"], ".".join(name_chain), buffer_format._base_ndim_.value)
    #        elif isinstance(buffer_format, StructuredBufferFormat):
    #            for child in buffer_format._children_:
    #                name = child._name_.value
    #                traverse(
    #                    callback,
    #                    np_buffer_pointer[name],
    #                    child,
    #                    name_chain + (name,)
    #                )
    #        #assert (names := dtype.names) is not None
    #        #if names and (gl_dtype_str := names[0]) in cls._GL_DTYPES:
    #        #    assert len(names) == 1
    #        #    yield name_chain, buffer_pointer, gl_dtype_str
    #        #else:
    #        #    for name in names:
    #        #        yield from get_pointer_items(
    #        #            name_chain + (name,),
    #        #            buffer_pointer[name],
    #        #            dtype[name]
    #        #        )

    #    traverse(callback, np_buffer, self, ())

    #def write(
    #    self,
    #    buffer: moderngl.Buffer,
    #    data_dict: dict[str, np.ndarray]
    #) -> None:
    #    #cls = type(self)
    #    #if (buffer := self._buffer) is None:
    #    #    if cls._VACANT_BUFFERS:
    #    #        buffer = cls._VACANT_BUFFERS.pop()
    #    #    else:
    #    #        buffer = Context.buffer()
    #    #    weakref.finalize(buffer, list.append, cls._VACANT_BUFFERS, buffer)
    #    #    self._buffer = buffer

    #    #self.initialize_buffer()
    #    #buffer = self.get_buffer()
    #    #buffer_format = self._buffer_format_
    #    np_buffer = np.zeros(self._shape_.value, dtype=self._dtype_.value)
    #    #self._np_buffer = np_buffer
    #    #np_buffer_pointers = self._get_np_buffer_pointers(np_buffer, buffer_format)

    #    def traverse_callback(
    #        np_buffer_pointer: np.ndarray,
    #        key: str,
    #        base_ndim: int
    #    ) -> None:
    #        data = data_dict[key]
    #        if not np_buffer_pointer.size:
    #            assert not data.size
    #            return
    #        data_expanded = np.expand_dims(data, axis=tuple(range(-2, -base_ndim)))
    #        assert np_buffer_pointer.shape == data_expanded.shape
    #        np_buffer_pointer[...] = data_expanded

    #    self._traverse(
    #        callback=traverse_callback,
    #        np_buffer=np_buffer
    #        #buffer_format=self
    #    )

    #    #for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
    #    #    #if not np_buffer_pointer.size:
    #    #    #    continue
    #    #    data = data_dict[key]
    #    #    #gl_dtype_ndim = base_ndim._atomic_ndim_.value
    #    #    data_expanded = np.expand_dims(data, axis=tuple(range(-2, -base_ndim)))
    #    #    assert np_buffer_pointer.shape == data_expanded.shape
    #    #    np_buffer_pointer[...] = data_expanded

    #    buffer.orphan(self._nbytes_.value)
    #    buffer.write(np_buffer.tobytes())
    #    #self._buffer = buffer

    #def read(
    #    self,
    #    buffer: moderngl.Buffer
    #) -> dict[str, np.ndarray]:
    #    #cls = type(self)
    #    #buffer = self.get_buffer()
    #    #buffer_format = self._buffer_format_
    #    np_buffer = np.frombuffer(buffer.read(), dtype=self._dtype_.value).reshape(self._shape_.value)
    #    data_dict: dict[str, np.ndarray] = {}
    #    #np_buffer_pointers = self._get_np_buffer_pointers(np_buffer, buffer_format)

    #    def traverse_callback(
    #        np_buffer_pointer: np.ndarray,
    #        key: str,
    #        base_ndim: int
    #    ) -> None:
    #        data_expanded = np_buffer_pointer[...]
    #        data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))  # TODO
    #        data_dict[key] = data

    #    #for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
    #    #    #if not np_buffer_pointer.size:
    #    #    #    continue
    #    #    data_expanded = np_buffer_pointer[...]
    #    #    #gl_dtype_ndim = atomic_format._atomic_ndim_.value
    #    #    data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))  # TODO
    #    #    data_dict[key] = data

    #    self._traverse(
    #        callback=traverse_callback,
    #        np_buffer=np_buffer
    #        #buffer_format=self
    #    )

    #    return data_dict


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
        #layout: GLBufferLayout,
        #gl_dtype_str: str
    ) -> None:
        super().__init__(
            name=name,
            shape=shape
            #itemsize=n_row * row_itemsize_factor * atomic_base.itemsize
            #layout=layout
        )
        self._base_char_ = base_char
        self._base_itemsize_ = base_itemsize
        self._base_ndim_ = base_ndim
        self._n_col_ = n_col
        self._n_row_ = n_row
        self._row_itemsize_factor_ = row_itemsize_factor
        #atomic_base_str, atomic_shape = type(self)._GL_DTYPES[gl_dtype_str]
        #self._atomic_base_str_ = atomic_base_str
        #self._atomic_shape_ = atomic_shape

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

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _atomic_base_(
    #    cls,
    #    atomic_base_str: str
    #) -> np.dtype:
    #    return np.dtype(atomic_base_str)

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

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _base_alignment_(
    #    cls,
    #    row_itemsize: int
    #) -> int:
    #    return row_itemsize

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _item_format_str_(
    #    cls,
    #    n_col: int,
    #    base_char: str,
    #    base_itemsize: int,
    #    row_itemsize_factor: int,
    #    n_row: int
    #) -> str:
    #    components = [f"{n_col}{base_char}{base_itemsize}"]
    #    if row_itemsize_factor != n_col:
    #        components.append(f"{row_itemsize_factor - n_col}x{base_itemsize}")
    #    return " ".join(components * n_row)

    #@Lazy.property(LazyMode.SHARED)
    #@classmethod
    #def _name_chains_(cls) -> tuple[tuple[str, ...], ...]:
    #    # Implemented in subclasses.
    #    return ((),)

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

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _atomic_base_str_(cls) -> str:
    #    return ""

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _atomic_shape_(cls) -> tuple[int, ...]:
    #    return ()

    #@Lazy.property(LazyMode.SHARED)
    #@classmethod
    #def _atomic_ndim_(
    #    cls,
    #    atomic_shape: tuple[int, ...]
    #) -> int:
    #    return len(atomic_shape)

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _zero_dimensional_(
    #    cls,
    #    shape: tuple[int, ...],
    #) -> bool:
    #    return not shape

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _dtype_and_base_alignment_(
    #    cls,
    #    atomic_base_str: str,
    #    atomic_shape: tuple[int, ...],
    #    zero_dimensional: bool,
    #    layout: GLBufferLayout
    #) -> tuple[np.dtype, int]:
    #    atomic_base = np.dtype(atomic_base_str)
    #    assert len(atomic_shape) <= 2 and all(2 <= l <= 4 for l in atomic_shape)
    #    shape_dict = dict(enumerate(atomic_shape))
    #    n_col = shape_dict.get(0, 1)
    #    n_row = shape_dict.get(1, 1)
    #    if layout == GLBufferLayout.STD140:
    #        row_itemsize_factor = n_col if zero_dimensional and n_col <= 2 and n_row == 1 else 4
    #    else:
    #        row_itemsize_factor = n_col
    #    row_itemsize = row_itemsize_factor * atomic_base.itemsize
    #    return np.dtype((np.dtype({
    #        "names": ["_"],
    #        "formats": [(atomic_base, (n_col,))],
    #        "itemsize": row_itemsize
    #    }), (n_row,))), row_itemsize


class StructuredBufferFormat(BufferFormat):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        #layout: GLBufferLayout,
        children: tuple[BufferFormat, ...],
        offsets: tuple[int, ...],
        itemsize: int
    ) -> None:
        super().__init__(
            name=name,
            shape=shape
            #layout=layout
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

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _base_alignment_(cls) -> int:
    #    return 16

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _item_format_str_(
    #    cls,
    #    _children_: list[BufferFormat],
    #    offsets: tuple[int, ...],
    #    itemsize: int
    #) -> str:
    #    components: list[str] = []
    #    current_stop: int = 0
    #    for child, offset in zip(_children_, offsets):
    #        if current_stop != offset:
    #            components.append(f"{offset - current_stop}x")
    #        components.extend([child._item_format_str_.value] * child._size_.value)
    #        current_stop = offset + child._nbytes_.value
    #    if current_stop != itemsize:
    #        components.append(f"{itemsize - current_stop}x")
    #    return " ".join(components)

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _name_chains_(
    #    cls,
    #    children__name: list[str],
    #    children__name_chains: list[tuple[tuple[str, ...], ...]]
    #) -> tuple[tuple[str], ...]:
    #    return tuple(
    #        (child_name,) + name_chain
    #        for child_name, child_name_chains in zip(children__name, children__name_chains, strict=True)
    #        for name_chain in child_name_chains
    #    )

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

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _dtype_and_base_alignment_(
    #    cls,
    #    _children_: list[BufferFormat],
    #    layout: GLBufferLayout
    #) -> tuple[np.dtype, int]:
    #    base_alignment = 16
    #    names: list[str] = []
    #    formats: list[tuple[np.dtype, tuple[int, ...]]] = []
    #    offsets: list[int] = []
    #    offset: int = 0
    #    for child in _children_:
    #        names.append(child._name_.value)
    #        formats.append((child._dtype_.value, child._shape_.value))

    #        if layout == GLBufferLayout.STD140:
    #            offset += (-offset) % child._base_alignment_.value
    #        offsets.append(offset)
    #        offset += child._nbytes_.value
    #    if layout == GLBufferLayout.STD140:
    #        offset += (-offset) % base_alignment

    #    return np.dtype({
    #        "names": names,
    #        "formats": formats,
    #        "offsets": offsets,
    #        "itemsize": offset
    #    }), base_alignment


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
        #data_dict: dict[str, np.ndarray] | None
        #data: dict[str, np.ndarray] | None
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
        #if data_dict is not None:
        #    self._data_dict_ = data_dict
        #if data is not None:
        #    self._data_ = data
        #self._buffer: moderngl.Buffer | None = None
        #self._np_buffer: np.ndarray | None = None

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _field_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _child_struct_items_(cls) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return ()

    #@Lazy.variable(LazyMode.UNWRAPPED)
    #@classmethod
    #def _data_dict_(cls) -> dict[str, np.ndarray]:
    #    return {}

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _array_len_items_(cls) -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.PACKED

    #@Lazy.variable(LazyMode.UNWRAPPED)
    #@classmethod
    #def _data_(cls) -> dict[str, np.ndarray]:
    #    return {}

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
            #row_itemsize = row_itemsize_factor * atomic_base.itemsize
            #return np.dtype((np.dtype({
            #    "names": [gl_dtype_str],
            #    "formats": [(atomic_base, (n_col,))],
            #    "itemsize": row_itemsize
            #}), (n_row,))), row_itemsize

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


        #def get_composite_dtype(
        #    children_buffer_format: tuple[BufferFormat, ...]
        #    #base_alignment: int,
        #    #children__name: list[str],
        #    #children__dtype: list[np.dtype],
        #    #children__base_alignment: list[int],
        #    #shape: tuple[int, ...]
        #) -> tuple[np.dtype, int]:
        #    base_alignment = 16
        #    names: list[str] = []
        #    formats: list[tuple[np.dtype, tuple[int, ...]]] = []
        #    offsets: list[int] = []
        #    offset: int = 0
        #    for child_buffer_format in children_buffer_format:
        #        names.append(child_buffer_format._name_.value)
        #        formats.append((child_buffer_format._dtype_.value, child_buffer_format._shape_.value))

        #        if layout == GLBufferLayout.STD140:
        #            offset += (-offset) % child_buffer_format._base_alignment_.value
        #        offsets.append(offset)
        #        offset += child_buffer_format._nbytes_.value
        #    if layout == GLBufferLayout.STD140:
        #        offset += (-offset) % base_alignment

        #    return np.dtype({
        #        "names": names,
        #        "formats": formats,
        #        "offsets": offsets,
        #        "itemsize": offset
        #    }), base_alignment

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
                    #layout=layout,
                    gl_dtype_str=dtype_str
                )
                #dtype, base_alignment, atomic_ndim = get_atomic_dtype(dtype_str, not shape)
            #else:
                #dtype, base_alignment = get_composite_dtype(tuple(
                #    get_buffer_format(child_struct_field)
                #    for child_struct_field in child_struct_fields
                #))
                #atomic_ndim = None
            return get_structured_format(
                name=name,
                shape=shape,
                #layout=layout,
                children=[
                    get_buffer_format(child_struct_field)
                    for child_struct_field in child_struct_fields
                ]
            )
            #return BufferFormat(
            #    name=name,
            #    #dtype=dtype,
            #    shape=shape,
            #    #base_alignment=base_alignment,
            #    #atomic_ndim=atomic_ndim
            #    layout=layout
            #)
                #return AtomicDTypeNode(
                #    name=name,
                #    shape=shape,
                #    dtype_str=dtype_str,
                #    layout=layout
                #)
            #return DTypeNode(
            #    name=name,
            #    shape=shape,
            #    children=[
            #        get_buffer_format(child_struct_field)
            #        for child_struct_field in child_struct_fields
            #    ],
            #    layout=layout
            #)

        #name, template, _ = get_buffer_format(field)
        return get_buffer_format(field)

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _name_(
    #    cls,
    #    buffer_format: tuple[str, np.ndarray]
    #) -> str:
    #    name, _ = buffer_format
    #    return name

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _shape_(
    #    cls,
    #    buffer_format: tuple[str, np.ndarray]
    #) -> tuple[int, ...]:
    #    _, template = buffer_format
    #    return template.shape

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _nbytes_(
    #    cls,
    #    buffer_format: tuple[str, np.ndarray]
    #) -> int:
    #    _, template = buffer_format
    #    return template.nbytes

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _is_empty_(
    #    cls,
    #    nbytes: int
    #) -> bool:
    #    return not nbytes

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _dtype_(
    #    cls,
    #    name_and_dtype: tuple[str, np.dtype]
    #) -> np.dtype:
    #    _, dtype = name_and_dtype
    #    return dtype

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _buffer_(
    #    cls,
    #    _buffer_format_: BufferFormat,
    #    data_dict: dict[str, np.ndarray]
    #) -> moderngl.Buffer:
    #    if cls._VACANT_BUFFERS:
    #        buffer = cls._VACANT_BUFFERS.pop()
    #    else:
    #        buffer = Context.buffer()
    #    _buffer_format_.write(buffer, data_dict)
    #    return buffer

    #@_buffer_.finalizer
    #@classmethod
    #def _buffer_finalizer(
    #    cls,
    #    buffer: moderngl.Buffer
    #) -> None:
    #    cls._VACANT_BUFFERS.append(buffer)

    #@Lazy.variable(LazyMode.UNWRAPPED)
    #@classmethod
    #def _buffer_(cls) -> moderngl.Buffer:
    #    return NotImplemented

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
                #callback(np_buffer_pointer["_"], ".".join(name_chain), buffer_format._base_ndim_.value)
            elif isinstance(buffer_format, StructuredBufferFormat):
                for child in buffer_format._children_:
                    name = child._name_.value
                    yield from get_pointers(
                        np_buffer_pointer[name],
                        child,
                        name_chain + (name,)
                    )
            #assert (names := dtype.names) is not None
            #if names and (gl_dtype_str := names[0]) in cls._GL_DTYPES:
            #    assert len(names) == 1
            #    yield name_chain, buffer_pointer, gl_dtype_str
            #else:
            #    for name in names:
            #        yield from get_pointer_items(
            #            name_chain + (name,),
            #            buffer_pointer[name],
            #            dtype[name]
            #        )

        return {
            key: (np_buffer_pointer, base_ndim)
            for key, np_buffer_pointer, base_ndim in get_pointers(np_buffer, _buffer_format_, ())
        }

        #return np.zeros(shape, dtype=buffer_format__dtype)

    #@Lazy.variable(LazyMode.UNWRAPPED)
    #@classmethod
    #def _data_dict_(cls) -> dict[str, np.ndarray]:
    #    return NotImplemented

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
            data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))  # TODO
            data_dict[key] = data
        return data_dict


    #def initialize_buffer(self) -> None:
    #    cls = type(self)
    #    if (buffer := self._buffer_.value) is None:
    #        if cls._VACANT_BUFFERS:
    #            buffer = cls._VACANT_BUFFERS.pop()
    #        else:
    #            buffer = Context.buffer()
    #        #weakref.finalize(buffer, list.append, cls._VACANT_BUFFERS, buffer)
    #        self._buffer_ = buffer
    #    buffer.clear()

    #def get_buffer(self) -> moderngl.Buffer:
    #    assert (buffer := self._buffer_.value) is not None
    #    return buffer


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

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _data_dict_(
    #    cls,
    #    buffer: moderngl.Buffer,
    #    np_buffer: np.ndarray,
    #    np_buffer_pointers: dict[str, tuple[np.ndarray, int]]
    #) -> dict[str, np.ndarray]:
    #    return cls._read_from_buffer(
    #        buffer=buffer,
    #        np_buffer=np_buffer,
    #        np_buffer_pointers=np_buffer_pointers
    #    )

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


#class DTypeNode(LazyObject):
#    __slots__ = ()

#    def __init__(
#        self,
#        name: str,
#        shape: tuple[int, ...],
#        children: "list[DTypeNode]",
#        layout: GLBufferLayout
#    ) -> None:
#        super().__init__()
#        self._name_ = name
#        self._shape_ = shape
#        self._children_ = children
#        self._layout_ = layout

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _name_(cls) -> str:
#        return ""

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _shape_(cls) -> tuple[int, ...]:
#        return ()

#    @Lazy.property(LazyMode.SHARED)
#    @classmethod
#    def _base_alignment_(cls) -> int:
#        return 16

#    @Lazy.variable(LazyMode.COLLECTION)
#    @classmethod
#    def _children_(cls) -> "list[DTypeNode]":
#        return []

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _layout_(cls) -> GLBufferLayout:
#        return GLBufferLayout.PACKED

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _dtype_(
#        cls,
#        shape: tuple[int, ...],
#        base_alignment: int,
#        children__name: list[str],
#        children__dtype: list[np.dtype],
#        children__base_alignment: list[int],
#        layout: GLBufferLayout
#    ) -> np.dtype:
#        offsets: list[int] = []
#        offset: int = 0
#        for child_dtype, child_base_alignment in zip(children__dtype, children__base_alignment, strict=True):
#            if layout == GLBufferLayout.STD140:
#                offset += (-offset) % child_base_alignment
#            offsets.append(offset)
#            offset += child_dtype.itemsize
#        if layout == GLBufferLayout.STD140:
#            offset += (-offset) % base_alignment

#        return np.dtype((np.dtype({
#            "names": children__name,
#            "formats": children__dtype,
#            "offsets": offsets,
#            "itemsize": offset
#        }), shape))

#    def _write(
#        self,
#        array_ptr: np.ndarray,
#        data: np.ndarray | dict[str, Any]
#    ) -> None:
#        assert isinstance(data, dict)
#        for child_node in self._children_:
#            name = child_node._name_.value
#            child_node._write(array_ptr[name], data[name])


#class AtomicDTypeNode(DTypeNode):
#    __slots__ = ()

#    _GL_DTYPES: ClassVar[dict[str, np.dtype]] = {
#        "int":     np.dtype(("i4", ())),
#        "ivec2":   np.dtype(("i4", (2,))),
#        "ivec3":   np.dtype(("i4", (3,))),
#        "ivec4":   np.dtype(("i4", (4,))),
#        "uint":    np.dtype(("u4", ())),
#        "uvec2":   np.dtype(("u4", (2,))),
#        "uvec3":   np.dtype(("u4", (3,))),
#        "uvec4":   np.dtype(("u4", (4,))),
#        "float":   np.dtype(("f4", ())),
#        "vec2":    np.dtype(("f4", (2,))),
#        "vec3":    np.dtype(("f4", (3,))),
#        "vec4":    np.dtype(("f4", (4,))),
#        "mat2":    np.dtype(("f4", (2, 2))),
#        "mat2x3":  np.dtype(("f4", (2, 3))),  # TODO: check order
#        "mat2x4":  np.dtype(("f4", (2, 4))),
#        "mat3x2":  np.dtype(("f4", (3, 2))),
#        "mat3":    np.dtype(("f4", (3, 3))),
#        "mat3x4":  np.dtype(("f4", (3, 4))),
#        "mat4x2":  np.dtype(("f4", (4, 2))),
#        "mat4x3":  np.dtype(("f4", (4, 3))),
#        "mat4":    np.dtype(("f4", (4, 4))),
#        "double":  np.dtype(("f8", ())),
#        "dvec2":   np.dtype(("f8", (2,))),
#        "dvec3":   np.dtype(("f8", (3,))),
#        "dvec4":   np.dtype(("f8", (4,))),
#        "dmat2":   np.dtype(("f8", (2, 2))),
#        "dmat2x3": np.dtype(("f8", (2, 3))),
#        "dmat2x4": np.dtype(("f8", (2, 4))),
#        "dmat3x2": np.dtype(("f8", (3, 2))),
#        "dmat3":   np.dtype(("f8", (3, 3))),
#        "dmat3x4": np.dtype(("f8", (3, 4))),
#        "dmat4x2": np.dtype(("f8", (4, 2))),
#        "dmat4x3": np.dtype(("f8", (4, 3))),
#        "dmat4":   np.dtype(("f8", (4, 4))),
#    }

#    def __init__(
#        self,
#        name: str,
#        shape: tuple[int, ...],
#        dtype_str: str,
#        layout: GLBufferLayout
#    ) -> None:
#        super().__init__(
#            name=name,
#            shape=shape,
#            children=[],
#            layout=layout
#        )
#        self._dtype_str_ = dtype_str

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _dtype_str_(cls) -> str:
#        return ""

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _atomic_dtype_(
#        cls,
#        dtype_str: str
#    ) -> np.dtype:
#        return cls._GL_DTYPES[dtype_str]

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _dtype_(
#        cls,
#        atomic_dtype: np.dtype,
#        shape: tuple[int, ...],
#        layout: GLBufferLayout
#    ) -> np.dtype:
#        atomic_base = atomic_dtype.base
#        atomic_shape = atomic_dtype.shape
#        assert len(atomic_shape) <= 2 and all(2 <= l <= 4 for l in atomic_shape)
#        shape_dict = dict(enumerate(atomic_shape))
#        n_col = shape_dict.get(0, 1)
#        n_row = shape_dict.get(1, 1)
#        if layout == GLBufferLayout.STD140:
#            col_itemsize_factor = n_col if not shape and n_col <= 2 and n_row == 1 else 4
#        else:
#            col_itemsize_factor = n_col
#        col_itemsize = col_itemsize_factor * atomic_base.itemsize
#        return np.dtype((np.dtype({
#            "names": ["_"],
#            "formats": [(atomic_base, (n_col,))],
#            "itemsize": col_itemsize
#        }), (*shape, n_row)))

#    @Lazy.property(LazyMode.SHARED)
#    @classmethod
#    def _base_alignment_(
#        cls,
#        dtype: np.dtype
#    ) -> int:
#        return dtype.base.itemsize

#    def _write(
#        self,
#        array_ptr: np.ndarray,
#        data: np.ndarray | dict[str, Any]
#    ) -> None:
#        assert isinstance(data, np.ndarray)
#        if not array_ptr.size:
#            return
#        atomic_dtype_dim = self._atomic_dtype_.value.ndim
#        data_expanded = np.expand_dims(data, tuple(range(-2, -atomic_dtype_dim)))
#        assert array_ptr["_"].shape == data_expanded.shape
#        array_ptr["_"] = data_expanded


#class GLDynamicStruct(LazyObject):
#    __slots__ = ()

#    def __init__(
#        self,
#        *,
#        field: str,
#        child_structs: dict[str, list[str]] | None,
#        dynamic_array_lens: dict[str, int] | None
#    ) -> None:
#        super().__init__()
#        self._field_ = field
#        if child_structs is not None:
#            self._child_structs_ = tuple(
#                (name, tuple(child_struct_fields))
#                for name, child_struct_fields in child_structs.items()
#            )
#        if dynamic_array_lens is not None:
#            self._dynamic_array_lens_ = tuple(dynamic_array_lens.items())

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _field_(cls) -> str:
#        return ""

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _child_structs_(cls) -> tuple[tuple[str, tuple[str, ...]], ...]:
#        return ()

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _dynamic_array_lens_(cls) -> tuple[tuple[str, int], ...]:
#        return ()

#    @Lazy.variable(LazyMode.SHARED)
#    @classmethod
#    def _layout_(cls) -> GLBufferLayout:
#        return GLBufferLayout.PACKED

#    @Lazy.property(LazyMode.OBJECT)
#    @classmethod
#    def _dtype_node_(
#        cls,
#        field: str,
#        child_structs: tuple[tuple[str, tuple[str, ...]], ...],
#        dynamic_array_lens: tuple[tuple[str, int], ...],
#        layout: GLBufferLayout
#    ) -> DTypeNode:

#        def parse_field_str(
#            field_str: str,
#            dynamic_array_lens_dict: dict[str, int]
#        ) -> tuple[str, str, tuple[int, ...]]:
#            pattern = re.compile(r"""
#                (?P<dtype_str>\w+?)
#                \s
#                (?P<name>\w+?)
#                (?P<shape>(\[\w+?\])*)
#            """, flags=re.VERBOSE)
#            match_obj = pattern.fullmatch(field_str)
#            assert match_obj is not None
#            dtype_str = match_obj.group("dtype_str")
#            name = match_obj.group("name")
#            shape = tuple(
#                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else dynamic_array_lens_dict[s]
#                for index_match in re.finditer(r"\[(\w+?)\]", match_obj.group("shape"))
#            )
#            return (dtype_str, name, shape)

#        child_structs_dict = dict(child_structs)
#        dynamic_array_lens_dict = dict(dynamic_array_lens)

#        def get_dtype_node(
#            field: str,
#        ) -> DTypeNode:
#            dtype_str, name, shape = parse_field_str(field, dynamic_array_lens_dict)
#            if (child_struct_fields := child_structs_dict.get(dtype_str)) is None:
#                return AtomicDTypeNode(
#                    name=name,
#                    shape=shape,
#                    dtype_str=dtype_str,
#                    layout=layout
#                )
#            return DTypeNode(
#                name=name,
#                shape=shape,
#                children=[
#                    get_dtype_node(child_struct_field)
#                    for child_struct_field in child_struct_fields
#                ],
#                layout=layout
#            )

#        return get_dtype_node(field)

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _field_name_(
#        cls,
#        dtype_node__name: str
#    ) -> str:
#        return dtype_node__name

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _itemsize_(
#        cls,
#        dtype_node__dtype: np.dtype
#    ) -> int:
#        return dtype_node__dtype.itemsize

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _is_empty_(
#        cls,
#        itemsize: int
#    ) -> bool:
#        return itemsize == 0


#class GLDynamicBuffer(GLDynamicStruct):
#    __slots__ = ()

#    _VACANT_BUFFERS: list[moderngl.Buffer] = []

#    def __init__(
#        self,
#        *,
#        field: str,
#        child_structs: dict[str, list[str]] | None,
#        dynamic_array_lens: dict[str, int] | None,
#        data: np.ndarray | dict[str, Any]
#    ) -> None:
#        super().__init__(
#            field=field,
#            child_structs=child_structs,
#            dynamic_array_lens=dynamic_array_lens
#        )
#        self._data_ = data

#    @Lazy.variable(LazyMode.UNWRAPPED)
#    @classmethod
#    def _data_(cls) -> np.ndarray | dict[str, Any]:
#        return {}

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _data_storage_(
#        cls,
#        _dtype_node_: DTypeNode,
#        data: np.ndarray | dict[str, Any]
#    ) -> np.ndarray:
#        data_storage = np.zeros((), dtype=_dtype_node_._dtype_.value)
#        _dtype_node_._write(data_storage, data)
#        return data_storage

#    @Lazy.property(LazyMode.UNWRAPPED)
#    @classmethod
#    def _buffer_(
#        cls,
#        data_storage: np.ndarray,
#        itemsize: int
#    ) -> moderngl.Buffer:
#        if cls._VACANT_BUFFERS:
#            buffer = cls._VACANT_BUFFERS.pop()
#        else:
#            buffer = Context.buffer()

#        bytes_data = data_storage.tobytes()
#        assert itemsize == len(bytes_data)
#        if itemsize == 0:
#            buffer.clear()
#            return buffer
#        buffer.orphan(itemsize)
#        buffer.write(bytes_data)
#        return buffer

#    @_buffer_.finalizer
#    @classmethod
#    def _buffer_finalizer(
#        cls,
#        buffer: moderngl.Buffer
#    ) -> None:
#        cls._VACANT_BUFFERS.append(buffer)


class TextureIDBuffer(GLBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        field: str,
        array_lens: dict[str, int] | None = None
        #shape: tuple[int, ...] | None = None
    ) -> None:
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        super().__init__(
            field=replaced_field,
            child_structs=None,
            array_lens=array_lens
        )
        #if shape is not None:
        #    self._shape_ = shape

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _layout_(cls) -> GLBufferLayout:
    #    return GLBufferLayout.PACKED

    #@Lazy.variable(LazyMode.UNWRAPPED)
    #@classmethod
    #def _shape_(cls) -> tuple[int, ...]:
    #    return ()


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

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _vertex_dtype_(
    #    cls,
    #    dtype_node__dtype: np.dtype
    #) -> np.dtype:
    #    return dtype_node__dtype.base


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

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _layout_(cls) -> GLBufferLayout:
    #    return GLBufferLayout.PACKED


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
