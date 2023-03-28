__all__ = [
    "AttributesBuffer",
    "IndexBuffer",
    "TexturePlaceholders",
    "UniformBlockBuffer"
]


from enum import Enum
from functools import reduce
import operator as op
import re
from typing import (
    Any,
    ClassVar
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


class DTypeNode(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        children: "list[DTypeNode]",
        layout: GLBufferLayout
    ) -> None:
        super().__init__()
        self._name_ = name
        self._shape_ = shape
        self._children_ = children
        self._layout_ = layout

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _name_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _shape_(cls) -> tuple[int, ...]:
        return ()

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _base_alignment_(cls) -> int:
        return 16

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _children_(cls) -> "list[DTypeNode]":
        return []

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.PACKED

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _dtype_(
        cls,
        shape: tuple[int, ...],
        base_alignment: int,
        children__name: list[str],
        children__dtype: list[np.dtype],
        children__base_alignment: list[int],
        layout: GLBufferLayout
    ) -> np.dtype:
        offsets: list[int] = []
        offset: int = 0
        for child_dtype, child_base_alignment in zip(children__dtype, children__base_alignment, strict=True):
            if layout == GLBufferLayout.STD140:
                offset += (-offset) % child_base_alignment
            offsets.append(offset)
            offset += child_dtype.itemsize
        if layout == GLBufferLayout.STD140:
            offset += (-offset) % base_alignment

        return np.dtype((np.dtype({
            "names": children__name,
            "formats": children__dtype,
            "offsets": offsets,
            "itemsize": offset
        }), shape))

    def _write(
        self,
        array_ptr: np.ndarray,
        data: np.ndarray | dict[str, Any]
    ) -> None:
        assert isinstance(data, dict)
        for child_node in self._children_:
            name = child_node._name_.value
            child_node._write(array_ptr[name], data[name])


class AtomicDTypeNode(DTypeNode):
    __slots__ = ()

    _GL_DTYPE: ClassVar[dict[str, np.dtype]] = {
        "int":     np.dtype(("i4", ())),
        "ivec2":   np.dtype(("i4", (2,))),
        "ivec3":   np.dtype(("i4", (3,))),
        "ivec4":   np.dtype(("i4", (4,))),
        "uint":    np.dtype(("u4", ())),
        "uvec2":   np.dtype(("u4", (2,))),
        "uvec3":   np.dtype(("u4", (3,))),
        "uvec4":   np.dtype(("u4", (4,))),
        "float":   np.dtype(("f4", ())),
        "vec2":    np.dtype(("f4", (2,))),
        "vec3":    np.dtype(("f4", (3,))),
        "vec4":    np.dtype(("f4", (4,))),
        "mat2":    np.dtype(("f4", (2, 2))),
        "mat2x3":  np.dtype(("f4", (2, 3))),  # TODO: check order
        "mat2x4":  np.dtype(("f4", (2, 4))),
        "mat3x2":  np.dtype(("f4", (3, 2))),
        "mat3":    np.dtype(("f4", (3, 3))),
        "mat3x4":  np.dtype(("f4", (3, 4))),
        "mat4x2":  np.dtype(("f4", (4, 2))),
        "mat4x3":  np.dtype(("f4", (4, 3))),
        "mat4":    np.dtype(("f4", (4, 4))),
        "double":  np.dtype(("f8", ())),
        "dvec2":   np.dtype(("f8", (2,))),
        "dvec3":   np.dtype(("f8", (3,))),
        "dvec4":   np.dtype(("f8", (4,))),
        "dmat2":   np.dtype(("f8", (2, 2))),
        "dmat2x3": np.dtype(("f8", (2, 3))),
        "dmat2x4": np.dtype(("f8", (2, 4))),
        "dmat3x2": np.dtype(("f8", (3, 2))),
        "dmat3":   np.dtype(("f8", (3, 3))),
        "dmat3x4": np.dtype(("f8", (3, 4))),
        "dmat4x2": np.dtype(("f8", (4, 2))),
        "dmat4x3": np.dtype(("f8", (4, 3))),
        "dmat4":   np.dtype(("f8", (4, 4))),
    }

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype_str: str,
        layout: GLBufferLayout
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            children=[],
            layout=layout
        )
        self._dtype_str_ = dtype_str

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _dtype_str_(cls) -> str:
        return ""

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _atomic_dtype_(
        cls,
        dtype_str: str
    ) -> np.dtype:
        return cls._GL_DTYPE[dtype_str]

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _dtype_(
        cls,
        atomic_dtype: np.dtype,
        shape: tuple[int, ...],
        layout: GLBufferLayout
    ) -> np.dtype:
        atomic_base = atomic_dtype.base
        atomic_shape = atomic_dtype.shape
        assert len(atomic_shape) <= 2 and all(2 <= l <= 4 for l in atomic_shape)
        shape_dict = dict(enumerate(atomic_shape))
        n_col = shape_dict.get(0, 1)
        n_row = shape_dict.get(1, 1)
        if layout == GLBufferLayout.STD140:
            col_itemsize_factor = n_col if not shape and n_col <= 2 and n_row == 1 else 4
        else:
            col_itemsize_factor = n_col
        col_itemsize = col_itemsize_factor * atomic_base.itemsize
        return np.dtype((np.dtype({
            "names": ["_"],
            "formats": [(atomic_base, (n_col,))],
            "itemsize": col_itemsize
        }), (*shape, n_row)))

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _base_alignment_(
        cls,
        dtype: np.dtype
    ) -> int:
        return dtype.base.itemsize

    def _write(
        self,
        array_ptr: np.ndarray,
        data: np.ndarray | dict[str, Any]
    ) -> None:
        assert isinstance(data, np.ndarray)
        if not array_ptr.size:
            return
        atomic_dtype_dim = self._atomic_dtype_.value.ndim
        data_expanded = np.expand_dims(data, tuple(range(-2, -atomic_dtype_dim)))
        assert array_ptr["_"].shape == data_expanded.shape
        array_ptr["_"] = data_expanded


class GLDynamicStruct(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        field: str,
        child_structs: dict[str, list[str]] | None,
        dynamic_array_lens: dict[str, int] | None
    ) -> None:
        super().__init__()
        self._field_ = field
        if child_structs is not None:
            self._child_structs_ = tuple(
                (name, tuple(child_struct_fields))
                for name, child_struct_fields in child_structs.items()
            )
        if dynamic_array_lens is not None:
            self._dynamic_array_lens_ = tuple(dynamic_array_lens.items())

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _field_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _child_structs_(cls) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return ()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _dynamic_array_lens_(cls) -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.PACKED

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _dtype_node_(
        cls,
        field: str,
        child_structs: tuple[tuple[str, tuple[str, ...]], ...],
        dynamic_array_lens: tuple[tuple[str, int], ...],
        layout: GLBufferLayout
    ) -> DTypeNode:

        def parse_field_str(
            field_str: str,
            dynamic_array_lens_dict: dict[str, int]
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
                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else dynamic_array_lens_dict[s]
                for index_match in re.finditer(r"\[(\w+?)\]", match_obj.group("shape"))
            )
            return (dtype_str, name, shape)

        child_structs_dict = dict(child_structs)
        dynamic_array_lens_dict = dict(dynamic_array_lens)

        def get_dtype_node(
            field: str,
        ) -> DTypeNode:
            dtype_str, name, shape = parse_field_str(field, dynamic_array_lens_dict)
            if (child_struct_fields := child_structs_dict.get(dtype_str)) is None:
                return AtomicDTypeNode(
                    name=name,
                    shape=shape,
                    dtype_str=dtype_str,
                    layout=layout
                )
            return DTypeNode(
                name=name,
                shape=shape,
                children=[
                    get_dtype_node(child_struct_field)
                    for child_struct_field in child_struct_fields
                ],
                layout=layout
            )

        return get_dtype_node(field)

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _field_name_(
        cls,
        dtype_node__name: str
    ) -> str:
        return dtype_node__name

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _itemsize_(
        cls,
        dtype_node__dtype: np.dtype
    ) -> int:
        return dtype_node__dtype.itemsize

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _is_empty_(
        cls,
        itemsize: int
    ) -> bool:
        return itemsize == 0


class GLDynamicBuffer(GLDynamicStruct):
    __slots__ = ()

    _VACANT_BUFFERS: list[moderngl.Buffer] = []

    def __init__(
        self,
        *,
        field: str,
        child_structs: dict[str, list[str]] | None,
        dynamic_array_lens: dict[str, int] | None,
        data: np.ndarray | dict[str, Any]
    ) -> None:
        super().__init__(
            field=field,
            child_structs=child_structs,
            dynamic_array_lens=dynamic_array_lens
        )
        self._data_ = data

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _data_(cls) -> np.ndarray | dict[str, Any]:
        return {}

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _data_storage_(
        cls,
        _dtype_node_: DTypeNode,
        data: np.ndarray | dict[str, Any]
    ) -> np.ndarray:
        data_storage = np.zeros((), dtype=_dtype_node_._dtype_.value)
        _dtype_node_._write(data_storage, data)
        return data_storage

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _buffer_(
        cls,
        data_storage: np.ndarray,
        itemsize: int
    ) -> moderngl.Buffer:
        if cls._VACANT_BUFFERS:
            buffer = cls._VACANT_BUFFERS.pop()
        else:
            buffer = Context.buffer()

        bytes_data = data_storage.tobytes()
        assert itemsize == len(bytes_data)
        if itemsize == 0:
            buffer.clear()
            return buffer
        buffer.orphan(itemsize)
        buffer.write(bytes_data)
        return buffer

    @_buffer_.finalizer
    @classmethod
    def _buffer_finalizer(
        cls,
        buffer: moderngl.Buffer
    ) -> None:
        cls._VACANT_BUFFERS.append(buffer)


class TexturePlaceholders(GLDynamicStruct):
    __slots__ = ()

    def __init__(
        self,
        *,
        field: str,
        dynamic_array_lens: dict[str, int] | None = None,
        shape: tuple[int, ...] | None = None
    ) -> None:
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        super().__init__(
            field=replaced_field,
            child_structs=None,
            dynamic_array_lens=dynamic_array_lens
        )
        if shape is not None:
            self._shape_ = shape

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.PACKED

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _shape_(cls) -> tuple[int, ...]:
        return ()


class UniformBlockBuffer(GLDynamicBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        fields: list[str],
        child_structs: dict[str, list[str]] | None = None,
        dynamic_array_lens: dict[str, int] | None = None,
        data: np.ndarray | dict[str, Any]
    ) -> None:
        if child_structs is None:
            child_structs = {}
        super().__init__(
            field=f"__UniformBlockStruct__ {name}",
            child_structs={
                "__UniformBlockStruct__": fields,
                **child_structs
            },
            dynamic_array_lens=dynamic_array_lens,
            data=data
        )

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.STD140


class AttributesBuffer(GLDynamicBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        fields: list[str],
        num_vertex: int,
        dynamic_array_lens: dict[str, int] | None = None,
        data: np.ndarray | dict[str, Any],
    ) -> None:
        # Passing structs to an attribute is not allowed, so we eliminate the parameter `child_structs`.
        if dynamic_array_lens is None:
            dynamic_array_lens = {}
        super().__init__(
            field="__VertexStruct__ __vertex__[__NUM_VERTEX__]",
            child_structs={
                "__VertexStruct__": fields
            },
            dynamic_array_lens={
                "__NUM_VERTEX__": num_vertex,
                **dynamic_array_lens
            },
            data=data,
        )

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        # Let's keep using std140 layout, hopefully giving a faster processing speed.
        return GLBufferLayout.STD140

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _vertex_dtype_(
        cls,
        dtype_node__dtype: np.dtype
    ) -> np.dtype:
        return dtype_node__dtype.base


class IndexBuffer(GLDynamicBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        data: VertexIndexType,
    ) -> None:
        super().__init__(
            field="uint __index__[__NUM_INDEX__]",
            child_structs={},
            dynamic_array_lens={
                "__NUM_INDEX__": len(data)
            },
            data=data
        )

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _layout_(cls) -> GLBufferLayout:
        return GLBufferLayout.PACKED
