__all__ = [
    "AttributesBuffer",
    "IndexBuffer",
    "TextureStorage",
    "UniformBlockBuffer"
]


from dataclasses import dataclass
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

from ..rendering.context import ContextSingleton
#from ..utils.cache import CacheUtils
from ..utils.lazy import (
    LazyBase,
    NewData,
    lazy_basedata,
    lazy_basedata_cached,
    lazy_property,
    lazy_slot
)


class GLSLVariableLayout(Enum):
    PACKED = 0
    STD140 = 1


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class FieldInfo:
    dtype_str: str
    name: str
    shape: tuple[int, ...]


class GLSLDynamicStruct(LazyBase):
    __slots__ = ()

    _GLSL_DTYPE: ClassVar[dict[str, np.dtype]] = {
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
        "double":  np.dtype(("f8", ())),
        "dvec2":   np.dtype(("f8", (2,))),
        "dvec3":   np.dtype(("f8", (3,))),
        "dvec4":   np.dtype(("f8", (4,))),
        "mat2":    np.dtype(("f4", (2, 2))),
        "mat2x3":  np.dtype(("f4", (2, 3))),  # TODO: check order
        "mat2x4":  np.dtype(("f4", (2, 4))),
        "mat3x2":  np.dtype(("f4", (3, 2))),
        "mat3":    np.dtype(("f4", (3, 3))),
        "mat3x4":  np.dtype(("f4", (3, 4))),
        "mat4x2":  np.dtype(("f4", (4, 2))),
        "mat4x3":  np.dtype(("f4", (4, 3))),
        "mat4":    np.dtype(("f4", (4, 4))),
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
    #_LAYOUT: ClassVar[GLSLVariableLayout] = GLSLVariableLayout.PACKED
    #_STRUCT_DTYPE_CACHE: ClassVar[dict[bytes, NewData[np.dtype]]] = {}

    def __new__(
        cls,
        *,
        field: str,
        child_structs: dict[str, list[str]] | None = None,
        dynamic_array_lens: dict[str, int] | None = None,
        layout: GLSLVariableLayout,
        data: np.ndarray | dict[str, Any]
    ):
        if child_structs is None:
            child_structs = {}
        if dynamic_array_lens is None:
            dynamic_array_lens = {}
        instance = super().__new__(cls)
        instance._field_ = field
        instance._child_structs_ = child_structs
        instance._dynamic_array_lens_ = dynamic_array_lens
        instance._layout_ = layout
        instance._data_ = NewData(data)
        #hash_val = CacheUtils.hash_items(
        #    field,
        #    tuple(
        #        (name, *child_struct_fields)
        #        for name, child_struct_fields in child_structs.items()
        #    ),
        #    tuple(dynamic_array_lens.items())
        #)
        #if (struct_dtype := cls._STRUCT_DTYPE_CACHE.get(hash_val)) is None:
        #    struct_dtype = NewData(cls._build_struct_dtype(
        #        [cls._parse_field_str(field)],
        #        {
        #            name: [cls._parse_field_str(child_field) for child_field in child_struct_fields]
        #            for name, child_struct_fields in child_structs.items()
        #        },
        #        dynamic_array_lens,
        #        0
        #    ))
        #    cls._STRUCT_DTYPE_CACHE[hash_val] = struct_dtype

        #assert (field_names := struct_dtype.data.names) is not None
        #field_name = field_names[0]
        #data_storage = cls._write_data(data, struct_dtype.data, field_name)

        #instance._struct_dtype_ = struct_dtype
        #instance._data_storage_ = NewData(data_storage)
        #instance._dynamic_array_lens = dynamic_array_lens
        #instance._field_name = field_name
        return instance

    @staticmethod
    def __field_cacher(
        field: str
    ) -> str:
        return field

    @lazy_basedata_cached(__field_cacher)
    @staticmethod
    def _field_() -> str:
        return NotImplemented

    @staticmethod
    def __child_structs_cacher(
        child_structs: dict[str, list[str]]
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (name, tuple(child_struct_fields))
            for name, child_struct_fields in child_structs.items()
        )

    @lazy_basedata_cached(__child_structs_cacher)
    @staticmethod
    def _child_structs_() -> dict[str, list[str]]:
        return NotImplemented

    @staticmethod
    def __dynamic_array_lens_cacher(
        dynamic_array_lens: dict[str, int]
    ) -> tuple[tuple[str, int], ...]:
        return tuple(dynamic_array_lens.items())

    @lazy_basedata_cached(__dynamic_array_lens_cacher)
    @staticmethod
    def _dynamic_array_lens_() -> dict[str, int]:
        return NotImplemented

    @staticmethod
    def __layout_cacher(
        layout: GLSLVariableLayout
    ) -> GLSLVariableLayout:
        return layout

    @lazy_basedata_cached(__layout_cacher)
    @staticmethod
    def _layout_() -> GLSLVariableLayout:
        return NotImplemented

    @lazy_basedata
    @staticmethod
    def _data_() -> np.ndarray | dict[str, Any]:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _struct_dtype_(
        field: str,
        child_structs: dict[str, list[str]],
        dynamic_array_lens: dict[str, int],
        layout: GLSLVariableLayout
    ) -> np.dtype:
        return GLSLDynamicStruct._build_struct_dtype(
            [GLSLDynamicStruct._parse_field_str(field, dynamic_array_lens)],
            {
                name: [
                    GLSLDynamicStruct._parse_field_str(child_field, dynamic_array_lens)
                    for child_field in child_struct_fields
                ]
                for name, child_struct_fields in child_structs.items()
            },
            layout,
            0
        )

    @lazy_property
    @staticmethod
    def _field_name_(
        struct_dtype: np.dtype
    ) -> str:
        assert (field_names := struct_dtype.names) is not None
        return field_names[0]

    @lazy_property
    @staticmethod
    def _data_storage_(
        data: np.ndarray | dict[str, Any],
        struct_dtype: np.dtype,
        field_name: str
    ) -> np.ndarray:
        data_dict = GLSLDynamicStruct._flatten_as_data_dict(data, (field_name,))
        data_storage = np.zeros((), dtype=struct_dtype)
        for data_key, data_value in data_dict.items():
            if not data_value.size:
                continue
            data_ptr = data_storage
            while data_key:
                data_ptr = data_ptr[data_key[0]]
                data_key = data_key[1:]
            data_ptr["_"] = data_value.reshape(data_ptr["_"].shape)  # TODO
        return data_storage

    @lazy_property
    @staticmethod
    def _itemsize_(struct_dtype: np.dtype) -> int:
        return struct_dtype.itemsize

    @lazy_property
    @staticmethod
    def _is_empty_(itemsize: int) -> bool:
        return itemsize == 0

    @classmethod
    def _flatten_as_data_dict(
        cls,
        data: np.ndarray | dict[str, Any],
        prefix: tuple[str, ...]
    ) -> dict[tuple[str, ...], np.ndarray]:
        if isinstance(data, np.ndarray):
            return {prefix: data}
        result: dict[tuple[str, ...], np.ndarray] = {}
        for child_name, child_data in data.items():
            result.update(cls._flatten_as_data_dict(child_data, prefix + (child_name,)))
        return result

    @classmethod
    def _build_struct_dtype(
        cls,
        fields_info: list[FieldInfo],
        child_structs_info: dict[str, list[FieldInfo]],
        layout: GLSLVariableLayout,
        depth: int
    ) -> np.dtype:
        names: list[str] = []
        formats: list[tuple[np.dtype, tuple[int, ...]]] = []
        offsets: list[int] = []

        #dynamic_array_lens: dict[str, int] = {}
        offset = 0

        for field_info in fields_info:
            dtype_str = field_info.dtype_str
            name = field_info.name
            shape = field_info.shape
            next_depth = depth + len(shape)

            #child_data: dict[tuple[str, ...], np.ndarray] = {}
            #node_dynamic_array_lens: dict[str, int] = {}
            #for data_key, data_value in data_dict.items():
            #    if data_key[0] != name:
            #        continue
            #    if data_value.size:
            #        data_array_shape = data_value.shape[depth:next_depth]
            #    else:
            #        data_array_shape = tuple(0 for _ in array_shape)
            #    for array_len, data_array_len in zip(array_shape, data_array_shape, strict=True):
            #        if isinstance(array_len, int):
            #            assert array_len == data_array_len
            #        else:
            #            # Rewrite if the recorded array length is 0
            #            if node_dynamic_array_lens.get(array_len, 0) and data_array_len:
            #                assert node_dynamic_array_lens[array_len] == data_array_len
            #            else:
            #                node_dynamic_array_lens[array_len] = data_array_len
            #    child_data[data_key[1:]] = data_value
            #dynamic_array_lens.update(node_dynamic_array_lens)
            #shape = tuple(
            #    array_len if isinstance(array_len, int) else node_dynamic_array_lens[array_len]
            #    for array_len in array_shape
            #)
            #shape = tuple(
            #    array_len if isinstance(array_len, int) else dynamic_array_lens[array_len]
            #    for array_len in array_shape
            #)

            if (child_struct_fields_info := child_structs_info.get(dtype_str)) is not None:
                #child_dtype, child_dynamic_array_lens = cls._build_struct_dtype(
                #    child_data, child_struct_fields_info, child_structs_info, next_depth
                #)
                #dynamic_array_lens.update(child_dynamic_array_lens)
                child_dtype = cls._build_struct_dtype(
                    child_struct_fields_info, child_structs_info, layout, next_depth
                )
                base_alignment = 16
            else:
                atomic_dtype = cls._GLSL_DTYPE[dtype_str]
                #assert len(child_data) == 1 and (data_value := child_data.get(())) is not None
                #if not data_value.size:
                #    continue
                #assert atomic_dtype.shape == data_value.shape[next_depth:]
                child_dtype = cls._align_atomic_dtype(atomic_dtype, layout, not shape)
                base_alignment = child_dtype.base.itemsize

            if layout == GLSLVariableLayout.STD140:
                assert child_dtype.itemsize % base_alignment == 0
                offset += (-offset) % base_alignment
            names.append(name)
            formats.append((child_dtype, shape))
            offsets.append(offset)
            offset += cls._int_prod(shape) * child_dtype.itemsize

        if layout == GLSLVariableLayout.STD140:
            offset += (-offset) % 16

        return np.dtype({
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": offset
        })

    @classmethod
    def _parse_field_str(cls, field_str: str, dynamic_array_lens: dict[str, int]) -> FieldInfo:
        pattern = re.compile(r"""
            (?P<dtype_str>\w+?)
            \s
            (?P<name>\w+?)
            (?P<shape>(\[\w+?\])*)
        """, flags=re.VERBOSE)
        match_obj = pattern.fullmatch(field_str)
        assert match_obj is not None
        return FieldInfo(
            dtype_str=match_obj.group("dtype_str"),
            name=match_obj.group("name"),
            shape=tuple(
                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else dynamic_array_lens[s]
                for index_match in re.finditer(r"\[(\w+?)\]", match_obj.group("shape"))
            ),
        )

    @classmethod
    def _align_atomic_dtype(cls, atomic_dtype: np.dtype, layout: GLSLVariableLayout, zero_dimensional: bool) -> np.dtype:
        base = atomic_dtype.base
        shape = atomic_dtype.shape
        assert len(shape) <= 2 and all(2 <= l <= 4 for l in shape)
        shape_dict = dict(enumerate(shape))
        n_col = shape_dict.get(0, 1)
        n_row = shape_dict.get(1, 1)
        if layout == GLSLVariableLayout.STD140:
            itemsize = (n_col if zero_dimensional and n_col <= 2 and n_row == 1 else 4) * base.itemsize
        else:
            itemsize = n_col * base.itemsize
        return np.dtype((np.dtype({
            "names": ["_"],
            "formats": [(base, (n_col,))],
            "itemsize": itemsize
        }), (n_row,)))

    @classmethod
    def _int_prod(cls, shape: tuple[int, ...]) -> int:
        return reduce(op.mul, shape, 1)


class GLSLDynamicBuffer(GLSLDynamicStruct):
    __slots__ = ()

    _BUFFER_CACHE: list[moderngl.Buffer] = []

    @lazy_property
    @staticmethod
    def _buffer_(
        data_storage: np.ndarray,
        struct_dtype: np.dtype
    ) -> moderngl.Buffer:
        if GLSLDynamicBuffer._BUFFER_CACHE:
            buffer = GLSLDynamicBuffer._BUFFER_CACHE.pop()
        else:
            buffer = ContextSingleton().buffer(reserve=1, dynamic=True)  # TODO: dynamic?

        bytes_data = data_storage.tobytes()
        #assert struct_dtype.itemsize == len(bytes_data)
        if struct_dtype.itemsize == 0:
            buffer.clear()
            return buffer
        buffer.orphan(struct_dtype.itemsize)
        buffer.write(bytes_data)
        return buffer

    @_buffer_.restocker
    @staticmethod
    def _buffer_restocker(buffer: moderngl.Buffer) -> None:
        GLSLDynamicBuffer._BUFFER_CACHE.append(buffer)


class TextureStorage(GLSLDynamicStruct):
    __slots__ = ()

    def __new__(
        cls,
        *,
        field: str,
        dynamic_array_lens: dict[str, int] | None = None,
        texture_array: np.ndarray
    ):
        # Note, redundant textures are currently not supported
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        instance = super().__new__(
            cls,
            field=replaced_field,
            dynamic_array_lens=dynamic_array_lens,
            layout=GLSLVariableLayout.PACKED,
            data=np.zeros(texture_array.shape, dtype=np.uint32)
        )
        instance._texture_array = texture_array
        return instance

    @lazy_slot
    @staticmethod
    def _texture_array() -> np.ndarray:
        return NotImplemented


class UniformBlockBuffer(GLSLDynamicBuffer):
    __slots__ = ()

    def __new__(
        cls,
        *,
        name: str,
        fields: list[str],
        child_structs: dict[str, list[str]] | None = None,
        dynamic_array_lens: dict[str, int] | None = None,
        data: np.ndarray | dict[str, Any]
    ):
        if child_structs is None:
            child_structs = {}
        if dynamic_array_lens is None:
            dynamic_array_lens = {}
        return super().__new__(
            cls,
            field=f"__UniformBlockStruct__ {name}",
            child_structs={
                "__UniformBlockStruct__": fields,
                **child_structs
            },
            dynamic_array_lens=dynamic_array_lens,
            layout=GLSLVariableLayout.STD140,
            data=data
        )

    def _validate(self, uniform_block: moderngl.UniformBlock) -> None:
        assert uniform_block.name == self._field_name_
        assert uniform_block.size == self._itemsize_


class AttributesBuffer(GLSLDynamicBuffer):
    __slots__ = ()

    def __new__(
        cls,
        *,
        fields: list[str],
        num_vertex: int,
        dynamic_array_lens: dict[str, int] | None = None,
        data: np.ndarray | dict[str, Any],
    ):
        # Passing structs to an attribute is not allowed, so we eliminate the parameter `child_structs`.
        if dynamic_array_lens is None:
            dynamic_array_lens = {}
        dynamic_array_lens["__NUM_VERTEX__"] = num_vertex
        return super().__new__(
            cls,
            field="__VertexStruct__ __vertex__[__NUM_VERTEX__]",
            child_structs={
                "__VertexStruct__": fields
            },
            dynamic_array_lens=dynamic_array_lens,
            # Let's keep using std140 layout, hopefully leading to a faster processing speed.
            layout=GLSLVariableLayout.STD140,
            data=data,
        )

    @lazy_property
    @staticmethod
    def _vertex_dtype_(struct_dtype: np.dtype) -> np.dtype:
        return struct_dtype["__vertex__"].base

    def _get_buffer_format(self, attribute_name_set: set[str]) -> tuple[str, list[str]]:
        # TODO: This may require refactory
        vertex_dtype = self._vertex_dtype_
        vertex_fields = vertex_dtype.fields
        assert vertex_fields is not None
        dtype_stack: list[tuple[np.dtype, int]] = []
        attribute_names: list[str] = []
        for field_name, (field_dtype, field_offset, *_) in vertex_fields.items():
            if field_name not in attribute_name_set:
                continue
            dtype_stack.append((field_dtype, field_offset))
            attribute_names.append(field_name)

        components: list[str] = []
        current_offset = 0
        while dtype_stack:
            dtype, offset = dtype_stack.pop(0)
            dtype_size = self._int_prod(dtype.shape)
            dtype_itemsize = dtype.base.itemsize
            if dtype.base.fields is not None:
                dtype_stack = [
                    (child_dtype, offset + i * dtype_itemsize + child_offset)
                    for i in range(dtype_size)
                    for child_dtype, child_offset, *_ in dtype.base.fields.values()
                ] + dtype_stack
                continue
            if current_offset != offset:
                components.append(f"{offset - current_offset}x")
                current_offset = offset
            components.append(f"{dtype_size}{dtype.base.kind}{dtype_itemsize}")
            current_offset += dtype_size * dtype_itemsize
        if current_offset != vertex_dtype.itemsize:
            components.append(f"{vertex_dtype.itemsize - current_offset}x")
        components.append("/v")
        return " ".join(components), attribute_names

    def _validate(self, attributes: dict[str, moderngl.Attribute]) -> None:
        vertex_dtype = self._vertex_dtype_
        for attribute_name, attribute in attributes.items():
            field_dtype = vertex_dtype[attribute_name]
            #if any(atomic_dtype is field_dtype for atomic_dtype in self._GLSL_DTYPE.values()):
            #    array_shape = ()
            #    atomic_dtype = field_dtype
            #else:
            #    array_shape = field_dtype.shape
            #    atomic_dtype = field_dtype.base
            assert attribute.array_length == self._int_prod(field_dtype.shape)
            assert attribute.dimension == self._int_prod(field_dtype.base.shape) * self._int_prod(field_dtype.base["_"].shape)
            assert attribute.shape == field_dtype.base["_"].base.kind.replace("u", "I")


class IndexBuffer(GLSLDynamicBuffer):
    __slots__ = ()

    def __new__(
        cls,
        *,
        data: np.ndarray,
    ):
        return super().__new__(
            cls,
            field="uint __index__[__NUM_INDEX__]",
            dynamic_array_lens={
                "__NUM_INDEX__": len(data)
            },
            layout=GLSLVariableLayout.PACKED,
            data=data
        )
