__all__ = [
    "AttributesBuffer",
    "IndexBuffer",
    "RenderProcedure",
    "TextureStorage",
    "UniformBlockBuffer"
]


import atexit
from dataclasses import dataclass
from functools import reduce
import operator as op
import os
import re
from typing import (
    Any,
    ClassVar,
    Hashable,
)

import moderngl
from moderngl_window.context.pyglet.window import Window as PygletWindow
import numpy as np
from xxhash import xxh3_64_digest

from ..rendering.config import ConfigSingleton
from ..utils.lazy import (
    LazyBase,
    LazyData,
    lazy_basedata,
    lazy_property,
    lazy_slot
)


class ContextSingleton:
    __slots__ = ()

    _INSTANCE: ClassVar[moderngl.Context | None] = None
    _WINDOW: ClassVar[PygletWindow | None] = None
    _WINDOW_FRAMEBUFFER: ClassVar[moderngl.Framebuffer | None] = None

    def __new__(cls) -> moderngl.Context:
        if cls._INSTANCE is not None:
            return cls._INSTANCE
        if ConfigSingleton().preview:
            window = PygletWindow(
                size=ConfigSingleton().window_pixel_size,
                fullscreen=False,
                resizable=True,
                gl_version=(3, 3),
                vsync=True,
                cursor=True
            )
            context = window.ctx
            window_framebuffer = context.detect_framebuffer()
        else:
            window = None
            context = moderngl.create_context(standalone=True)
            window_framebuffer = None
        atexit.register(lambda: context.release())
        cls._WINDOW = window
        cls._INSTANCE = context
        cls._WINDOW_FRAMEBUFFER = window_framebuffer
        return context


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class FieldInfo:
    dtype_str: str
    name: str
    array_shape: list[int | str]


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class DataInfo:
#    data_storage: np.ndarray
#    dynamic_array_lens: dict[str, int]


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
    _LAYOUT: ClassVar[str] = "packed"
    _STRUCT_DTYPE_CACHE: ClassVar[dict[bytes, LazyData[np.dtype]]] = {}
    #_FIELD_INFO_CACHE: ClassVar[dict[str, LazyData[FieldInfo]]] = {}
    #_CHILD_STRUCTS_INFO_CACHE: ClassVar[dict[tuple[tuple[str, ...], ...], LazyData[dict[str, list[FieldInfo]]]]] = {}
    #_DYNAMIC_ARRAY_LENS_CACHE: ClassVar[dict[tuple[tuple[str, int], ...], LazyData[dict[str, int]]]] = {}

    def __new__(
        cls,
        *,
        field: str,
        child_structs: dict[str, list[str]] | None = None,
        dynamic_array_lens: dict[str, int] | None = None,
        data: np.ndarray | dict[str, Any]
    ):
        #instance._data_ = LazyData(data)
        if child_structs is None:
            child_structs = {}
        if dynamic_array_lens is None:
            dynamic_array_lens = {}
        hash_val = Program._hash_items(
            field,
            tuple(
                (name, *child_struct_fields)
                for name, child_struct_fields in child_structs.items()
            ),
            tuple(dynamic_array_lens.items())
        )
        struct_dtype = cls._STRUCT_DTYPE_CACHE.setdefault(hash_val, LazyData(cls._build_struct_dtype(
            [cls._parse_field_str(field)],
            {
                name: [cls._parse_field_str(child_field) for child_field in child_struct_fields]
                for name, child_struct_fields in child_structs.items()
            },
            dynamic_array_lens,
            0
        )))

        assert (field_names := struct_dtype.data.names) is not None
        field_name = field_names[0]
        data_storage = cls._write_data(data, struct_dtype.data, field_name)

        instance = super().__new__(cls)
        instance._struct_dtype_ = struct_dtype
        instance._data_storage_ = LazyData(data_storage)
        instance._dynamic_array_lens = dynamic_array_lens
        instance._field_name = field_name
        return instance

    @lazy_basedata
    @staticmethod
    def _struct_dtype_() -> np.dtype:
        return NotImplemented

    @lazy_basedata
    @staticmethod
    def _data_storage_() -> np.ndarray:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _itemsize_(struct_dtype: np.dtype) -> int:
        return struct_dtype.itemsize

    @lazy_property
    @staticmethod
    def _is_empty_(itemsize: int) -> bool:
        return itemsize == 0

    @lazy_slot
    @staticmethod
    def _dynamic_array_lens() -> dict[str, int]:
        return NotImplemented

    @lazy_slot
    @staticmethod
    def _field_name() -> str:
        return NotImplemented

    @classmethod
    def _write_data(
        cls,
        data: np.ndarray | dict[str, Any],
        struct_dtype: np.dtype,
        field_name: str
    ) -> np.ndarray:
        data_dict = cls._flatten_as_data_dict(data, (field_name,))
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
        dynamic_array_lens: dict[str, int],
        depth: int
    ) -> np.dtype:
        names: list[str] = []
        formats: list[tuple[np.dtype, tuple[int, ...]]] = []
        offsets: list[int] = []

        def add_field_item(name: str, child_dtype: np.dtype, shape: tuple[int, ...], offset: int) -> None:
            names.append(name)
            formats.append((child_dtype, shape))
            offsets.append(offset)

        #dynamic_array_lens: dict[str, int] = {}
        offset = 0

        for field_info in fields_info:
            dtype_str = field_info.dtype_str
            name = field_info.name
            array_shape = field_info.array_shape
            next_depth = depth + len(array_shape)

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
            shape = tuple(
                array_len if isinstance(array_len, int) else dynamic_array_lens[array_len]
                for array_len in array_shape
            )

            if (child_struct_fields_info := child_structs_info.get(dtype_str)) is not None:
                #child_dtype, child_dynamic_array_lens = cls._build_struct_dtype(
                #    child_data, child_struct_fields_info, child_structs_info, next_depth
                #)
                #dynamic_array_lens.update(child_dynamic_array_lens)
                child_dtype = cls._build_struct_dtype(
                    child_struct_fields_info, child_structs_info, dynamic_array_lens, next_depth
                )
                base_alignment = 16
            else:
                atomic_dtype = cls._GLSL_DTYPE[dtype_str]
                #assert len(child_data) == 1 and (data_value := child_data.get(())) is not None
                #if not data_value.size:
                #    continue
                #assert atomic_dtype.shape == data_value.shape[next_depth:]
                child_dtype = cls._align_atomic_dtype(atomic_dtype, not shape)
                base_alignment = child_dtype.base.itemsize

            if cls._LAYOUT == "std140":
                assert child_dtype.itemsize % base_alignment == 0
                offset += (-offset) % base_alignment
            add_field_item(name, child_dtype, shape, offset)
            offset += cls._int_prod(shape) * child_dtype.itemsize

        if cls._LAYOUT == "std140":
            offset += (-offset) % 16

        return np.dtype({
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": offset
        })

    @classmethod
    def _parse_field_str(cls, field_str: str) -> FieldInfo:
        pattern = re.compile(r"""
            (?P<dtype_str>\w+?)
            \s
            (?P<name>\w+?)
            (?P<array_shape>(\[\w+?\])*)
        """, flags=re.VERBOSE)
        match_obj = pattern.fullmatch(field_str)
        assert match_obj is not None
        return FieldInfo(
            dtype_str=match_obj.group("dtype_str"),
            name=match_obj.group("name"),
            array_shape=[
                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else s
                for index_match in re.finditer(r"\[(\w+?)\]", match_obj.group("array_shape"))
            ],
        )

    @classmethod
    def _int_prod(cls, shape: tuple[int, ...]) -> int:
        return reduce(op.mul, shape, 1)

    @classmethod
    def _align_atomic_dtype(cls, atomic_dtype: np.dtype, zero_dimensional: bool) -> np.dtype:
        base = atomic_dtype.base
        shape = atomic_dtype.shape
        assert len(shape) <= 2 and all(2 <= l <= 4 for l in shape)
        shape_dict = dict(enumerate(shape))
        n_col = shape_dict.get(0, 1)
        n_row = shape_dict.get(1, 1)
        if cls._LAYOUT == "packed":
            itemsize = n_row * n_col * base.itemsize
        elif cls._LAYOUT == "std140":
            itemsize = (n_col if zero_dimensional and n_col <= 2 and n_row == 1 else 4) * base.itemsize
        else:
            raise NotImplementedError
        return np.dtype((np.dtype({
            "names": ["_"],
            "formats": [(base, (n_col,))],
            "itemsize": itemsize
        }), (n_row,)))


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

    _LAYOUT = "std140"

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
            data=data
        )

    def _validate(self, uniform_block: moderngl.UniformBlock) -> None:
        assert uniform_block.name == self._field_name
        assert uniform_block.size == self._itemsize_


class AttributesBuffer(GLSLDynamicBuffer):
    __slots__ = ()

    # Let's keep using std140 layout, hopefully leading to a faster processing speed.
    _LAYOUT = "std140"

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
            data=data
        )


class Program(LazyBase):  # TODO: make abstract base class Cachable
    __slots__ = ()

    _CACHE: "ClassVar[dict[bytes, Program]]" = {}

    def __new__(
        cls,
        shader_str: str,
        custom_macros: list[str],
        dynamic_array_lens: dict[str, int],
        texture_storage_shape_dict: dict[str, tuple[int, ...]]
    ):
        # TODO: move function to somewhere suitable
        hash_val = cls._hash_items(shader_str, tuple(custom_macros), cls._dict_as_hashable(dynamic_array_lens))
        cached_instance = cls._CACHE.get(hash_val)
        if cached_instance is not None:
            return cached_instance

        instance = super().__new__(cls)
        moderngl_program = cls._construct_moderngl_program(shader_str, custom_macros, dynamic_array_lens)
        instance._program_ = LazyData(moderngl_program)
        instance._texture_binding_offset_dict_ = LazyData(
            cls._set_texture_bindings(moderngl_program, texture_storage_shape_dict)
        )
        instance._uniform_block_binding_dict_ = LazyData(
            cls._set_uniform_block_bindings(moderngl_program)
        )
        cls._CACHE[hash_val] = instance
        return instance

    @lazy_basedata
    @staticmethod
    def _program_() -> moderngl.Program:
        return NotImplemented

    @lazy_basedata
    @staticmethod
    def _texture_binding_offset_dict_() -> dict[str, int]:
        return NotImplemented

    @lazy_basedata
    @staticmethod
    def _uniform_block_binding_dict_() -> dict[str, int]:
        return NotImplemented

    #@lazy_property
    #@staticmethod
    #def _uniforms_(program: moderngl.Program) -> dict[str, moderngl.Uniform]:
    #    return {
    #        name: member
    #        for name in program
    #        if isinstance(member := program[name], moderngl.Uniform)
    #    }

    #@lazy_property
    #@staticmethod
    #def _uniform_blocks_(program: moderngl.Program) -> dict[str, moderngl.UniformBlock]:
    #    return {
    #        name: member
    #        for name in program
    #        if isinstance(member := program[name], moderngl.UniformBlock)
    #    }

    @lazy_property
    @staticmethod
    def _subroutines_(program: moderngl.Program) -> dict[str, moderngl.Subroutine]:
        return {
            name: member
            for name in program
            if isinstance(member := program[name], moderngl.Subroutine)
        }

    @lazy_property
    @staticmethod
    def _attributes_(program: moderngl.Program) -> dict[str, moderngl.Attribute]:
        return {
            name: member
            for name in program
            if isinstance(member := program[name], moderngl.Attribute)
        }

    @classmethod
    def _construct_moderngl_program(
        cls,
        shader_str: str,
        custom_macros: list[str],
        dynamic_array_lens: dict[str, int]
    ) -> moderngl.Program:
        version_string = f"#version {ContextSingleton().version_code} core"
        array_len_macros = [
            f"#define {array_len_name} {array_len}"
            for array_len_name, array_len in dynamic_array_lens.items()
        ]
        shaders = {
            shader_type: "\n".join([
                version_string,
                "\n",
                f"#define {shader_type}",
                *custom_macros,
                *array_len_macros,
                "\n",
                shader_str
            ])
            for shader_type in (
                "VERTEX_SHADER",
                "FRAGMENT_SHADER",
                "GEOMETRY_SHADER",
                "TESS_CONTROL_SHADER",
                "TESS_EVALUATION_SHADER"
            )
            if re.search(rf"\b{shader_type}\b", shader_str, flags=re.MULTILINE) is not None
        }
        program = ContextSingleton().program(
            vertex_shader=shaders["VERTEX_SHADER"],
            fragment_shader=shaders.get("FRAGMENT_SHADER"),
            geometry_shader=shaders.get("GEOMETRY_SHADER"),
            tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
            tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER"),
        )
        return program

    @classmethod
    def _set_texture_bindings(cls, program: moderngl.Program, texture_storage_shape_dict: dict[str, tuple[int, ...]]) -> dict[str, int]:
        texture_binding_offset_dict: dict[str, int] = {}
        binding_offset = 1
        texture_uniform_match_pattern = re.compile(r"""
            (?P<texture_name>\w+?)
            (?P<multi_index>(\[\d+?\])*)
        """, flags=re.VERBOSE)
        for name in program:
            member = program[name]
            if not isinstance(member, moderngl.Uniform):
                continue
            # Used as a sampler2D
            assert member.dimension == 1
            match_obj = texture_uniform_match_pattern.fullmatch(name)
            assert match_obj is not None
            texture_storage_name = match_obj.group("texture_name")
            texture_storage_shape = texture_storage_shape_dict[texture_storage_name]
            if texture_storage_name not in texture_binding_offset_dict:
                texture_binding_offset_dict[texture_storage_name] = binding_offset
                binding_offset += GLSLDynamicStruct._int_prod(texture_storage_shape)  # TODO
            multi_index = tuple(
                int(index_match.group(1))
                for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
            )
            if not texture_storage_shape:
                assert not multi_index
                uniform_size = 1
                local_offset = 0
            else:
                assert len(multi_index) == len(texture_storage_shape) - 1
                uniform_size = texture_storage_shape[-1]
                local_offset = np.ravel_multi_index(multi_index, texture_storage_shape[:-1]) * uniform_size if multi_index else 0
            assert member.array_length == uniform_size
            offset = texture_binding_offset_dict[texture_storage_name] + local_offset
            member.value = offset if uniform_size == 1 else list(range(offset, offset + uniform_size))
        return texture_binding_offset_dict

    @classmethod
    def _set_uniform_block_bindings(cls, program: moderngl.Program) -> dict[str, int]:
        uniform_block_binding_dict: dict[str, int] = {}
        binding = 0
        for name in program:
            member = program[name]
            if not isinstance(member, moderngl.UniformBlock):
                continue
            assert re.fullmatch(r"\w+", name) is not None
            uniform_block_binding_dict[name] = binding
            member.binding = binding
            binding += 1
        return uniform_block_binding_dict

    @classmethod
    def _hash_items(cls, *items: Hashable) -> bytes:
        return xxh3_64_digest(
            b"".join(
                bytes(hex(hash(item)), encoding="ascii")
                for item in items
            )
        )

    @classmethod
    def _dict_as_hashable(cls, d: dict[Hashable, Hashable]) -> Hashable:
        return tuple(sorted(d.items()))


class IntermediateTexture:
    _CACHE: ClassVar[dict[bytes, list[moderngl.Texture]]] = {}

    def __init__(
        self,
        *,
        size: tuple[int, int],
        components: int,
        samples: int,
        dtype: str
    ):
        hash_val = Program._hash_items(size, components, samples, dtype)  # TODO
        if (cached_instances := self._CACHE.get(hash_val)) is not None and cached_instances:
            texture = cached_instances.pop()
        else:
            texture = ContextSingleton().texture(
                size=size,
                components=components,
                samples=samples,
                dtype=dtype
            )
        self._hash_val: bytes = hash_val
        self._instance: moderngl.Texture = texture

    def __enter__(self) -> moderngl.Texture:
        return self._instance

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_type is None:
            self._CACHE.setdefault(self._hash_val, []).append(self._instance)
        else:
            self._instance.release()
            for cached_instances in self._CACHE.values():
                for texture in cached_instances:
                    texture.release()


class IntermediateDepthTexture:
    _CACHE: ClassVar[dict[bytes, list[moderngl.Texture]]] = {}

    def __init__(
        self,
        *,
        size: tuple[int, int],
        samples: int
    ):
        hash_val = Program._hash_items(size, samples)  # TODO
        if (cached_instances := self._CACHE.get(hash_val)) is not None and cached_instances:
            texture = cached_instances.pop()
        else:
            texture = ContextSingleton().depth_texture(
                size=size,
                samples=samples
            )
        self._hash_val: bytes = hash_val
        self._instance: moderngl.Texture = texture

    def __enter__(self) -> moderngl.Texture:
        return self._instance

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_type is None:
            self._CACHE.setdefault(self._hash_val, []).append(self._instance)
        else:
            self._instance.release()
            for cached_instances in self._CACHE.values():
                for texture in cached_instances:
                    texture.release()


# Create custom BufferBatch classes
class IntermediateFramebuffer:
    def __init__(
        self,
        *,
        color_attachments: list[moderngl.Texture | moderngl.Renderbuffer],
        depth_attachment: moderngl.Texture | moderngl.Renderbuffer | None
    ):
        framebuffer = ContextSingleton().framebuffer(
            color_attachments=tuple(color_attachments),
            depth_attachment=depth_attachment
        )
        framebuffer.clear()
        self._instance: moderngl.Framebuffer = framebuffer

    def __enter__(self) -> moderngl.Framebuffer:
        return self._instance

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._instance.release()


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ContextState:
    enable_only: int
    depth_func: str
    blend_func: tuple[int, int] | tuple[int, int, int, int]
    blend_equation: int | tuple[int, int]
    front_face: str
    cull_face: str
    wireframe: bool


class RenderProcedure(LazyBase):
    __slots__ = ()

    _SHADER_STRS: ClassVar[dict[str, str]] = {}

    def __new__(cls):
        raise NotImplementedError

    def __init_subclass__(cls) -> None:
        raise NotImplementedError

    @classmethod
    def read_shader(cls, filename: str) -> str:
        if (content := cls._SHADER_STRS.get(filename)) is not None:
            return content
        with open(os.path.join(ConfigSingleton().shaders_dir, f"{filename}.glsl")) as shader_file:
            content = shader_file.read()
        cls._SHADER_STRS[filename] = content
        return content

    @classmethod
    def context_state(
        cls,
        *,
        enable_only: int,
        depth_func: str = "<",
        blend_func: tuple[int, int] | tuple[int, int, int, int] = moderngl.DEFAULT_BLENDING,
        blend_equation: int | tuple[int, int] = moderngl.FUNC_ADD,
        front_face: str = "ccw",
        cull_face: str = "back",
        wireframe: bool = False
    ) -> ContextState:
        return ContextState(
            enable_only=enable_only,
            depth_func=depth_func,
            blend_func=blend_func,
            blend_equation=blend_equation,
            front_face=front_face,
            cull_face=cull_face,
            wireframe=wireframe
        )

    @classmethod
    def texture(
        cls,
        *,
        size: tuple[int, int] | None = None,
        components: int = 4,
        samples: int = 0,
        dtype: str = "f1"
    ) -> IntermediateTexture:
        if size is None:
            size = ConfigSingleton().pixel_size
        return IntermediateTexture(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )

    @classmethod
    def depth_texture(
        cls,
        *,
        size: tuple[int, int] | None = None,
        samples: int = 0
    ) -> IntermediateDepthTexture:
        if size is None:
            size = ConfigSingleton().pixel_size
        return IntermediateDepthTexture(
            size=size,
            samples=samples
        )

    @classmethod
    def framebuffer(
        cls,
        *,
        color_attachments: list[moderngl.Texture | moderngl.Renderbuffer],
        depth_attachment: moderngl.Texture | moderngl.Renderbuffer | None
    ) -> IntermediateFramebuffer:
        return IntermediateFramebuffer(
            color_attachments=color_attachments,
            depth_attachment=depth_attachment
        )

    @classmethod
    def downsample_framebuffer(cls, src: moderngl.Framebuffer, dst: moderngl.Framebuffer) -> None:
        ContextSingleton().copy_framebuffer(dst=dst, src=src)

    @classmethod
    def render_step(
        cls,
        *,
        shader_str: str,
        custom_macros: list[str],
        texture_storages: list[TextureStorage],
        uniform_blocks: list[UniformBlockBuffer],
        attributes: AttributesBuffer,
        index_buffer: IndexBuffer,
        framebuffer: moderngl.Framebuffer,
        context_state: ContextState,
        mode: int
    ) -> None:
        #import pprint
        #pprint.pprint(attributes.__class__._field_info_.instance_to_basedata_dict)
        #print(attributes._is_empty_)
        if attributes._is_empty_ or index_buffer._is_empty_:
            return

        dynamic_array_lens: dict[str, int] = {}
        for texture_storage in texture_storages:
            dynamic_array_lens.update(texture_storage._dynamic_array_lens)
        for uniform_block in uniform_blocks:
            dynamic_array_lens.update(uniform_block._dynamic_array_lens)
        dynamic_array_lens.update(attributes._dynamic_array_lens)
        filtered_array_lens = {
            array_len_name: array_len
            for array_len_name, array_len in dynamic_array_lens.items()
            if not re.fullmatch(r"__\w+__", array_len_name)
        }

        context = ContextSingleton()
        program = Program(
            shader_str=shader_str,
            custom_macros=custom_macros,
            dynamic_array_lens=filtered_array_lens,
            texture_storage_shape_dict={
                texture_storage._field_name: texture_storage._texture_array.shape
                for texture_storage in texture_storages
            }
        )

        ## Remove redundancies
        #textures: list[moderngl.Texture] = list(dict.fromkeys(
        #    texture for texture_storage in texture_storages
        #    for texture in texture_storage._texture_list_
        #))

        # texture storages
        texture_storage_dict = {
            texture_storage._field_name: texture_storage
            for texture_storage in texture_storages
        }
        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_storage_name, binding_offset in program._texture_binding_offset_dict_.items():
            texture_storage = texture_storage_dict[texture_storage_name]
            assert not texture_storage._is_empty_
            texture_bindings.extend(
                (texture, binding)
                for binding, texture in enumerate(texture_storage._texture_array.flat, start=binding_offset)
            )

        # uniform blocks
        uniform_block_dict = {
            uniform_block._field_name: uniform_block
            for uniform_block in uniform_blocks
        }
        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
        for uniform_block_name, binding in program._uniform_block_binding_dict_.items():
            uniform_block = uniform_block_dict[uniform_block_name]
            assert not uniform_block._is_empty_
            program_uniform_block = program._program_[uniform_block_name]
            assert isinstance(program_uniform_block, moderngl.UniformBlock)
            uniform_block._validate(program_uniform_block)
            uniform_block_bindings.append((uniform_block._buffer_, binding))

        # subroutines
        #subroutine_indices: list[int] = [
        #    program._subroutines_[subroutines[subroutine_name]].index
        #    for subroutine_name in program._program_.subroutines
        #]

        # attributes
        program_attributes = program._attributes_
        attributes._validate(program_attributes)
        buffer_format, attribute_names = attributes._get_buffer_format(set(program_attributes))
        vertex_array = context.vertex_array(
            program=program._program_,
            content=[(attributes._buffer_, buffer_format, *attribute_names)],
            index_buffer=index_buffer._buffer_,
            mode=mode
        )

        context.depth_func = context_state.depth_func
        context.blend_func = context_state.blend_func
        context.blend_equation = context_state.blend_equation
        context.front_face = context_state.front_face
        context.cull_face = context_state.cull_face
        context.wireframe = context_state.wireframe
        with context.scope(
            framebuffer=framebuffer,
            enable_only=context_state.enable_only,
            textures=tuple(texture_bindings),
            uniform_buffers=tuple(uniform_block_bindings)
        ):
            #vertex_array.subroutines = tuple(subroutine_indices)
            vertex_array.render()

    # TODO
    _FULLSCREEN_ATTRIBUTES: ClassVar[AttributesBuffer | None] = None
    _FULLSCREEN_INDEX_BUFFER: ClassVar[IndexBuffer | None] = None

    @classmethod
    def fullscreen_render_step(
        cls,
        *,
        shader_str: str,
        custom_macros: list[str],
        texture_storages: list[TextureStorage],
        uniform_blocks: list[UniformBlockBuffer],
        framebuffer: moderngl.Framebuffer,
        context_state: ContextState
    ) -> None:
        if cls._FULLSCREEN_ATTRIBUTES is None:
            cls._FULLSCREEN_ATTRIBUTES = AttributesBuffer(
                fields=[
                    "vec3 in_position",
                    "vec2 in_uv"
                ],
                num_vertex=4,
                data={
                    "in_position": np.array((
                        [-1.0, -1.0, 0.0],
                        [1.0, -1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [-1.0, 1.0, 0.0],
                    )),
                    "in_uv": np.array((
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                    ))
                }
            )
        if cls._FULLSCREEN_INDEX_BUFFER is None:
            cls._FULLSCREEN_INDEX_BUFFER = IndexBuffer(
                data=np.array((
                    0, 1, 2, 3
                ))
            )
        cls.render_step(
            shader_str=shader_str,
            custom_macros=custom_macros,
            texture_storages=texture_storages,
            uniform_blocks=uniform_blocks,
            attributes=cls._FULLSCREEN_ATTRIBUTES,
            index_buffer=cls._FULLSCREEN_INDEX_BUFFER,
            framebuffer=framebuffer,
            context_state=context_state,
            mode=moderngl.TRIANGLE_FAN
        )
