__all__ = [
    "AttributesBuffer",
    "ContextState",
    "Framebuffer",
    "IndexBuffer",
    "IntermediateDepthTextures",
    "IntermediateFramebuffer",
    "IntermediateTextures",
    "RenderStep",
    "Renderable",
    "UniformBlockBuffer"
]


from abc import (
    ABC,
    abstractmethod
)
#from contextlib import contextmanager
from dataclasses import dataclass
from functools import (
    lru_cache,
    reduce
)
import operator as op
import os
import re
from typing import (
    Any,
    ClassVar,
    #Generator,
    Generic,
    Hashable,
    TypeVar
)

import moderngl
import numpy as np
from xxhash import xxh3_64_digest

from ..constants import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    SHADERS_PATH
)
from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)


_T = TypeVar("_T")


class ResourceFactory(Generic[_T], ABC):
    __HASHES__: dict[_T, bytes]
    __VACANT__: dict[bytes, list[_T]]
    #__OCCUPIED__: list[_T]
    #__OCCUPIED__: dict[Self, bytes]
    #__VACANT__: dict[bytes, list[Self]]
    #__RESOURCE_GENERATOR__: Generator[_T, None, None]

    def __init_subclass__(cls) -> None:
        #cls.__OCCUPIED__ = []
        cls.__HASHES__ = {}
        cls.__VACANT__ = {}
        #cls.__RESOURCE_GENERATOR__ = cls._generate()
        super().__init_subclass__()

    @classmethod
    def fetch(cls, **kwargs):
        hash_val = cls._hash_items(cls._dict_as_hashable(kwargs))
        if (vacant_list := cls.__VACANT__.get(hash_val)) is not None and vacant_list:
            instance = vacant_list.pop()
            #cls._reset(instance)
            return instance
        instance = cls._construct(**kwargs)
        cls.__HASHES__[instance] = hash_val
        return instance

    @classmethod
    def restore(cls, instance: _T) -> None:
        hash_val = cls.__HASHES__[instance]
        if hash_val not in cls.__VACANT__:
            cls.__VACANT__[hash_val] = []
        cls.__VACANT__[hash_val].append(instance)

    @classmethod
    @abstractmethod
    def _construct(cls, **kwargs) -> _T:
        pass

    #@classmethod
    #def _reset(cls, instance: _T) -> None:
    #    pass

    #@classmethod
    #@abstractmethod
    #def _generate(cls) -> Generator[_T, None, None]:
    #    pass

    #@classmethod
    #def _reset(cls, resource: _T) -> None:
    #    pass

    #@classmethod
    #def _release(cls, resource: _T) -> None:
    #    pass

    #@classmethod
    #def _register_enter(cls) -> _T:
    #    if cls.__VACANT__:
    #        resource = cls.__VACANT__.pop(0)
    #    else:
    #        try:
    #            resource = next(cls.__RESOURCE_GENERATOR__)
    #        except StopIteration:
    #            raise MemoryError(f"{cls.__name__} cannot allocate a new object") from None
    #    cls.__OCCUPIED__.append(resource)
    #    return resource

    #@classmethod
    #def _register_exit(cls, resource: _T) -> None:
    #    cls.__OCCUPIED__.remove(resource)
    #    #cls._reset(resource)
    #    cls.__VACANT__.append(resource)

    #@classmethod
    #@contextmanager
    #def register(cls) -> Generator[_T, None, None]:
    #    resource = cls._register_enter()
    #    try:
    #        yield resource
    #    finally:
    #        cls._register_exit(resource)

    #@classmethod
    #@contextmanager
    #def register_n(cls, n: int) -> Generator[list[_T], None, None]:
    #    resource_list = [
    #        cls._register_enter()
    #        for _ in range(n)
    #    ]
    #    try:
    #        yield resource_list
    #    finally:
    #        for resource in resource_list:
    #            cls._register_exit(resource)

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

    #@classmethod
    #def _release_all(cls) -> None:
    #    while cls.__OCCUPIED__:
    #        resource = cls.__OCCUPIED__.pop()
    #        cls._release(resource)
    #    while cls.__VACANT__:
    #        resource = cls.__VACANT__.pop()
    #        cls._release(resource)


#class TextureBindings(ResourceFactory[int]):
#    @classmethod
#    def _generate(cls) -> Generator[int, None, None]:
#        for texture_binding in range(1, 32):  # TODO
#            yield texture_binding


#class UniformBindings(ResourceFactory[int]):
#    @classmethod
#    def _generate(cls) -> Generator[int, None, None]:
#        for uniform_binding in range(64):  # TODO
#            yield uniform_binding


class IntermediateTextures(ResourceFactory[moderngl.Texture]):
    @classmethod
    def _construct(
        cls,
        size: tuple[int, int] = (PIXEL_WIDTH, PIXEL_HEIGHT),
        components: int = 4,
        dtype: str = "f1"
    ) -> moderngl.Texture:
        return ContextSingleton().texture(
            size=size,
            components=components,
            dtype=dtype
        )


class IntermediateDepthTextures(ResourceFactory[moderngl.Texture]):
    @classmethod
    def _construct(
        cls,
        size: tuple[int, int] = (PIXEL_WIDTH, PIXEL_HEIGHT)
    ) -> moderngl.Texture:
        return ContextSingleton().depth_texture(
            size=size,
            data=np.ones((PIXEL_WIDTH, PIXEL_HEIGHT), dtype=np.float32).tobytes()
        )


#class IntermediateTextures(ResourceFactory[moderngl.Texture]):
#    @classmethod
#    def _generate(cls) -> Generator[moderngl.Texture, None, None]:
#        while True:
#            yield ContextSingleton().texture(
#                size=(PIXEL_WIDTH, PIXEL_HEIGHT),
#                components=4
#            )

    #@classmethod
    #def _release(cls, resource: moderngl.Texture) -> None:
    #    resource.release()


#class IntermediateDepthTextures(IntermediateTextures):
#    @classmethod
#    def _generate(cls) -> Generator[moderngl.Texture, None, None]:
#        while True:
#            # Initialized as ones (far clip plane)
#            yield ContextSingleton().depth_texture(
#                size=(PIXEL_WIDTH, PIXEL_HEIGHT),
#                data=np.ones((PIXEL_WIDTH, PIXEL_HEIGHT), dtype=np.float32).tobytes()
#            )


@dataclass
class FieldInfo:
    dtype_str: str
    name: str
    array_shape: list[int | str]


class GLSLBuffer(LazyBase):
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

    def __init__(
        self,
        field: str,
        child_structs: dict[str, list[str]] | None = None
    ):
        super().__init__()
        if child_structs is None:
            child_structs = {}
        self.field_info: FieldInfo = self._parse_field_str(field)
        self.child_structs_info: dict[str, list[FieldInfo]] = {
            name: [self._parse_field_str(child_field) for child_field in child_struct_fields]
            for name, child_struct_fields in child_structs.items()
        }

    def __del__(self):
        pass  # TODO: release buffer

    @lazy_property_initializer
    @staticmethod
    def _buffer_() -> moderngl.Buffer:
        return ContextSingleton().buffer(reserve=1, dynamic=True)  # TODO: dynamic?

    @lazy_property_initializer_writable
    @staticmethod
    def _struct_dtype_() -> np.dtype:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _dynamic_array_lens_() -> dict[str, int]:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _itemsize_(struct_dtype: np.dtype) -> int:
        return struct_dtype.itemsize

    def _is_empty(self) -> bool:
        return self._itemsize_ == 0

    @_buffer_.updater
    def write(self, data: np.ndarray | dict[str, Any]):
        data_dict = self._flatten_as_data_dict(data, (self.field_info.name,))
        struct_dtype, dynamic_array_lens = self._build_struct_dtype(
            [self.field_info], self.child_structs_info, data_dict, 0
        )
        data_storage = np.zeros((), dtype=struct_dtype)
        for data_key, data_value in data_dict.items():
            if not data_value.size:
                continue
            last_field = data_key[-1]
            data_key = data_key[:-1]
            data_ptr = data_storage
            while data_key:
                data_ptr = data_ptr[data_key[0]]
                data_key = data_key[1:]
            data_ptr[last_field] = data_value
        self._struct_dtype_ = struct_dtype
        self._dynamic_array_lens_ = dynamic_array_lens

        bytes_data = data_storage.tobytes()
        assert struct_dtype.itemsize == len(bytes_data)
        buffer = self._buffer_
        if struct_dtype.itemsize == 0:
            buffer.clear()
            return self
        #print(struct_dtype.itemsize)
        buffer.orphan(struct_dtype.itemsize)
        buffer.write(bytes_data)
        return self

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
        data_dict: dict[tuple[str, ...], np.ndarray],
        depth: int
    ) -> tuple[np.dtype, dict[str, int]]:
        field_items: list[tuple[str, np.dtype, tuple[int, ...], int]] = []
        dynamic_array_lens: dict[str, int] = {}
        offset = 0

        for field_info in fields_info:
            dtype_str = field_info.dtype_str
            name = field_info.name
            array_shape = field_info.array_shape
            next_depth = depth + len(array_shape)

            child_data: dict[tuple[str, ...], np.ndarray] = {}
            node_dynamic_array_lens: dict[str, int] = {}
            for data_key, data_value in data_dict.items():
                if data_key[0] != name:
                    continue
                if data_value.size:
                    data_array_shape = data_value.shape[depth:next_depth]
                else:
                    data_array_shape = tuple(0 for _ in array_shape)
                for array_len, data_array_len in zip(array_shape, data_array_shape, strict=True):
                    if isinstance(array_len, int):
                        assert array_len == data_array_len
                    else:
                        # Rewrite if the recorded array length is 0
                        if node_dynamic_array_lens.get(array_len, 0) and data_array_len:
                            assert node_dynamic_array_lens[array_len] == data_array_len
                        else:
                            node_dynamic_array_lens[array_len] = data_array_len
                child_data[data_key[1:]] = data_value
            dynamic_array_lens.update(node_dynamic_array_lens)
            shape = tuple(
                array_len if isinstance(array_len, int) else node_dynamic_array_lens[array_len]
                for array_len in array_shape
            )

            if (child_struct_fields_info := child_structs_info.get(dtype_str)) is not None:
                child_dtype, child_dynamic_array_lens = cls._build_struct_dtype(
                    child_struct_fields_info, child_structs_info, child_data, next_depth
                )
                dynamic_array_lens.update(child_dynamic_array_lens)
                base_alignment, pad_tail = cls._get_component_alignment(None, shape)
            else:
                child_dtype = cls._GLSL_DTYPE[dtype_str]
                assert len(child_data) == 1 and (data_value := child_data.get(())) is not None
                if not data_value.size:
                    continue
                assert child_dtype.shape == data_value.shape[next_depth:]
                base_alignment, pad_tail = cls._get_component_alignment(child_dtype, shape)

            if base_alignment:
                offset += (-offset) % base_alignment
            field_items.append((name, child_dtype, shape, offset))
            offset += cls._int_prod(shape) * child_dtype.itemsize
            if pad_tail and base_alignment:
                offset += (-offset) % base_alignment

        if not field_items:
            dtype = np.dtype([])
        else:
            names, child_dtypes, shapes, offsets = zip(*field_items)
            dtype = np.dtype({
                "names": names,
                "formats": [(child_dtype, shape) for child_dtype, shape in zip(child_dtypes, shapes)],
                "offsets": offsets,
                "itemsize": offset
            })  # type: ignore
        return dtype, dynamic_array_lens

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
    def _get_component_alignment(cls, child_dtype: np.dtype | None, shape: tuple[int, ...]) -> tuple[int, bool]:
        if cls._LAYOUT == "packed":
            base_alignment = 0
            pad_tail = False
        elif cls._LAYOUT == "std140":
            if child_dtype is None:
                base_alignment = 16
                pad_tail = True
            elif not shape and len(child_dtype.shape) <= 1:
                if child_dtype.shape == ():
                    base_alignment_units = 1
                elif child_dtype.shape == (2,):
                    base_alignment_units = 2
                elif child_dtype.shape == (3,):
                    base_alignment_units = 4
                elif child_dtype.shape == (4,):
                    base_alignment_units = 4
                else:
                    raise
                base_alignment = base_alignment_units * child_dtype.base.itemsize
                pad_tail = False
            else:
                base_alignment = 4 * child_dtype.base.itemsize
                pad_tail = True
        else:
            raise NotImplementedError
        return base_alignment, pad_tail


class TextureStorage(GLSLBuffer):
    def __init__(self, field: str):
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        super().__init__(field=replaced_field)

    @lazy_property_initializer_writable
    @staticmethod
    def _texture_array_() -> np.ndarray:
        return NotImplemented

    def write(self, texture_array: np.ndarray):
        # Note, redundant textures are currently not supported
        self._texture_array_ = texture_array
        #texture_list: list[moderngl.Texture] = []
        ## Remove redundancies
        #for texture in textures.flatten():
        #    assert isinstance(texture, moderngl.Texture)
        #    if texture not in texture_list:
        #        texture_list.append(texture)
        #self._texture_list_ = texture_list
        #data = np.vectorize(lambda texture: texture_list.index(texture), otypes=[np.uint32])(textures)
        super().write(np.zeros(texture_array.shape, dtype=np.uint32))
        return self

    #def _get_indices(self) -> list[int]:
    #    return list(np.frombuffer(self._buffer_.read(), np.uint32))

    #def _validate(self, uniform: moderngl.Uniform) -> None:
    #    assert uniform.name == self.field_info.name
    #    assert uniform.dimension == 1
    #    assert uniform.array_length == self._int_prod(self._struct_dtype_[self.field_info.name].shape)


class UniformBlockBuffer(GLSLBuffer):
    _LAYOUT = "std140"

    def __init__(self, name: str, fields: list[str], child_structs: dict[str, list[str]] | None = None):
        if child_structs is None:
            child_structs = {}
        super().__init__(
            field=f"__UniformBlockStruct__ {name}",
            child_structs={
                "__UniformBlockStruct__": fields,
                **child_structs
            }
        )

    def _validate(self, uniform_block: moderngl.UniformBlock) -> None:
        assert uniform_block.name == self.field_info.name
        assert uniform_block.size == self._itemsize_


class AttributesBuffer(GLSLBuffer):
    # Let's keep using std140 layout, hopefully leading to a faster processing speed
    _LAYOUT = "std140"

    def __init__(self, attributes: list[str], child_structs: dict[str, list[str]] | None = None):
        if child_structs is None:
            child_structs = {}
        super().__init__(
            field="__VertexStruct__ __vertex__[__NUM_VERTEX__]",
            child_structs={
                "__VertexStruct__": attributes,
                **child_structs
            }
        )

    @lazy_property
    @staticmethod
    def _vertex_dtype_(struct_dtype: np.dtype) -> np.dtype:
        return struct_dtype["__vertex__"].base

    def _get_buffer_format(self, attribute_name_set: set[str]) -> tuple[str, list[str]]:
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
            if any(glsl_dtype is field_dtype for glsl_dtype in self._GLSL_DTYPE.values()):
                array_shape = ()
                atom_dtype = field_dtype
            else:
                array_shape = field_dtype.shape
                atom_dtype = field_dtype.base
            assert attribute.array_length == self._int_prod(array_shape)
            assert attribute.dimension == self._int_prod(atom_dtype.shape)
            assert attribute.shape == atom_dtype.base.kind.replace("u", "I")


class IndexBuffer(GLSLBuffer):
    def __init__(self):
        super().__init__(field="uint __index__[__NUM_INDEX__]")


class Framebuffer:
    def __init__(self, framebuffer: moderngl.Framebuffer):
        self._framebuffer: moderngl.Framebuffer = framebuffer

    def clear(
        self,
        red: float = 0.0,
        green: float = 0.0,
        blue: float = 0.0,
        alpha: float = 0.0,
        depth: float = 1.0,
    ) -> None:
        self._framebuffer.clear(
            red=red,
            green=green,
            blue=blue,
            alpha=alpha,
            depth=depth
        )

    def release(self) -> None:
        self._framebuffer.release()


class IntermediateFramebuffer(Framebuffer):
    def __init__(
        self,
        color_attachments: list[moderngl.Texture],
        depth_attachment: moderngl.Texture | None
    ):
        self._color_attachments: list[moderngl.Texture] = color_attachments
        self._depth_attachment: moderngl.Texture | None = depth_attachment
        super().__init__(
            ContextSingleton().framebuffer(
                color_attachments=tuple(color_attachments),
                depth_attachment=depth_attachment
            )
        )

    def get_attachment(self, index: int) -> moderngl.Texture:
        if index >= 0:
            return self._color_attachments[index]
        assert index == -1
        assert (depth_attachment := self._depth_attachment) is not None
        return depth_attachment


class Program(LazyBase):  # TODO: make abstract base class Cachable
    _CACHE: "ClassVar[dict[bytes, Program]]" = {}

    def __new__(cls, shader_str: str, dynamic_array_lens: dict[str, int]):
        # TODO: move function to somewhere suitable
        hash_val = ResourceFactory._hash_items(shader_str, ResourceFactory._dict_as_hashable(dynamic_array_lens))
        cached_instance = cls._CACHE.get(hash_val)
        if cached_instance is not None:
            return cached_instance

        instance = super().__new__(cls)
        moderngl_program = cls._construct_moderngl_program(shader_str, dynamic_array_lens)
        instance._program_ = moderngl_program
        instance._texture_binding_dict_ = cls._set_texture_bindings(moderngl_program)
        instance._uniform_block_binding_dict_ = cls._set_uniform_block_bindings(moderngl_program)
        cls._CACHE[hash_val] = instance
        return instance

    @lazy_property_initializer_writable
    @staticmethod
    def _program_() -> moderngl.Program:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _texture_binding_dict_() -> dict[str, dict[tuple[int, ...], int]]:
        return NotImplemented

    @lazy_property_initializer_writable
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
    def _construct_moderngl_program(cls, shader_str: str, dynamic_array_lens: dict[str, int]) -> moderngl.Program:
        array_len_macros = [
            f"#define {array_len_name} {array_len}"
            for array_len_name, array_len in dynamic_array_lens.items()
        ]
        shaders = {
            shader_type: cls._insert_macros(shader_str, [f"#define {shader_type}", *array_len_macros])
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
    def _insert_macros(cls, shader: str, macros: list[str]) -> str:
        def repl(match_obj: re.Match) -> str:
            return match_obj.group() + "\n" + "".join(
                f"{macro}\n" for macro in macros
            )
        return re.sub(r"#version .*\n", repl, shader, flags=re.MULTILINE)

    @classmethod
    def _set_texture_bindings(cls, program: moderngl.Program) -> dict[str, dict[tuple[int, ...], int]]:
        texture_binding_dict: dict[str, dict[tuple[int, ...], int]] = {}
        texture_uniform_match_pattern = re.compile(r"""
            (?P<texture_name>\w+?)
            (?P<multi_index>(\[\d+?\])*)
        """, flags=re.VERBOSE)
        binding = 1
        for name in program:
            member = program[name]
            if not isinstance(member, moderngl.Uniform):
                continue
            # Used as a sampler2D
            assert member.dimension == 1
            match_obj = texture_uniform_match_pattern.fullmatch(name)
            assert match_obj is not None
            texture_name = match_obj.group("texture_name")
            multi_index = tuple(
                int(index_match.group(1))
                for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
            )
            if texture_name not in texture_binding_dict:
                texture_binding_dict[texture_name] = {}
            child_binding_dict = texture_binding_dict[texture_name]
            uniform_values: list[int] = []
            if member.array_length == 1:
                child_binding_dict[multi_index] = binding
                uniform_values.append(binding)
                binding += 1
            else:
                for array_index in range(member.array_length):
                    child_binding_dict[(*multi_index, array_index)] = binding
                    uniform_values.append(binding)
                    binding += 1
            member.value = uniform_values[0] if len(uniform_values) == 1 else uniform_values
        return texture_binding_dict

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


@dataclass
class ContextState:
    depth_func: str = "<"
    blend_func: tuple[int, int] | tuple[int, int, int, int] = moderngl.DEFAULT_BLENDING
    blend_equation: int | tuple[int, int] = moderngl.FUNC_ADD
    front_face: str = "ccw"
    cull_face: str = "back"
    wireframe: bool = False


@dataclass(
    order=True,
    unsafe_hash=True,
    frozen=True,
    kw_only=True,
    slots=True
)
class RenderStep:
    shader_str: str
    texture_storages: list[TextureStorage]
    uniform_blocks: list[UniformBlockBuffer]
    subroutines: dict[str, str]
    attributes: AttributesBuffer
    index_buffer: IndexBuffer
    framebuffer: Framebuffer
    enable_only: int
    context_state: ContextState
    mode: int


class Renderable(LazyBase):
    _DEFAULT_CONTEXT_STATE: ClassVar[ContextState] = ContextState()

    @classmethod
    def _render_single_step(cls, render_step: RenderStep
        #vertex_array: moderngl.VertexArray,
        #textures: dict[str, moderngl.Texture],
        #uniforms: dict[str, moderngl.Buffer],
        #subroutines: dict[str, str],
        #framebuffer: moderngl.Framebuffer
    ) -> None:
        shader_str = render_step.shader_str
        texture_storages = render_step.texture_storages
        uniform_blocks = render_step.uniform_blocks
        subroutines = render_step.subroutines
        attributes = render_step.attributes
        index_buffer = render_step.index_buffer
        framebuffer = render_step.framebuffer
        enable_only = render_step.enable_only
        context_state = render_step.context_state
        mode = render_step.mode

        if attributes._is_empty() or index_buffer._is_empty():
            return

        dynamic_array_lens: dict[str, int] = {}
        for texture_storage in texture_storages:
            dynamic_array_lens.update(texture_storage._dynamic_array_lens_)
        for uniform_block in uniform_blocks:
            dynamic_array_lens.update(uniform_block._dynamic_array_lens_)
        dynamic_array_lens.update(attributes._dynamic_array_lens_)
        filtered_array_lens = {
            array_len_name: array_len
            for array_len_name, array_len in dynamic_array_lens.items()
            if not re.fullmatch(r"__\w+__", array_len_name)
        }

        program = Program(shader_str, filtered_array_lens)
        #program_uniforms = program._uniforms_
        #program_uniform_blocks = program._uniform_blocks_
        #program_attributes = program._attributes_
        #program_subroutines = program._subroutines_

        ## Remove redundancies
        #textures: list[moderngl.Texture] = list(dict.fromkeys(
        #    texture for texture_storage in texture_storages
        #    for texture in texture_storage._texture_list_
        #))

        # texture storages
        texture_storage_dict = {
            texture_storage.field_info.name: texture_storage
            for texture_storage in texture_storages
        }
        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_storage_name, child_binding_dict in program._texture_binding_dict_.items():
            texture_storage = texture_storage_dict[texture_storage_name]
            assert not texture_storage._is_empty()
            for multi_index, binding in child_binding_dict.items():
                if texture_storage._texture_array_.ndim != len(multi_index):
                    multi_index = (*multi_index, 0)
                texture_bindings.append((texture_storage._texture_array_[multi_index], binding))

        # uniform blocks
        uniform_block_dict = {
            uniform_block.field_info.name: uniform_block
            for uniform_block in uniform_blocks
        }
        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
        for uniform_block_name, binding in program._uniform_block_binding_dict_.items():
            uniform_block = uniform_block_dict[uniform_block_name]
            assert not uniform_block._is_empty()
            program_uniform_block = program._program_[uniform_block_name]
            assert isinstance(program_uniform_block, moderngl.UniformBlock)
            uniform_block._validate(program_uniform_block)
            uniform_block_bindings.append((uniform_block._buffer_, binding))

        # subroutines
        subroutine_indices: list[int] = [
            program._subroutines_[subroutines[subroutine_name]].index
            for subroutine_name in program._program_.subroutines
        ]

        # attributes
        program_attributes = program._attributes_
        attributes._validate(program_attributes)
        buffer_format, attribute_names = attributes._get_buffer_format(set(program_attributes))
        vertex_array = ContextSingleton().vertex_array(
            program=program._program_,
            content=[(attributes._buffer_, buffer_format, *attribute_names)],
            index_buffer=index_buffer._buffer_,
            mode=mode
        )

        cls._set_context_state(context_state)
        with ContextSingleton().scope(
            framebuffer=framebuffer._framebuffer,
            enable_only=enable_only,
            textures=tuple(texture_bindings),
            uniform_buffers=tuple(uniform_block_bindings)
        ):
            vertex_array.subroutines = tuple(subroutine_indices)
            vertex_array.render()
        cls._set_context_state(cls._DEFAULT_CONTEXT_STATE)

    @classmethod
    def _render_by_step(cls, *render_steps: RenderStep) -> None:
        for render_step in render_steps:
            cls._render_single_step(render_step)

    #@classmethod
    #def _render_by_routine(cls, render_routine: list[RenderStep]) -> None:
    #    for render_step in render_routine:
    #        cls._render_by_step(render_step)

    @lru_cache(maxsize=8)
    @staticmethod
    def _read_shader(filename: str) -> str:
        with open(os.path.join(SHADERS_PATH, f"{filename}.glsl")) as shader_file:
            content = shader_file.read()
        return content

    @classmethod
    def _set_context_state(cls, context_state: ContextState) -> None:
        context = ContextSingleton()
        context.depth_func = context_state.depth_func
        context.blend_func = context_state.blend_func
        context.blend_equation = context_state.blend_equation
        context.front_face = context_state.front_face
        context.cull_face = context_state.cull_face
        context.wireframe = context_state.wireframe
