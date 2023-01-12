__all__ = [
    "AttributesBuffer",
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
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import os
import re
from typing import (
    Any,
    ClassVar,
    Generator,
    Generic,
    Hashable,
    TypeVar
)

import moderngl
import numpy as np
from xxhash import xxh3_64_digest

from ..constants import (
    GLSL_DTYPE,
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    SHADERS_PATH
)
#from ..custom_typing import VertexIndicesType
from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)


_T = TypeVar("_T", bound="Hashable")


class ResourceFactory(Generic[_T], ABC):
    __OCCUPIED__: list[_T]
    __VACANT__: list[_T]
    __RESOURCE_GENERATOR__: Generator[_T, None, None]

    def __init_subclass__(cls) -> None:
        cls.__OCCUPIED__ = []
        cls.__VACANT__ = []
        cls.__RESOURCE_GENERATOR__ = cls._generate()
        return super().__init_subclass__()

    @classmethod
    @abstractmethod
    def _generate(cls) -> Generator[_T, None, None]:
        pass

    #@classmethod
    #def _reset(cls, resource: _T) -> None:
    #    pass

    #@classmethod
    #def _release(cls, resource: _T) -> None:
    #    pass

    @classmethod
    def _register_enter(cls) -> _T:
        if cls.__VACANT__:
            resource = cls.__VACANT__.pop(0)
        else:
            try:
                resource = next(cls.__RESOURCE_GENERATOR__)
            except StopIteration:
                raise MemoryError(f"{cls.__name__} cannot allocate a new object") from None
        cls.__OCCUPIED__.append(resource)
        return resource

    @classmethod
    def _register_exit(cls, resource: _T) -> None:
        cls.__OCCUPIED__.remove(resource)
        #cls._reset(resource)
        cls.__VACANT__.append(resource)

    #@classmethod
    #@contextmanager
    #def register(cls) -> Generator[_T, None, None]:
    #    resource = cls._register_enter()
    #    try:
    #        yield resource
    #    finally:
    #        cls._register_exit(resource)

    @classmethod
    @contextmanager
    def register_n(cls, n: int) -> Generator[list[_T], None, None]:
        resource_list = [
            cls._register_enter()
            for _ in range(n)
        ]
        try:
            yield resource_list
        finally:
            for resource in resource_list:
                cls._register_exit(resource)

    #@classmethod
    #def _release_all(cls) -> None:
    #    while cls.__OCCUPIED__:
    #        resource = cls.__OCCUPIED__.pop()
    #        cls._release(resource)
    #    while cls.__VACANT__:
    #        resource = cls.__VACANT__.pop()
    #        cls._release(resource)


class TextureBindings(ResourceFactory[int]):
    @classmethod
    def _generate(cls) -> Generator[int, None, None]:
        for texture_binding in range(1, 32):  # TODO
            yield texture_binding


class UniformBindings(ResourceFactory[int]):
    @classmethod
    def _generate(cls) -> Generator[int, None, None]:
        for uniform_binding in range(64):  # TODO
            yield uniform_binding


class IntermediateTextures(ResourceFactory[moderngl.Texture]):
    @classmethod
    def _generate(cls) -> Generator[moderngl.Texture, None, None]:
        while True:
            yield ContextSingleton().texture(
                size=(PIXEL_WIDTH, PIXEL_HEIGHT),
                components=4
            )

    #@classmethod
    #def _release(cls, resource: moderngl.Texture) -> None:
    #    resource.release()


class IntermediateDepthTextures(IntermediateTextures):
    @classmethod
    def _generate(cls) -> Generator[moderngl.Texture, None, None]:
        while True:
            yield ContextSingleton().depth_texture(
                size=(PIXEL_WIDTH, PIXEL_HEIGHT)
            )


@dataclass
class FieldInfo:
    dtype_str: str
    name: str
    array_shape: list[int | str]


class GLSLBuffer(LazyBase):
    # std140 format
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
        return ContextSingleton().buffer(reserve=1024)  # TODO: dynamic?

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
    def write(self, data: np.ndarray | dict[str, Any]) -> None:
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
            return
        #print(struct_dtype.itemsize)
        buffer.orphan(struct_dtype.itemsize)
        buffer.write(bytes_data)

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

                base_alignment = 16
                pad_tail = True
            else:
                child_dtype = GLSL_DTYPE[dtype_str]
                assert len(child_data) == 1 and (data_value := child_data.get(())) is not None
                if not data_value.size:
                    continue
                assert child_dtype.shape == data_value.shape[next_depth:]

                if not shape and len(child_dtype.shape) <= 1:
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

            offset += (-offset) % base_alignment
            field_items.append((name, child_dtype, shape, offset))
            offset += np.prod(shape, dtype=int) * child_dtype.itemsize
            if pad_tail:
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
                int(s) if re.match(r"^\d+$", s := match_obj.group(1)) is not None else s
                for match_obj in re.finditer(r"\[(\w+?)\]", match_obj.group("array_shape"))
            ],
        )


class TextureStorage(GLSLBuffer):
    def __init__(self, field: str):
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        super().__init__(field=replaced_field)

    @lazy_property_initializer_writable
    @staticmethod
    def _texture_list_() -> list[moderngl.Texture]:
        return NotImplemented

    def write(self, textures: np.ndarray) -> None:
        texture_list: list[moderngl.Texture] = []
        # Remove redundancies
        for texture in textures.flatten():
            assert isinstance(texture, moderngl.Texture)
            if texture not in texture_list:
                texture_list.append(texture)
        self._texture_list_ = texture_list
        data = np.vectorize(lambda texture: texture_list.index(texture), otypes=[np.uint32])(textures)
        super().write(data)

    def _get_indices(self) -> list[int]:
        size = self._struct_dtype_[self.field_info.name].itemsize
        return list(np.frombuffer(self._buffer_.read(size=size), np.uint32))

    def _validate(self, uniform: moderngl.Uniform) -> None:
        assert uniform.name == self.field_info.name
        assert uniform.dimension == 1
        assert uniform.array_length == np.prod(self._struct_dtype_[self.field_info.name].shape, dtype=int)


class UniformBlockBuffer(GLSLBuffer):
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
            if dtype.base.fields is not None:
                dtype_stack = [
                    (child_dtype, offset + i * dtype.base.itemsize + child_offset)
                    for i in range(np.prod(dtype.shape, dtype=int))
                    for child_dtype, child_offset, *_ in dtype.base.fields.values()
                ] + dtype_stack
                continue
            if current_offset != offset:
                components.append(f"{offset - current_offset}x")
                current_offset = offset
            dtype_size = np.prod(dtype.shape, dtype=int)
            dtype_itemsize = dtype.base.itemsize
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
            if any(glsl_dtype is field_dtype for glsl_dtype in GLSL_DTYPE.values()):
                array_shape = ()
                atom_dtype = field_dtype
            else:
                array_shape = field_dtype.shape
                atom_dtype = field_dtype.base
            assert attribute.array_length == np.prod(array_shape, dtype=int)
            assert attribute.dimension == np.prod(atom_dtype.shape, dtype=int)
            assert attribute.shape == atom_dtype.base.kind.replace("u", "I")


class IndexBuffer(GLSLBuffer):
    def __init__(self):
        super().__init__(field="uint __index__[__NUM_INDEX__]")


class Framebuffer:
    def __init__(self, framebuffer: moderngl.Framebuffer):
        self._framebuffer: moderngl.Framebuffer = framebuffer


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


class Programs:  # TODO: make abstract base class Cachable
    _CACHE: ClassVar[dict[bytes, moderngl.Program]] = {}

    @classmethod
    def _get_program(cls, shader_str: str, dynamic_array_lens: dict[str, int]) -> moderngl.Program:
        hash_val = cls._hash_items(shader_str, cls._dict_as_hashable(dynamic_array_lens))
        cached_program = cls._CACHE.get(hash_val)
        if cached_program is not None:
            return cached_program

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
        cls._CACHE[hash_val] = program
        return program

    @classmethod
    def _insert_macros(cls, shader: str, macros: list[str]) -> str:
        def repl(match_obj: re.Match) -> str:
            return match_obj.group() + "\n" + "".join(
                f"{macro}\n" for macro in macros
            )
        return re.sub(r"#version .*\n", repl, shader, flags=re.MULTILINE)

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
    attributes: AttributesBuffer
    subroutines: dict[str, str]
    index_buffer: IndexBuffer
    framebuffer: Framebuffer
    enable_only: int
    mode: int


class Renderable(LazyBase):
    @classmethod
    def _render_by_step(cls, render_step: RenderStep
        #vertex_array: moderngl.VertexArray,
        #textures: dict[str, moderngl.Texture],
        #uniforms: dict[str, moderngl.Buffer],
        #subroutines: dict[str, str],
        #framebuffer: moderngl.Framebuffer
    ) -> None:
        shader_str = render_step.shader_str
        texture_storages = render_step.texture_storages
        uniform_blocks = render_step.uniform_blocks
        attributes = render_step.attributes
        subroutines = render_step.subroutines
        index_buffer = render_step.index_buffer
        framebuffer = render_step.framebuffer
        enable_only = render_step.enable_only
        mode = render_step.mode

        if attributes._is_empty() or index_buffer._is_empty():
            return

        dynamic_array_lens: dict[str, int] = {}
        for texture_storage in texture_storages:
            dynamic_array_lens.update(texture_storage._dynamic_array_lens_)
        for uniform_block in uniform_blocks:
            dynamic_array_lens.update(uniform_block._dynamic_array_lens_)
        dynamic_array_lens.update(attributes._dynamic_array_lens_)

        program = Programs._get_program(shader_str, dynamic_array_lens)
        program_uniforms, program_uniform_blocks, program_attributes, program_subroutines \
            = cls._get_program_parameters(program)

        texture_storage_dict = {
            texture_storage.field_info.name: texture_storage
            for texture_storage in texture_storages
        }
        #texture_dict: dict[tuple[str, int], moderngl.Texture] = {
        #    (texture_storage_name, index): texture
        #    for texture_storage_name, texture_storage in texture_storage_dict.items()
        #    for index, texture in enumerate(texture_storage._texture_list_)
        #}
        uniform_block_dict = {
            uniform_block.field_info.name: uniform_block
            for uniform_block in uniform_blocks
        }
        # Remove redundancies
        textures: list[moderngl.Texture] = list(dict.fromkeys(
            texture for texture_storage in texture_storages
            for texture in texture_storage._texture_list_
        ))

        with TextureBindings.register_n(len(textures)) as texture_bindings:
            with UniformBindings.register_n(len(uniform_blocks)) as uniform_block_bindings:
                texture_binding_dict = dict(zip(textures, texture_bindings))
                uniform_block_binding_dict = dict(zip(uniform_blocks, uniform_block_bindings))

                #texture_binding_dict = {
                #    texture: TextureBindings.log_in()
                #    for texture in textures.values()
                #}
                #uniform_binding_dict = {
                #    uniform: UniformBindings.log_in()
                #    for uniform in uniforms.values()
                #}

                with ContextSingleton().scope(
                    framebuffer=framebuffer._framebuffer,
                    enable_only=enable_only,
                    textures=tuple(texture_binding_dict.items()),
                    uniform_buffers=tuple(
                        (uniform_block._buffer_, binding)
                        for uniform_block, binding in uniform_block_binding_dict.items()
                    )
                ):
                    # texture storages
                    for texture_storage_name, program_uniform in program_uniforms.items():
                        texture_storage = texture_storage_dict[texture_storage_name]
                        assert not texture_storage._is_empty()
                        texture_storage._validate(program_uniform)
                        binding_values = [
                            texture_binding_dict[texture_storage._texture_list_[index]]
                            for index in texture_storage._get_indices()
                        ]
                        program_uniform.value = binding_values[0] if len(binding_values) == 1 else binding_values

                    # uniform blocks
                    for uniform_block_name, program_uniform_block in program_uniform_blocks.items():
                        uniform_block = uniform_block_dict[uniform_block_name]
                        assert not uniform_block._is_empty()
                        uniform_block._validate(program_uniform_block)
                        program_uniform_block.value = uniform_block_binding_dict[uniform_block]

                    # attributes
                    #assert not attributes._is_empty()
                    attributes._validate(program_attributes)
                    #assert not index_buffer._is_empty()
                    buffer_format, attribute_names = attributes._get_buffer_format(set(program_attributes))
                    vertex_array = ContextSingleton().vertex_array(
                        program=program,
                        content=[(attributes._buffer_, buffer_format, *attribute_names)],
                        index_buffer=index_buffer._buffer_,
                        mode=mode
                    )

                    # subroutines
                    vertex_array.subroutines = tuple(
                        program_subroutines[subroutines[subroutine_name]].index
                        for subroutine_name in program.subroutines
                    )

                    vertex_array.render()

        #assert not program_subroutines
        #for texture_binding in texture_binding_dict.values():
        #    TextureBindings.log_out(texture_binding)
        #for uniform_binding in uniform_binding_dict.values():
        #    UniformBindings.log_out(uniform_binding)

    @classmethod
    def _render_by_routine(cls, render_routine: list[RenderStep]) -> None:
        for render_step in render_routine:
            cls._render_by_step(render_step)

    @classmethod
    def _get_program_parameters(
        cls,
        program: moderngl.Program
    ) -> tuple[
        dict[str, moderngl.Uniform],
        dict[str, moderngl.UniformBlock],
        dict[str, moderngl.Attribute],
        dict[str, moderngl.Subroutine]
    ]:
        program_uniforms: dict[str, moderngl.Uniform] = {}
        program_uniform_blocks: dict[str, moderngl.UniformBlock] = {}
        program_attributes: dict[str, moderngl.Attribute] = {}
        program_subroutines: dict[str, moderngl.Subroutine] = {}
        for name in program:
            member = program[name]
            if isinstance(member, moderngl.Uniform):
                program_uniforms[name] = member
            elif isinstance(member, moderngl.UniformBlock):
                program_uniform_blocks[name] = member
            elif isinstance(member, moderngl.Attribute):
                program_attributes[name] = member
            elif isinstance(member, moderngl.Subroutine):
                program_subroutines[name] = member
        return program_uniforms, program_uniform_blocks, program_attributes, program_subroutines

    @lru_cache(maxsize=8)
    @staticmethod
    def _read_shader(filename: str) -> str:
        with open(os.path.join(SHADERS_PATH, f"{filename}.glsl")) as shader_file:
            content = shader_file.read()
        return content
