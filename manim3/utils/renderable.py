__all__ = [
    "AttributeBuffer",
    "IndexBuffer",
    "IntermediateDepthTextures",
    "IntermediateTextures",
    "RenderStep",
    "Renderable",
    "ShaderStrings",
    "UniformBlockBuffer"
]


from abc import (
    ABC,
    abstractmethod
)
from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import (
    ClassVar,
    Generator,
    Generic,
    Hashable,
    TypeVar,
    overload
)

import moderngl
import numpy as np
from xxhash import xxh3_64_digest

from ..constants import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH
)
from ..custom_typing import VertexIndicesType
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
class DeclarationInfo:
    shape: tuple[int, ...]
    dtype: np.dtype
    array_len: int | str | None


class GLSLVariable(LazyBase):
    _DTYPE_STR_CONVERSION_DICT: ClassVar[dict[str, np.dtype]] = {
        "bool": np.dtype(np.int32),
        "uint": np.dtype(np.uint32),
        "int": np.dtype(np.int32),
        "float": np.dtype(np.float32),
        "double": np.dtype(np.float64)
    }
    _DTYPE_CHAR_CONVERSION_DICT: ClassVar[dict[str, np.dtype]] = {
        dtype_str[0]: dtype
        for dtype_str, dtype in _DTYPE_STR_CONVERSION_DICT.items()
    }
    _DTYPE_CHAR_CONVERSION_DICT[""] = _DTYPE_CHAR_CONVERSION_DICT.pop("f")

    def __init__(self, declaration: str):
        super().__init__()
        self._declaration_ = declaration

    @lazy_property_initializer_writable
    @staticmethod
    def _declaration_() -> str:
        return NotImplemented

    @classmethod
    def _parse_shape_and_dtype(cls, dtype_str: str) -> tuple[tuple[int, ...], np.dtype]:
        str_to_dtype = GLSLVariable._DTYPE_STR_CONVERSION_DICT
        char_to_dtype = GLSLVariable._DTYPE_CHAR_CONVERSION_DICT
        if (type_match := re.fullmatch(r"(?P<dtype>[a-z]+)", dtype_str)) is not None:
            shape = ()
            dtype = str_to_dtype[type_match.group("dtype")]
        elif (type_match := re.fullmatch(r"(?P<dtype_char>\w*)vec(?P<n>[2-4])", dtype_str)) is not None:
            n = int(type_match.group("n"))
            shape = (n,)
            dtype = char_to_dtype[type_match.group("dtype_char")]
        elif (type_match := re.fullmatch(r"(?P<dtype_char>\w*)mat(?P<n>[2-4])", dtype_str)) is not None:
            n = int(type_match.group("n"))
            shape = (n, n)
            dtype = char_to_dtype[type_match.group("dtype_char")]
        elif (type_match := re.fullmatch(r"(?P<dtype_char>\w*)mat(?P<n>[2-4])x(?P<m>[2-4])", dtype_str)) is not None:
            n = int(type_match.group("n"))
            m = int(type_match.group("m"))
            shape = (n, m)  # TODO: check order
            dtype = char_to_dtype[type_match.group("dtype_char")]
        else:
            raise ValueError(f"Invalid dtype string: `{dtype_str}`")
        return (shape, dtype)

    @lazy_property
    @staticmethod
    def _declaration_info_(declaration: str) -> DeclarationInfo:
        pattern = re.compile(r"""
            (?P<dtype>\w+)
            (?:\[(?:
                (?P<array_len>\d+)
                |(?P<dynamic_array_len>[a-zA-Z_]+)
            )\])?
        """, flags=re.X)
        match_obj = pattern.fullmatch(declaration)
        assert match_obj is not None
        shape, dtype = GLSLVariable._parse_shape_and_dtype(match_obj.group("dtype"))
        if (array_len := match_obj.group("array_len")) is not None:
            array_len = int(array_len)
        elif (array_len := match_obj.group("dynamic_array_len")) is not None:
            array_len = str(array_len)
        else:
            array_len = None
        return DeclarationInfo(
            shape=shape,
            dtype=dtype,
            array_len=array_len
        )

    @lazy_property
    @staticmethod
    def _info_shape_(declaration_info: DeclarationInfo) -> tuple[int, ...]:
        return declaration_info.shape

    @lazy_property
    @staticmethod
    def _info_dtype_(declaration_info: DeclarationInfo) -> np.dtype:
        return declaration_info.dtype

    @lazy_property
    @staticmethod
    def _info_array_len_(declaration_info: DeclarationInfo) -> int | str | None:
        return declaration_info.array_len

    @lazy_property
    @staticmethod
    def _info_size_(info_shape: tuple[int, ...]) -> int:
        return np.prod(info_shape, dtype=int)

    @lazy_property_initializer_writable
    @staticmethod
    def _data_bytes_() -> bytes:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _data_num_blocks_() -> int | None:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _data_array_len_() -> int:
        return NotImplemented

    def write(self, data: np.ndarray, *, multiple_blocks: bool = False) -> None:
        info_array_len = self._info_array_len_
        data = data.astype(self._info_dtype_, casting="same_kind")
        if not data.size:
            assert isinstance(info_array_len, str)
            self._data_bytes_ = bytes()
            self._data_num_blocks_ = None
            self._data_array_len_ = 0
            return

        shape = data.shape
        if multiple_blocks:
            data_num_blocks = shape[0]
            shape = shape[1:]
        else:
            data_num_blocks = None
        if info_array_len is not None:
            data_array_len = shape[0]
            if isinstance(info_array_len, int):
                assert info_array_len == data_array_len
            shape = shape[1:]
        else:
            data_array_len = 1

        self._data_bytes_ = data.tobytes()
        self._data_num_blocks_ = data_num_blocks
        self._data_array_len_ = data_array_len

    def _is_empty(self) -> bool:
        return self._data_array_len_ == 0


class DynamicBuffer(LazyBase):
    @abstractmethod
    def _get_variables(self) -> list[GLSLVariable]:
        pass

    def _dump_dynamic_array_lens(self) -> dict[str, int]:
        dynamic_array_lens: dict[str, int] = {}
        for variable in self._get_variables():
            if isinstance(variable._info_array_len_, str):
                dynamic_array_lens[variable._info_array_len_] = variable._data_array_len_
        return dynamic_array_lens

    def _is_empty(self) -> bool:
        return all(variable._is_empty() for variable in self._get_variables())


class TextureStorage(DynamicBuffer):
    def __init__(self, declaration: str):
        super().__init__()
        assert declaration.startswith("sampler2D")
        self._variable_ = GLSLVariable(declaration.replace("sampler2D", "uint"))

    @lazy_property_initializer_writable
    @staticmethod
    def _variable_() -> GLSLVariable:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _texture_list_() -> list[moderngl.Texture]:
        return NotImplemented

    @_variable_.updater
    def write(self, textures: moderngl.Texture | list[moderngl.Texture]) -> None:
        if isinstance(textures, moderngl.Texture):
            texture_list = [textures]
            data = np.array(0, dtype=np.uint32)
        else:
            texture_list = textures[:]
            data = np.arange(len(textures), dtype=np.uint32)
        self._texture_list_ = texture_list
        self._variable_.write(data)

    def _validate(self, uniform: moderngl.Uniform) -> None:
        assert uniform.dimension == 1
        assert uniform.array_length == self._variable_._data_array_len_

    def _get_variables(self) -> list[GLSLVariable]:
        return [self._variable_]


class UniformBlockBuffer(DynamicBuffer):
    def __init__(self, declaration_dict: dict[str, str]):
        super().__init__()
        self._variables_ = {
            name: GLSLVariable(declaration)
            for name, declaration in declaration_dict.items()
        }

    def __del__(self):
        pass  # TODO: release buffer

    @lazy_property_initializer_writable
    @staticmethod
    def _variables_() -> dict[str, GLSLVariable]:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _buffer_() -> moderngl.Buffer:
        return ContextSingleton().buffer(reserve=1024)  # TODO: dynamic?

    @_variables_.updater
    @_buffer_.updater
    def write(self, data: dict[str, np.ndarray]) -> None:
        # std140 format
        chunk_items: list[tuple[bytes, int, int, int]] = []
        offset = 0
        data_copy = data.copy()
        for name, variable in self._variables_.items():
            uniform = data_copy.pop(name)
            variable.write(uniform)
            if variable._is_empty():
                continue

            shape = variable._info_shape_
            size = variable._info_size_
            if variable._info_array_len_ is None and len(shape) < 2:
                chunk_alignment = size if size <= 2 else 4
                chunk_count = 1
                occupied_units = size
            else:
                chunk_alignment = 4
                chunk_count = variable._data_array_len_ * size // (shape[-1] if shape else 1)
                occupied_units = chunk_count * chunk_alignment

            itemsize = variable._info_dtype_.itemsize
            base_alignment = chunk_alignment * itemsize
            offset += (-offset) % base_alignment
            chunk_items.append((variable._data_bytes_, offset, base_alignment, chunk_count))
            offset += occupied_units * itemsize

        assert not data_copy

        offset += (-offset) % 16
        buffer = self._buffer_
        if not offset:
            buffer.clear()
            return
        #print("UniformBlockBuffer", offset)
        buffer.orphan(offset)
        for data_bytes, offset, base_alignment, chunk_count in chunk_items:
            buffer.write_chunks(data_bytes, start=offset, step=base_alignment, count=chunk_count)

    def _validate(self, uniform_block: moderngl.UniformBlock) -> None:
        assert uniform_block.size == self._buffer_.size

    def _get_variables(self) -> list[GLSLVariable]:
        return list(self._variables_.values())


class AttributeBuffer(DynamicBuffer):
    def __init__(self, declaration: str):
        super().__init__()
        self._variable_ = GLSLVariable(declaration)

    @lazy_property_initializer_writable
    @staticmethod
    def _variable_() -> GLSLVariable:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _usage_() -> str:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _buffer_() -> moderngl.Buffer:
        return ContextSingleton().buffer(reserve=1024)  # TODO: dynamic?

    @_variable_.updater
    @_buffer_.updater
    def write(self, data: np.ndarray, usage: str) -> None:
        assert usage in ("v", "r")
        multiple_blocks = usage != "r"
        self._usage_ = usage
        variable = self._variable_
        variable.write(data, multiple_blocks=multiple_blocks)
        #data_info = variable._data_info_
        #if data_info is None:
        #    buffer.clear()
        #    return
        data_bytes = variable._data_bytes_
        buffer = self._buffer_
        #print("AttributeBuffer", len(data_bytes))
        buffer.orphan(len(data_bytes))
        buffer.write(data_bytes)

    @lazy_property
    @staticmethod
    def _buffer_format_(variable: GLSLVariable, usage: str) -> str:
        declaration_info = variable._declaration_info_
        size = np.prod(declaration_info.shape, dtype=int)
        dtype = declaration_info.dtype
        return f"{size}{dtype.kind}{dtype.itemsize} /{usage}"

    def _validate(self, attribute: moderngl.Attribute) -> None:
        assert attribute.array_length == self._variable_._data_array_len_
        assert attribute.dimension == self._variable_._info_size_
        assert attribute.shape == self._variable_._info_dtype_.kind.replace("u", "I")

    def _get_variables(self) -> list[GLSLVariable]:
        return [self._variable_]


class IndexBuffer(DynamicBuffer):
    @lazy_property_initializer
    @staticmethod
    def _variable_() -> GLSLVariable:
        return GLSLVariable("uint")

    @lazy_property_initializer
    @staticmethod
    def _buffer_() -> moderngl.Buffer:
        return ContextSingleton().buffer(reserve=1024)  # TODO: dynamic? check capacity

    @_variable_.updater
    @_buffer_.updater
    def write(self, data: VertexIndicesType) -> None:
        variable = self._variable_
        variable.write(data, multiple_blocks=True)
        #data_info = variable._data_info_
        #if data_info is None:
        #    buffer.clear()
        #    return
        data_bytes = variable._data_bytes_
        buffer = self._buffer_
        #print("IndexBuffer", len(data_bytes))
        buffer.orphan(len(data_bytes))
        buffer.write(data_bytes)

    def _get_variables(self) -> list[GLSLVariable]:
        return [self._variable_]


@dataclass(
    order=True,
    unsafe_hash=True,
    frozen=True,
    kw_only=True,
    slots=True
)
class ShaderStrings:
    vertex_shader: str
    fragment_shader: str | None = None
    geometry_shader: str | None = None
    tess_control_shader: str | None = None
    tess_evaluation_shader: str | None = None


class Programs:  # TODO: make abstract base class Cachable
    _CACHE: ClassVar[dict[bytes, moderngl.Program]] = {}

    @classmethod
    def _get_program(cls, shader_strings: ShaderStrings, dynamic_array_lens: dict[str, int]) -> moderngl.Program:
        hash_val = cls._hash_items(shader_strings, cls._dict_as_hashable(dynamic_array_lens))
        cached_program = cls._CACHE.get(hash_val)
        if cached_program is not None:
            return cached_program
        program = ContextSingleton().program(
            vertex_shader=cls._replace_array_lens(shader_strings.vertex_shader, dynamic_array_lens),
            fragment_shader=cls._replace_array_lens(shader_strings.fragment_shader, dynamic_array_lens),
            geometry_shader=cls._replace_array_lens(shader_strings.geometry_shader, dynamic_array_lens),
            tess_control_shader=cls._replace_array_lens(shader_strings.tess_control_shader, dynamic_array_lens),
            tess_evaluation_shader=cls._replace_array_lens(shader_strings.tess_evaluation_shader, dynamic_array_lens),
        )
        cls._CACHE[hash_val] = program
        return program

    @overload
    @classmethod
    def _replace_array_lens(cls, shader: str, dynamic_array_lens: dict[str, int]) -> str: ...

    @overload
    @classmethod
    def _replace_array_lens(cls, shader: None, dynamic_array_lens: dict[str, int]) -> None: ...

    @classmethod
    def _replace_array_lens(cls, shader: str | None, dynamic_array_lens: dict[str, int]) -> str | None:
        if shader is None:
            return None

        def repl(match_obj: re.Match) -> str:
            array_len_name = match_obj.group(1)
            assert isinstance(array_len_name, str)
            return f"#define {array_len_name} {dynamic_array_lens[array_len_name]}"
        return re.sub(r"#define (\w+) _(?=\n)", repl, shader, flags=re.MULTILINE)

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
    #program: moderngl.Program
    shader_strings: ShaderStrings
    texture_storages: dict[str, TextureStorage]
    uniform_blocks: dict[str, UniformBlockBuffer]
    attributes: dict[str, AttributeBuffer]
    subroutines: dict[str, str]
    index_buffer: IndexBuffer
    framebuffer: moderngl.Framebuffer
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
        shader_strings = render_step.shader_strings
        texture_storages = render_step.texture_storages
        uniform_blocks = render_step.uniform_blocks
        attributes = render_step.attributes
        subroutines = render_step.subroutines
        index_buffer = render_step.index_buffer
        framebuffer = render_step.framebuffer
        enable_only = render_step.enable_only
        mode = render_step.mode

        dynamic_array_lens: dict[str, int] = {}
        for texture_storage in texture_storages.values():
            dynamic_array_lens.update(texture_storage._dump_dynamic_array_lens())
        for uniform_block in uniform_blocks.values():
            dynamic_array_lens.update(uniform_block._dump_dynamic_array_lens())
        for attribute in attributes.values():
            dynamic_array_lens.update(attribute._dump_dynamic_array_lens())

        program = Programs._get_program(shader_strings, dynamic_array_lens)
        program_uniforms, program_uniform_blocks, program_attributes, program_subroutines \
            = cls._get_program_parameters(program)

        textures: dict[tuple[str, int], moderngl.Texture] = {
            (texture_storage_name, index): texture
            for texture_storage_name, texture_storage in texture_storages.items()
            for index, texture in enumerate(texture_storage._texture_list_)
        }
        subroutines_copy = subroutines.copy()
        with TextureBindings.register_n(len(textures)) as texture_bindings:
            with UniformBindings.register_n(len(uniform_blocks)) as uniform_block_bindings:
                texture_binding_dict = dict(zip(textures.values(), texture_bindings))
                uniform_block_binding_dict = dict(zip(uniform_blocks.values(), uniform_block_bindings))

                #texture_binding_dict = {
                #    texture: TextureBindings.log_in()
                #    for texture in textures.values()
                #}
                #uniform_binding_dict = {
                #    uniform: UniformBindings.log_in()
                #    for uniform in uniforms.values()
                #}

                with ContextSingleton().scope(
                    framebuffer=framebuffer,
                    enable_only=enable_only,
                    textures=tuple(texture_binding_dict.items()),
                    uniform_buffers=tuple(
                        (uniform_block._buffer_, binding)
                        for uniform_block, binding in uniform_block_binding_dict.items()
                    )
                ):
                    # texture storage
                    for texture_storage_name, texture_storage in texture_storages.items():
                        if texture_storage._is_empty():
                            continue
                        program_uniform = program_uniforms.pop(texture_storage_name)
                        texture_storage._validate(program_uniform)
                        texture_bindings = [
                            texture_binding_dict[textures[(texture_storage_name, index)]]
                            for index, _ in enumerate(texture_storage._texture_list_)
                        ]
                        program_uniform.value = texture_bindings[0] if len(texture_bindings) == 1 else texture_bindings

                    # uniform block
                    for uniform_block_name, uniform_block in uniform_blocks.items():
                        if uniform_block._is_empty():
                            continue
                        program_uniform_block = program_uniform_blocks.pop(uniform_block_name)
                        uniform_block._validate(program_uniform_block)
                        program_uniform_block.value = uniform_block_binding_dict[uniform_block]

                    # attribute
                    attribute_num_blocks: list[int] = []
                    content: list[tuple[moderngl.Buffer, str, str]] = []
                    for attribute_name, attribute in attributes.items():
                        if attribute._is_empty():
                            continue
                        program_attribute = program_attributes.pop(attribute_name)
                        attribute._validate(program_attribute)
                        if (attribute_num_block := attribute._variable_._data_num_blocks_) is not None:
                            attribute_num_blocks.append(attribute_num_block)
                        content.append((attribute._buffer_, attribute._buffer_format_, attribute_name))
                    assert len(set(attribute_num_blocks)) <= 1

                    vertex_array = ContextSingleton().vertex_array(
                        program=program,
                        content=content,
                        index_buffer=index_buffer._buffer_,
                        mode=mode
                    )

                    # subroutine
                    subroutine_indices: list[int] = []
                    for subroutine_name in program.subroutines:
                        subroutine_value = subroutines_copy.pop(subroutine_name)
                        subroutine_indices.append(program_subroutines[subroutine_value].index)
                    vertex_array.subroutines = tuple(subroutine_indices)

                    vertex_array.render()

        assert not program_uniforms
        assert not program_uniform_blocks
        assert not program_attributes
        assert not subroutines_copy
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
