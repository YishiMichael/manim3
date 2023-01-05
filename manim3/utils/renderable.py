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


from abc import ABC, abstractmethod
#import atexit
from contextlib import contextmanager
from dataclasses import dataclass
#from functools import lru_cache
#import inspect
#import os
import re
from typing import ClassVar, Generator, Generic, Hashable, TypeVar, overload


import moderngl
import numpy as np
import numpy.typing as npt
#from numpy.typing import DTypeLike
from xxhash import xxh3_64_digest

#from PIL import Image
#import skia

#from ..utils.node import Node
from ..utils.lazy import LazyBase, lazy_property, lazy_property_initializer_writable
#from ..utils.lazy import LazyBase, lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..constants import PIXEL_HEIGHT, PIXEL_WIDTH
from ..custom_typing import *
from ..utils.context_singleton import ContextSingleton


_T = TypeVar("_T", bound="Hashable")


class ResourceFactory(Generic[_T], ABC):
    __OCCUPIED__: list[_T]
    __VACANT__: list[_T]
    __RESOURCE_GENERATOR__: Generator[_T, None, None]

    def __init_subclass__(cls) -> None:
        cls.__OCCUPIED__ = []
        cls.__VACANT__ = []
        cls.__RESOURCE_GENERATOR__ = cls._generate()
        #atexit.register(lambda: cls._release_all())
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


    #_RESOURCE_DICT: dict[_T, int] = {}
    #_VACANT_BINDINGS: set[int] = set()

    #@abstractmethod
    #@classmethod
    #def _init_vacant_bindings(cls) -> set[int]:
    #    pass

    #def __init_subclass__(cls):
    #    cls._VACANT_BINDINGS.update(cls._init_vacant_bindings())

    #@classmethod
    #def log_in(cls, resource: _T) -> int:
    #    if not cls._VACANT_BINDINGS:
    #        raise ValueError(f"Cannot allocate a binding for {resource}")
    #    binding = min(cls._VACANT_BINDINGS)
    #    cls._VACANT_BINDINGS.remove(binding)
    #    cls._RESOURCE_DICT[resource] = binding
    #    return binding

    #@classmethod
    #def log_out(cls, resource: _T) -> None:
    #    binding = cls._RESOURCE_DICT.pop(resource)
    #    cls._VACANT_BINDINGS.add(binding)


#class TextureBindings(ResourceBindings[moderngl.Texture]):
#    @classmethod
#    def _init_vacant_bindings(cls) -> set[int]:
#        return set(range(1, 32))  # TODO


#class UniformBindings(ResourceBindings[moderngl.Buffer]):
#    @classmethod
#    def _init_vacant_bindings(cls) -> set[int]:
#        return set(range(64))  # TODO




    #@contextmanager
    #def register(cls, texture: moderngl.Texture) -> Generator[int, None, None]:
    #    try:
    #        yield cls.log_in(texture)
    #    finally:
    #        cls.log_out(texture)


    #_MAX_SCENES: ClassVar[int] = 64                 #  0 ~  63, cannot rewrite
    #_MAX_EXTERNAL_TEXTURES: ClassVar[int] = 32      # 64 ~  95, rewrite when overflow
    #_MAX_INTERMEDIATE_TEXTURES: ClassVar[int] = 32  # 96 ~ 127, always rewrite
    #_SCENES: "ClassVar[list[Scene | None]]" = [None] * _MAX_SCENES
    #_SCENE_PTR: ClassVar[int] = 0
    #_TEXTURE_PATHS: ClassVar[list[str | None]] = [None] * _MAX_EXTERNAL_TEXTURES
    #_TEXTURE_PATH_PTR: ClassVar[int] = 0

    #@classmethod
    #def bind_scene(cls, scene: "Scene") -> int:
    #    if scene in cls._SCENES:
    #        return cls._SCENES.index(scene)
    #    if cls._SCENE_PTR == cls._MAX_SCENES:
    #        raise ValueError(f"Cannot allocate a texture unit for {scene}")
    #    cls._SCENES[cls._SCENE_PTR] = scene
    #    cls._SCENE_PTR += 1
    #    return cls._SCENE_PTR - 1

    #@classmethod
    #def bind_image(cls, path: str) -> int:
    #    if path in cls._TEXTURE_PATHS:
    #        return cls._MAX_SCENES + cls._TEXTURE_PATHS.index(path)
    #    if cls._TEXTURE_PATH_PTR == cls._MAX_EXTERNAL_TEXTURES:
    #        cls._TEXTURE_PATH_PTR = 0
    #    cls._TEXTURE_PATHS[cls._TEXTURE_PATH_PTR] = path
    #    cls._TEXTURE_PATH_PTR += 1
    #    return cls._MAX_SCENES + cls._TEXTURE_PATH_PTR - 1

    #@classmethod
    #def get_intermediate_units(cls, num: int) -> list[int]:
    #    if not 0 <= num < cls._MAX_INTERMEDIATE_TEXTURES:
    #        raise ValueError(f"Cannot allocate {num} intermediate texture units")
    #    head = cls._MAX_SCENES + cls._MAX_EXTERNAL_TEXTURES
    #    return list(range(head, head + num))


class TextureStorage(LazyBase):
    @lazy_property_initializer_writable
    @staticmethod
    def _data_() -> moderngl.Texture | tuple[list[moderngl.Texture], int | str]:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _textures_(data: moderngl.Texture | tuple[list[moderngl.Texture], int | str]) -> list[moderngl.Texture]:
        if isinstance(data, moderngl.Texture):
            return [data]
        textures, array_len = data
        if isinstance(array_len, int):
            assert len(textures) == array_len
        return textures[:]

    #@lazy_property
    #@staticmethod
    #def _array_len_(data: moderngl.Texture | tuple[list[moderngl.Texture], int | str]) -> int | None:
    #    if isinstance(data, moderngl.Texture):
    #        return 1
    #    textures, _ = data
    #    return len(textures)

    @lazy_property
    @staticmethod
    def _dynamic_array_len_(data: moderngl.Texture | tuple[list[moderngl.Texture], int | str]) -> tuple[str, int] | None:
        if isinstance(data, moderngl.Texture):
            return None
        textures, array_len = data
        if isinstance(array_len, int):
            return None
        return (array_len, len(textures))


class UniformBlockBuffer(LazyBase):
    # std140 format
    _SHAPE_TO_SIZE: ClassVar[dict[tuple[int, ...], int]] = {
        (): 1,
        (2,): 2,
        (3,): 4,
        (4,): 4
    }
    _DTYPE_TO_SIZE: ClassVar[dict[npt.DTypeLike, int]] = {
        np.int32: 4,
        np.uint32: 4,
        np.float32: 4,
        np.float64: 8
    }

    #def __init__(self, arrays: list[UniformType]):
    #    super().__init__()
    #    self._arrays_ = arrays

    @lazy_property_initializer_writable
    @staticmethod
    def _data_() -> list[tuple[np.ndarray, npt.DTypeLike, int | str | None]]:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _buffer_(data: list[tuple[np.ndarray, npt.DTypeLike, int | str | None]]) -> moderngl.Buffer:  # TODO: assert size equality
        bytes_offset_pairs: list[tuple[bytes, int]] = []
        offset = 0
        for uniform, dtype, array_len in data:
            bytes_and_base_alignment = UniformBlockBuffer._get_bytes_and_base_alignment(uniform, dtype, array_len)
            if bytes_and_base_alignment is None:
                continue
            bytes_, base_alignment = bytes_and_base_alignment
            offset += (-offset) % base_alignment
            bytes_offset_pairs.append((bytes_, offset))
            offset += len(bytes_)

        offset += (-offset) % 16
        buffer = ContextSingleton().buffer(reserve=offset)
        for bytes_, bytes_offset in bytes_offset_pairs:
            buffer.write(bytes_, offset=bytes_offset)
        return buffer

    @_buffer_.releaser
    @staticmethod
    def _buffer_releaser(buffer: moderngl.Buffer) -> None:
        buffer.release()

    @lazy_property
    @staticmethod
    def _dynamic_array_lens_(data: list[tuple[np.ndarray, npt.DTypeLike, int | str | None]]) -> dict[str, int]:
        return {
            array_len: uniform.shape[0]
            for uniform, _, array_len in data if isinstance(array_len, str)
        }

    @classmethod
    def _get_bytes_and_base_alignment(
        cls, uniform: np.ndarray, dtype: npt.DTypeLike, array_len: int | str | None
    ) -> tuple[bytes, int] | None:
        if not uniform.size:
            assert isinstance(array_len, str)
            return None
        size_from_dtype = cls._DTYPE_TO_SIZE[dtype]
        if array_len is None:
            atom_shape = uniform.shape[:]
            size_from_shape = cls._SHAPE_TO_SIZE.get(atom_shape)
            if size_from_shape is not None:
                return (uniform.astype(dtype, casting="same_kind").tobytes(), size_from_shape * size_from_dtype)
        else:
            atom_shape = uniform.shape[1:]
            if isinstance(array_len, int):
                assert uniform.shape[0] == array_len
        assert len(atom_shape) <= 2 and all(
            axis_size in (2, 3, 4) for axis_size in atom_shape
        )
        padded = uniform[..., None] if not atom_shape else uniform
        if padded.shape[-1] != 4:
            pad_width = ((0, 0),) * (len(padded.shape) - 1) + ((0, 4 - padded.shape[-1]),)
            padded = np.pad(padded, pad_width)
        return (padded.astype(dtype, casting="same_kind").tobytes(), padded.size * size_from_dtype)


class AttributeBuffer(LazyBase):
    _DTYPE_TO_BUFFER_FORMAT_UNIT: ClassVar[dict[npt.DTypeLike, str]] = {
        np.int32: "i4",
        np.uint32: "u4",
        np.float32: "f4",
        np.float64: "f8"
    }

    #def __init__(self, array: AttributeType):
    #    super().__init__()
    #    self._array_ = array

    @lazy_property_initializer_writable
    @staticmethod
    def _data_() -> tuple[np.ndarray, npt.DTypeLike]:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _dimension_(data: tuple[np.ndarray, npt.DTypeLike]) -> int:  # TODO: assert matched with program attribute
        arr, _ = data
        return arr.size // arr.shape[0]

    @lazy_property
    @staticmethod
    def _buffer_format_(dimension: int, data: tuple[np.ndarray, npt.DTypeLike]) -> str:  # TODO: assert this one instead?
        _, dtype = data
        buffer_format_unit = AttributeBuffer._DTYPE_TO_BUFFER_FORMAT_UNIT[dtype]
        return f"{dimension}{buffer_format_unit} /v"

    @lazy_property
    @staticmethod
    def _buffer_(data: tuple[np.ndarray, npt.DTypeLike]) -> moderngl.Buffer:
        arr, dtype = data
        return ContextSingleton().buffer(arr.astype(dtype, casting="same_kind").tobytes())
        #return ContextSingleton().buffer(data.astype(np.float32).tobytes())

    @_buffer_.releaser
    @staticmethod
    def _buffer_releaser(buffer: moderngl.Buffer) -> None:
        buffer.release()


class IndexBuffer(LazyBase):
    #def __init__(self, indices: VertexIndicesType):
    #    super().__init__()
    #    self._indices_ = indices

    @lazy_property_initializer_writable
    @staticmethod
    def _data_() -> VertexIndicesType:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _buffer_(data: VertexIndicesType) -> moderngl.Buffer:
        return ContextSingleton().buffer(data.astype(np.uint32, casting="same_kind").tobytes())

    @_buffer_.releaser
    @staticmethod
    def _buffer_releaser(buffer: moderngl.Buffer) -> None:
        buffer.release()


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

        dynamic_array_lens: dict[str, int] = dict(
            dynamic_array_len
            for texture_storage in texture_storages.values()
            if (dynamic_array_len := texture_storage._dynamic_array_len_) is not None
        )
        for uniform_block in uniform_blocks.values():
            dynamic_array_lens.update(uniform_block._dynamic_array_lens_)
        program = Programs._get_program(shader_strings, dynamic_array_lens)
        program_uniforms, program_uniform_blocks, program_attributes, program_subroutines \
            = cls._get_program_parameters(program)
        assert set(program_uniforms) == set(texture_storages)
        assert set(program_uniform_blocks) == set(uniform_blocks)
        assert set(program_attributes) == set(attributes)
        assert set(program_subroutines) == set(subroutines)
        assert all(
            uniform_block._buffer_.size == program_uniform_blocks[uniform_block_name].size
            for uniform_block_name, uniform_block in uniform_blocks.items()
        )
        assert all(
            attribute._dimension_ == program_attributes[attribute_name].dimension
            for attribute_name, attribute in attributes.items()
        )

        textures: dict[tuple[str, int], moderngl.Texture] = {
            (texture_storage_name, index): texture
            for texture_storage_name, texture_storage in texture_storages.items()
            for index, texture in enumerate(texture_storage._textures_)
        }
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
                    for texture_storage_name, texture_storage in texture_storages.items():
                        texture_bindings = [
                            texture_binding_dict[textures[(texture_storage_name, index)]]
                            for index, _ in enumerate(texture_storage._textures_)
                        ]
                        textures_len = len(texture_storage._textures_)
                        if textures_len == 0:
                            continue
                        if textures_len == 1:
                            value = texture_bindings[0]
                        else:
                            value = texture_bindings
                        program_uniforms[texture_storage_name].value = value
                    for uniform_block_name, uniform_block in uniform_blocks.items():
                        program_uniform_blocks[uniform_block_name].value = uniform_block_binding_dict[uniform_block]
                    vertex_array = ContextSingleton().vertex_array(
                        program=program,
                        content=[
                            (attribute._buffer_, attribute._buffer_format_, attribute_name)
                            for attribute_name, attribute in attributes.items()
                        ],
                        index_buffer=index_buffer._buffer_,
                        mode=mode
                    )
                    vertex_array.subroutines = tuple(
                        program_subroutines[subroutines[subroutine_name]].index
                        for subroutine_name in vertex_array.program.subroutines
                    )
                    vertex_array.render()

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

    #@classmethod
    #def _make_buffer(cls, array: np.ndarray, dtype: DTypeLike) -> moderngl.Buffer:
    #    return ContextSingleton().buffer(array.astype(dtype).tobytes())

    #@classmethod
    #def _make_index_buffer(cls, array: np.ndarray) -> moderngl.Buffer:
    #    return ContextSingleton().buffer(array.astype(np.int32).tobytes())
        #def method(indices: VertexIndicesType) -> moderngl.Buffer:
        #    return ContextSingleton().buffer(indices.astype(np.int32).tobytes())

        #method.__signature__ = inspect.signature(method).replace(
        #    parameters=(inspect.Parameter(array_property.stripped_name, inspect.Parameter.POSITIONAL_ONLY),),
        #)
        #_result_ = lazy_property(staticmethod(method))
        #_result_.releaser(moderngl.Buffer.release)
        #return _result_

    #@classmethod
    #def _make_buffer_block(cls, content: list[tuple[np.ndarray, DTypeLike]]) -> moderngl.Buffer:
    #    # std140 format
    #    bytes_list = [
    #        array.astype(dtype).tobytes()
    #        for array, dtype in content
    #    ]
    #    
    #    buffer = ContextSingleton().buffer(reserve=128)
    #    buffer.write(projection_matrix.tobytes(), offset=0)
    #    buffer.write(view_matrix.tobytes(), offset=64)
    #    return buffer


"""
class Renderable(LazyBase):
    _PROGRAM_CACHE: ClassVar[dict[tuple[str, str | None, str | None, str | None, str | None], moderngl.Program]] = {}
    #_IMAGE_CACHE: ClassVar[dict[str, Image.Image]] = {}

    #@lazy_property_initializer_writable
    #@classmethod
    #def _enable_depth_test_(cls) -> bool:
    #    return True

    #@lazy_property_initializer_writable
    #@classmethod
    #def _enable_blend_(cls) -> bool:
    #    return True

    #@lazy_property_initializer_writable
    #@classmethod
    #def _cull_face_(cls) -> str:
    #    return "back"

    #@lazy_property_initializer_writable
    #@classmethod
    #def _wireframe_(cls) -> bool:
    #    return False

    #@lazy_property_initializer_writable
    #@classmethod
    #def _flags_(cls) -> int:
    #    return moderngl.NOTHING

    #@lazy_property_initializer
    #@staticmethod
    #def _invisible_() -> bool:
    #    return True

    #@lazy_property_initializer
    #@staticmethod
    #def _shader_filename_() -> str:
    #    return NotImplemented

    #@lazy_property_initializer
    #@staticmethod
    #def _define_macros_() -> list[str]:
    #    return NotImplemented

    @lazy_property_initializer
    @classmethod
    def _program_(cls) -> moderngl.Program:
        return NotImplemented

    @lazy_property_initializer
    @classmethod
    def _texture_dict_(cls) -> dict[str, moderngl.Texture]:
        # Only involves external textures and does not contain generated ones
        return NotImplemented

    @lazy_property_initializer
    @classmethod
    def _vbo_dict_(cls) -> dict[str, tuple[moderngl.Buffer, str]]:
        return NotImplemented

    @lazy_property_initializer
    @classmethod
    def _ibo_(cls) -> moderngl.Buffer | None:
        return NotImplemented

    @lazy_property_initializer
    @classmethod
    def _render_primitive_(cls) -> int:
        return NotImplemented

    #@lazy_property_initializer
    #@classmethod
    #def _fbo_(cls) -> moderngl.Framebuffer:
    #    return NotImplemented

    @lazy_property
    @classmethod
    def _vao_(
        cls,
        program: moderngl.Program,
        vbo_dict: dict[str, tuple[moderngl.Buffer, str]],
        ibo: moderngl.Buffer | None,
        render_primitive: int
    ) -> moderngl.VertexArray:
        return ContextSingleton().vertex_array(
            program=program,
            content=[
                (buffer, buffer_format, name)
                for name, (buffer, buffer_format) in vbo_dict.items()
            ],
            index_buffer=ibo,
            index_element_size=4,
            mode=render_primitive
        )

    @_vao_.releaser
    @staticmethod
    def _vao_releaser(vao: moderngl.VertexArray) -> None:
        vao.release()

    #@classmethod
    #def _make_texture(cls, image: skia.Image) -> moderngl.Texture:
    #    return ContextSingleton().texture(
    #        size=(image.width(), image.height()),
    #        components=image.imageInfo().bytesPerPixel(),
    #        data=image.tobytes(),
    #    )

    #@_fbo_.updater
    #def render(self) -> None:
    #    indexed_texture_dict = self._texture_dict_.copy()

    #    self._static_render(
    #        enable_depth_test=self._enable_depth_test_,
    #        enable_blend=self._enable_blend_,
    #        cull_face=self._cull_face_,
    #        wireframe=self._wireframe_,
    #        program=self._program_,
    #        indexed_texture_dict=indexed_texture_dict,
    #        vbo_dict=self._vbo_dict_,
    #        ibo=self._ibo_,
    #        render_primitive=self._render_primitive_,
    #        fbo=self._fbo_,
    #    )

    #@abstractmethod
    #def generate_fbo(self) -> moderngl.Framebuffer:
    #    pass

    def render(
        self,
        uniform_dict: dict[str, UniformType],
        #scope: moderngl.Scope
        #fbo: moderngl.Framebuffer
    ) -> None:
        #ctx = ContextSingleton()
        #scope = ctx.scope(
        #    framebuffer=fbo,
        #    enable_only=self._flags_,
        #    textures=
        #)
        #if self._enable_depth_test_:
        #    ctx.enable(moderngl.DEPTH_TEST)
        #else:
        #    ctx.disable(moderngl.DEPTH_TEST)
        #if self._enable_blend_:
        #    ctx.enable(moderngl.BLEND)
        #else:
        #    ctx.disable(moderngl.BLEND)
        #ctx.cull_face = self._cull_face_
        #ctx.wireframe = self._wireframe_

        #with scope:
        vao = self._vao_
        program = vao.program
        for name, value in uniform_dict.items():
            uniform = program.__getitem__(name)
            if not isinstance(uniform, moderngl.Uniform):
                raise ValueError
            uniform.__setattr__("value", value)
            #texture.use(location=location)

        #fbo.use()
        vao.render()
        #vao.release()

    @classmethod
    def get_program(
        cls,
        vertex_shader: str,
        fragment_shader: str | None = None,
        geometry_shader: str | None = None,
        tess_control_shader: str | None = None,
        tess_evaluation_shader: str | None = None
    ) -> moderngl.Program:
        key = (
            vertex_shader,
            fragment_shader,
            geometry_shader,
            tess_control_shader,
            tess_evaluation_shader
        )
        if key in cls._PROGRAM_CACHE:
            return cls._PROGRAM_CACHE[key]

        program = ContextSingleton().program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            geometry_shader=geometry_shader,
            tess_control_shader=tess_control_shader,
            tess_evaluation_shader=tess_evaluation_shader
        )
        cls._PROGRAM_CACHE[key] = program
        return program

    #@classmethod
    #def _load_image(cls, path: str) -> Image.Image:
    #    key = os.path.abspath(path)
    #    if key in cls._IMAGE_CACHE:
    #        return cls._IMAGE_CACHE[key]

    #    image = Image.open(path)
    #    cls._IMAGE_CACHE[key] = image
    #    return image

    @classmethod
    def _make_buffer(cls, array: AttributeType) -> moderngl.Buffer:
        return ContextSingleton().buffer(array.tobytes())

    #@classmethod
    #@lru_cache(maxsize=8, typed=True)
    #def _read_glsl_file(cls, filename: str) -> str:
    #    with open(os.path.join(SHADERS_PATH, f"{filename}.glsl")) as f:
    #        result = f.read()
    #    return result

    #@classmethod
    #def _insert_defines(cls, content: str, define_macros: list[str]):
    #    version_str, rest = content.split("\n", 1)
    #    return "\n".join([
    #        version_str,
    #        *(
    #            f"#define {define_macro}"
    #            for define_macro in define_macros
    #        ),
    #        rest
    #    ])

    #@classmethod
    #@lru_cache(maxsize=8, typed=True)
    #def _get_program(
    #    cls,
    #    shader_filename: str,
    #    define_macros: tuple[str, ...]  # make hashable
    #) -> moderngl.Program:
    #    content = cls._insert_defines(
    #        cls._read_glsl_file(shader_filename),
    #        list(define_macros)
    #    )
    #    shaders_dict = dict.fromkeys((
    #        "VERTEX_SHADER",
    #        "FRAGMENT_SHADER",
    #        "GEOMETRY_SHADER",
    #        "TESS_CONTROL_SHADER",
    #        "TESS_EVALUATION_SHADER"
    #    ))
    #    for shader_type in shaders_dict:
    #        if re.search(f"\\b{shader_type}\\b", content):
    #            shaders_dict[shader_type] = cls._insert_defines(content, [shader_type])
    #    if shaders_dict["VERTEX_SHADER"] is None:
    #        raise
    #    return ContextSingleton().program(
    #        vertex_shader=shaders_dict["VERTEX_SHADER"],
    #        fragment_shader=shaders_dict["FRAGMENT_SHADER"],
    #        geometry_shader=shaders_dict["GEOMETRY_SHADER"],
    #        tess_control_shader=shaders_dict["TESS_CONTROL_SHADER"],
    #        tess_evaluation_shader=shaders_dict["TESS_EVALUATION_SHADER"]
    #    )

    #@lazy_property
    #@classmethod
    #def _program_(
    #    cls,
    #    shader_filename: str,
    #    define_macros: list[str]
    #) -> moderngl.Program:
    #    return cls._get_program(
    #        shader_filename,
    #        tuple(define_macros)
    #    )

    #@lazy_property
    #@classmethod
    #def _ibo_(
    #    cls,
    #    vertex_indices: VertexIndicesType
    #) -> moderngl.Buffer:
    #    return ContextSingleton().buffer(vertex_indices.tobytes())

    #@lazy_property
    #@classmethod
    #def _vao_(
    #    cls,
    #    buffers_dict: dict[str, tuple[moderngl.Buffer, str]],
    #    program: moderngl.Program,
    #    ibo: moderngl.Buffer,
    #    render_primitive: int
    #) -> moderngl.VertexArray:
    #    return ContextSingleton().vertex_array(
    #        program=program,
    #        content=[
    #            (buffer, buffer_format, name)
    #            for name, (buffer, buffer_format) in buffers_dict.items()
    #        ],
    #        index_buffer=ibo,
    #        index_element_size=4,
    #        mode=render_primitive
    #    )

    #def render(self) -> None:
    #    pass
        #if self._invisible_:
        #    return

        ##if not self._is_expired("_vao_"):
        ##    return

        #ctx = ContextSingleton()
        #if self._enable_depth_test_:
        #    ctx.enable(moderngl.DEPTH_TEST)
        #else:
        #    ctx.disable(moderngl.DEPTH_TEST)
        #if self._enable_blend_:
        #    ctx.enable(moderngl.BLEND)
        #else:
        #    ctx.disable(moderngl.BLEND)
        #ctx.cull_face = self._cull_face_
        #ctx.wireframe = self._wireframe_

        #for name, (texture, location) in self._textures_dict_.items():
        #    uniform = self._program_.__getitem__(name)
        #    if not isinstance(uniform, moderngl.Uniform):
        #        continue
        #    uniform.__setattr__("value", location)
        #    texture.use(location=location)

        #vao = self._vao_
        #vao.render()
        #vao.release()
"""
