__all__ = [
    "ContextState",
    "VertexArray"
]


from dataclasses import dataclass
from functools import reduce
import operator as op
import os
import re

import moderngl
import numpy as np

from ..lazy.core import (
    LazyCollection,
    LazyObject
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..rendering.config import ConfigSingleton
from ..rendering.context import ContextSingleton
from ..rendering.glsl_buffers import (
    AttributesBuffer,
    IndexBuffer,
    TextureStorage,
    UniformBlockBuffer
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ContextState:
    enable_only: int
    depth_func: str = "<"
    blend_func: tuple[int, int] | tuple[int, int, int, int] = moderngl.DEFAULT_BLENDING
    blend_equation: int | tuple[int, int] = moderngl.FUNC_ADD
    front_face: str = "ccw"
    cull_face: str = "back"
    wireframe: bool = False


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProgramData:
    program: moderngl.Program
    texture_binding_offset_dict: dict[str, int]
    uniform_block_binding_dict: dict[str, int]


class IndexedAttributesBuffer(LazyObject):
    def __init__(
        self,
        *,
        attributes: AttributesBuffer,
        index_buffer: IndexBuffer,
        mode: int
    ) -> None:
        super().__init__()
        self._attributes_ = attributes
        self._index_buffer_ = index_buffer
        self._mode_ = mode

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _attributes_(cls) -> AttributesBuffer:
        return NotImplemented

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _index_buffer_(cls) -> IndexBuffer:
        return NotImplemented

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _mode_(cls) -> int:
        return NotImplemented


class VertexArray(LazyObject):
    __slots__ = ()

    #def __init__(
    #    self,
    #    *,
    #    shader_filename: str,
    #    custom_macros: list[str],
    #    texture_storages: list[TextureStorage],
    #    uniform_blocks: list[UniformBlockBuffer],
    #    indexed_attributes: IndexedAttributesBuffer
    #    #attributes: AttributesBuffer,
    #    #index_buffer: IndexBuffer,
    #    #mode: int
    #) -> None:
    #    super().__init__()
    #    self._shader_filename_ = shader_filename
    #    self._custom_macros_ = tuple(custom_macros)
    #    self._texture_storages_.add(*texture_storages)
    #    self._uniform_blocks_.add(*uniform_blocks)
    #    self._indexed_attributes_ = indexed_attributes
    #    #self._index_buffer_ = index_buffer
    #    #self._mode_ = mode

    #@staticmethod
    #def __shader_filename_key(
    #    shader_filename: str
    #) -> str:
    #    return shader_filename

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _shader_filename_(cls) -> str:
        return NotImplemented

    #@staticmethod
    #def __custom_macros_key(
    #    custom_macros: list[str]
    #) -> tuple[str, ...]:
    #    return tuple(custom_macros)

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _custom_macros_(cls) -> tuple[str, ...]:
        return NotImplemented

    #@staticmethod
    #def __dynamic_array_lens_key(
    #    dynamic_array_lens: dict[str, int]
    #) -> tuple[tuple[str, int], ...]:
    #    return tuple(dynamic_array_lens.items())

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _texture_storages_(cls) -> LazyCollection[TextureStorage]:
        return LazyCollection()

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _uniform_blocks_(cls) -> LazyCollection[UniformBlockBuffer]:
        return LazyCollection()

    #@staticmethod
    #def __texture_storage_shapes_key(
    #    texture_storage_shapes: dict[str, tuple[int, ...]]
    #) -> tuple[tuple[str, tuple[int, ...]], ...]:
    #    return tuple(texture_storage_shapes.items())

    #@staticmethod
    #def __mode_key(
    #    mode: int
    #) -> int:
    #    return mode

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _indexed_attributes_(cls) -> IndexedAttributesBuffer:
        return NotImplemented

    #@Lazy.variable(LazyMode.OBJECT)
    #@classmethod
    #def _index_buffer_(cls) -> IndexBuffer:
    #    return NotImplemented

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _mode_(cls) -> int:
    #    return NotImplemented

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _dynamic_array_lens_(
        cls,
        texture_storages__dynamic_array_lens: list[tuple[tuple[str, int], ...]],
        uniform_blocks__dynamic_array_lens: list[tuple[tuple[str, int], ...]],
        indexed_attributes__attributes__dynamic_array_lens: list[tuple[str, int]]
    ) -> tuple[tuple[str, int], ...]:
        dynamic_array_lens: dict[str, int] = {}
        for texture_storage_dynamic_array_lens in texture_storages__dynamic_array_lens:
            dynamic_array_lens.update(dict(texture_storage_dynamic_array_lens))
        for uniform_block_dynamic_array_lens in uniform_blocks__dynamic_array_lens:
            dynamic_array_lens.update(uniform_block_dynamic_array_lens)
        dynamic_array_lens.update(dict(indexed_attributes__attributes__dynamic_array_lens))
        return tuple(
            (array_len_name, array_len)
            for array_len_name, array_len in dynamic_array_lens.items()
            if not re.fullmatch(r"__\w+__", array_len_name)
        )

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _texture_storage_shapes_(
        cls,
        _texture_storages_: LazyCollection[TextureStorage]
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (texture_storage._field_name_.value, texture_storage._texture_array_.value.shape)
            for texture_storage in _texture_storages_
        )

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _program_data_(
        cls,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        dynamic_array_lens: tuple[tuple[str, int], ...],
        texture_storage_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    ) -> ProgramData:
        #print("=" * 100)
        #print(shader_filename
        #    ,custom_macros
        #    ,dynamic_array_lens
        #    ,texture_storage_shapes)
        with open(os.path.join(ConfigSingleton().shaders_dir, f"{shader_filename}.glsl")) as shader_file:
            shader_str = shader_file.read()
        program = cls._construct_moderngl_program(shader_str, custom_macros, dynamic_array_lens)
        texture_binding_offset_dict = cls._set_texture_bindings(program, texture_storage_shapes)
        uniform_block_binding_dict = cls._set_uniform_block_bindings(program)
        #print(program)
        return ProgramData(
            program=program,
            texture_binding_offset_dict=texture_binding_offset_dict,
            uniform_block_binding_dict=uniform_block_binding_dict
        )

    @_program_data_.releaser
    @classmethod
    def _program_data_releaser(
        cls,
        program_data: ProgramData
    ) -> None:
        # TODO: check refcnt
        program_data.program.release()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _vertex_array_(
        cls,
        program_data: ProgramData,
        _indexed_attributes_: IndexedAttributesBuffer
        #_attributes_: AttributesBuffer,
        #_index_buffer_: IndexBuffer,
        #mode: int
    ) -> moderngl.VertexArray | None:
        attributes = _indexed_attributes_._attributes_
        index_buffer = _indexed_attributes_._index_buffer_
        mode = _indexed_attributes_._mode_.value

        if attributes._is_empty_.value or index_buffer._is_empty_.value:
            return None

        moderngl_program = program_data.program
        program_attributes = {
            name: member
            for name in moderngl_program
            if isinstance(member := moderngl_program[name], moderngl.Attribute)
        }
        attributes._validate(program_attributes)
        buffer_format, attribute_names = attributes._get_buffer_format(tuple(program_attributes))
        return ContextSingleton().vertex_array(
            program=moderngl_program,
            content=[(attributes._buffer_.value, buffer_format, *attribute_names)],
            index_buffer=index_buffer._buffer_.value,
            mode=mode
        )

    @_vertex_array_.releaser
    @classmethod
    def _vertex_array_releaser(
        cls,
        vertex_array: moderngl.VertexArray | None
    ) -> None:
        # TODO: check refcnt
        #import sys
        #print(sys.getrefcount(vertex_array))
        if vertex_array is not None:
            vertex_array.release()

    @classmethod
    def _construct_moderngl_program(
        cls,
        shader_str: str,
        custom_macros: tuple[str, ...],
        dynamic_array_lens: tuple[tuple[str, int], ...]
    ) -> moderngl.Program:
        version_string = f"#version {ContextSingleton().version_code} core"
        array_len_macros = [
            f"#define {array_len_name} {array_len}"
            for array_len_name, array_len in dynamic_array_lens
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
    def _set_texture_bindings(
        cls,
        program: moderngl.Program,
        texture_storage_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    ) -> dict[str, int]:
        texture_storage_shape_dict = dict(texture_storage_shapes)
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
                binding_offset += cls._int_prod(texture_storage_shape)
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
    def _set_uniform_block_bindings(
        cls,
        program: moderngl.Program
    ) -> dict[str, int]:
        uniform_block_binding_dict: dict[str, int] = {}
        binding = 0
        for name in program:
            member = program[name]
            if not isinstance(member, moderngl.UniformBlock):
                continue
            # Ensure the name doesn't contain wierd symbols like `[]`
            assert re.fullmatch(r"\w+", name) is not None
            uniform_block_binding_dict[name] = binding
            member.binding = binding
            binding += 1
        return uniform_block_binding_dict

    @classmethod
    def _int_prod(
        cls,
        shape: tuple[int, ...]
    ) -> int:
        return reduce(op.mul, shape, 1)  # TODO: redundant with the one in glsl_buffers.py

    def write(
        self,
        *,
        shader_filename: str,
        custom_macros: list[str],
        texture_storages: list[TextureStorage],
        uniform_blocks: list[UniformBlockBuffer],
        indexed_attributes: IndexedAttributesBuffer
    ):
        self._shader_filename_ = shader_filename
        self._custom_macros_ = tuple(custom_macros)
        self._texture_storages_ = LazyCollection(*texture_storages)
        self._uniform_blocks_ = LazyCollection(*uniform_blocks)
        self._indexed_attributes_ = indexed_attributes
        return self

    def render(
        self,
        *,
        #shader_filename: str,
        #custom_macros: list[str],
        #texture_storages: list[TextureStorage],
        #uniform_blocks: list[UniformBlockBuffer],
        #indexed_attributes: IndexedAttributesBuffer,
        #shader_filename: str,
        #custom_macros: list[str],
        #texture_storages: list[TextureStorage],
        #uniform_blocks: list[UniformBlockBuffer],
        #texture_array_dict: dict[str, np.ndarray],
        framebuffer: moderngl.Framebuffer,
        context_state: ContextState
    ) -> None:
        #dynamic_array_lens: dict[str, int] = {}
        #for texture_storage in texture_storages:
        #    dynamic_array_lens.update(texture_storage._dynamic_array_lens_.value)
        #for uniform_block in uniform_blocks:
        #    dynamic_array_lens.update(uniform_block._dynamic_array_lens_.value)
        #dynamic_array_lens.update(self._attributes_._dynamic_array_lens_.value)

        #self._shader_filename_ = shader_filename
        #self._custom_macros_ = tuple(custom_macros)
        #self._dynamic_array_lens_ = tuple(
        #    (array_len_name, array_len)
        #    for array_len_name, array_len in dynamic_array_lens.items()
        #    if not re.fullmatch(r"__\w+__", array_len_name)
        #)
        #self._texture_storage_shapes_ = tuple(
        #    (texture_storage._field_name_.value, texture_storage._texture_array_.value.shape)
        #    for texture_storage in texture_storages
        #)

        #self._texture_storages_ = LazyCollection(*texture_storages)

        if self._vertex_array_.value is None:
            return

        #print()
        #print(111)
        #print(
        #    self._shader_filename_.value,
        #    self._custom_macros_.value,
        #    self._dynamic_array_lens_.value,
        #    self._texture_storage_shapes_.value
        #)
        program_data = self._program_data_.value
        #print(program_data)

        # texture storages
        texture_storage_dict = {
            texture_storage._field_name_.value: texture_storage
            for texture_storage in self._texture_storages_
        }
        #print(texture_storage_dict)
        #print(program_data)
        #print(program_data.texture_binding_offset_dict)
        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_storage_name, binding_offset in program_data.texture_binding_offset_dict.items():
            texture_storage = texture_storage_dict[texture_storage_name]
            assert not texture_storage._is_empty_.value
            #texture_array = texture_array_dict[texture_storage_name]
            #assert texture_array.shape == texture_storage._shape_.value
            texture_bindings.extend(
                (texture, binding)
                for binding, texture in enumerate(texture_storage._texture_array_.value.flat, start=binding_offset)
            )

        # uniform blocks
        uniform_block_dict = {
            uniform_block._field_name_.value: uniform_block
            for uniform_block in self._uniform_blocks_
        }
        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
        for uniform_block_name, binding in program_data.uniform_block_binding_dict.items():
            uniform_block = uniform_block_dict[uniform_block_name]
            assert not uniform_block._is_empty_.value
            program_uniform_block = program_data.program[uniform_block_name]
            assert isinstance(program_uniform_block, moderngl.UniformBlock)
            uniform_block._validate(program_uniform_block)
            uniform_block_bindings.append((uniform_block._buffer_.value, binding))

        context = ContextSingleton()
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
            self._vertex_array_.value.render()
