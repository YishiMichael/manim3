__all__ = ["VertexArray"]


from dataclasses import dataclass
from functools import reduce
import operator as op
import os
import re

import moderngl
import numpy as np

from ..lazy.core import LazyObject
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..rendering.config import ConfigSingleton
from ..rendering.context import (
    Context,
    ContextState
)
from ..rendering.gl_buffer import (
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
class ProgramData:
    program: moderngl.Program
    texture_binding_offset_dict: dict[str, int]
    uniform_block_binding_dict: dict[str, int]


class IndexedAttributesBuffer(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        attributes_buffer: AttributesBuffer,
        index_buffer: IndexBuffer,
        mode: int
    ) -> None:
        super().__init__()
        self._attributes_buffer_ = attributes_buffer
        self._index_buffer_ = index_buffer
        self._mode_ = mode

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _attributes_buffer_(cls) -> AttributesBuffer:
        return AttributesBuffer(
            fields=[],
            num_vertex=0,
            data={}
        )

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _index_buffer_(cls) -> IndexBuffer:
        return IndexBuffer(
            data=np.zeros((0, 1), dtype=np.uint32)
        )

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _mode_(cls) -> int:
        return moderngl.TRIANGLES


class VertexArray(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        shader_filename: str,
        custom_macros: list[str] | None = None,
        texture_storages: list[TextureStorage] | None = None,
        uniform_blocks: list[UniformBlockBuffer] | None = None,
        indexed_attributes_buffer: IndexedAttributesBuffer | None = None
    ) -> None:
        super().__init__()
        self._shader_filename_ = shader_filename
        if custom_macros is not None:
            self._custom_macros_ = tuple(custom_macros)
        if texture_storages is not None:
            self._texture_storages_.add(*texture_storages)
        if uniform_blocks is not None:
            self._uniform_blocks_.add(*uniform_blocks)
        if indexed_attributes_buffer is not None:
            self._indexed_attributes_buffer_ = indexed_attributes_buffer

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _shader_filename_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _custom_macros_(cls) -> tuple[str, ...]:
        return ()

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _texture_storages_(cls) -> list[TextureStorage]:
        return []

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _uniform_blocks_(cls) -> list[UniformBlockBuffer]:
        return []

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _indexed_attributes_buffer_(cls) -> IndexedAttributesBuffer:
        # For full-screen rendering.
        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
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
            ),
            index_buffer=IndexBuffer(
                data=np.array((
                    0, 1, 2, 3
                ))
            ),
            mode=moderngl.TRIANGLE_FAN
        )

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _dynamic_array_lens_(
        cls,
        texture_storages__dynamic_array_lens: list[tuple[tuple[str, int], ...]],
        uniform_blocks__dynamic_array_lens: list[tuple[tuple[str, int], ...]],
        indexed_attributes_buffer__attributes_buffer__dynamic_array_lens: list[tuple[str, int]]
    ) -> tuple[tuple[str, int], ...]:
        dynamic_array_lens: dict[str, int] = {}
        for texture_storage_dynamic_array_lens in texture_storages__dynamic_array_lens:
            dynamic_array_lens.update(dict(texture_storage_dynamic_array_lens))
        for uniform_block_dynamic_array_lens in uniform_blocks__dynamic_array_lens:
            dynamic_array_lens.update(uniform_block_dynamic_array_lens)
        dynamic_array_lens.update(dict(indexed_attributes_buffer__attributes_buffer__dynamic_array_lens))
        return tuple(
            (array_len_name, array_len)
            for array_len_name, array_len in dynamic_array_lens.items()
            if not re.fullmatch(r"__\w+__", array_len_name)
        )

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _texture_storage_shapes_(
        cls,
        _texture_storages_: list[TextureStorage]
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (texture_storage._field_name_.value, texture_storage._shape_.value)
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
        with open(os.path.join(ConfigSingleton().shaders_dir, f"{shader_filename}.glsl")) as shader_file:
            shader_str = shader_file.read()
        program = cls._construct_moderngl_program(shader_str, custom_macros, dynamic_array_lens)
        texture_binding_offset_dict = cls._set_texture_bindings(program, texture_storage_shapes)
        uniform_block_binding_dict = cls._set_uniform_block_bindings(program)
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
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> moderngl.VertexArray | None:
        attributes_buffer = _indexed_attributes_buffer_._attributes_buffer_
        index_buffer = _indexed_attributes_buffer_._index_buffer_
        mode = _indexed_attributes_buffer_._mode_.value

        if attributes_buffer._is_empty_.value or index_buffer._is_empty_.value:
            return None

        moderngl_program = program_data.program
        program_attributes = {
            name: member
            for name in moderngl_program
            if isinstance(member := moderngl_program[name], moderngl.Attribute)
        }
        attributes_buffer._validate(program_attributes)
        buffer_format, attribute_names = attributes_buffer._get_buffer_format(tuple(program_attributes))
        return Context.vertex_array(
            program=moderngl_program,
            attributes_buffer=attributes_buffer._buffer_.value,
            buffer_format=buffer_format,
            attribute_names=attribute_names,
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

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _texture_binding_items_(
        cls,
        _texture_storages_: list[TextureStorage],
        program_data: ProgramData
    ) -> dict[str, tuple[tuple[int, ...], int]]:
        texture_storage_dict = {
            texture_storage._field_name_.value: texture_storage
            for texture_storage in _texture_storages_
        }
        texture_binding_item: dict[str, tuple[tuple[int, ...], int]] = {}
        for texture_storage_name, binding_offset in program_data.texture_binding_offset_dict.items():
            texture_storage = texture_storage_dict[texture_storage_name]
            assert not texture_storage._is_empty_.value
            texture_binding_item[texture_storage_name] = (texture_storage._shape_.value, binding_offset)
        return texture_binding_item

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _uniform_block_bindings_(
        cls,
        _uniform_blocks_: list[UniformBlockBuffer],
        program_data: ProgramData
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:
        uniform_block_dict = {
            uniform_block._field_name_.value: uniform_block
            for uniform_block in _uniform_blocks_
        }
        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
        for uniform_block_name, binding in program_data.uniform_block_binding_dict.items():
            uniform_block = uniform_block_dict[uniform_block_name]
            assert not uniform_block._is_empty_.value
            program_uniform_block = program_data.program[uniform_block_name]
            assert isinstance(program_uniform_block, moderngl.UniformBlock)
            uniform_block._validate(program_uniform_block)
            uniform_block_bindings.append((uniform_block._buffer_.value, binding))
        return tuple(uniform_block_bindings)

    @classmethod
    def _construct_moderngl_program(
        cls,
        shader_str: str,
        custom_macros: tuple[str, ...],
        dynamic_array_lens: tuple[tuple[str, int], ...]
    ) -> moderngl.Program:
        version_string = f"#version {Context.mgl_context.version_code} core"
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
        return Context.program(
            vertex_shader=shaders["VERTEX_SHADER"],
            fragment_shader=shaders.get("FRAGMENT_SHADER"),
            geometry_shader=shaders.get("GEOMETRY_SHADER"),
            tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
            tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER"),
        )

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
            # Used as a `sampler2D`.
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
                *dims, uniform_size = texture_storage_shape
                local_offset = np.ravel_multi_index(multi_index, dims) * uniform_size
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
            # Ensure the name doesn't contain wierd symbols like `[]`.
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
        return reduce(op.mul, shape, 1)  # TODO: redundant with the one in gl_buffer.py

    def render(
        self,
        *,
        # Note, redundant textures are currently not supported.
        texture_array_dict: dict[str, np.ndarray] | None = None,
        framebuffer: moderngl.Framebuffer,
        context_state: ContextState
    ) -> None:
        if (vertex_array := self._vertex_array_.value) is None:
            return

        if texture_array_dict is None:
            texture_array_dict = {}

        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_storage_name, (shape, binding_offset) in self._texture_binding_items_.value.items():
            texture_array = texture_array_dict[texture_storage_name]
            assert shape == texture_array.shape
            texture_bindings.extend(
                (texture, binding)
                for binding, texture in enumerate(texture_array.flat, start=binding_offset)
            )

        Context.set_state(context_state)
        with Context.mgl_context.scope(
            framebuffer=framebuffer,
            enable_only=context_state.enable_only,
            textures=tuple(texture_bindings),
            uniform_buffers=self._uniform_block_bindings_.value
        ):
            vertex_array.render()
