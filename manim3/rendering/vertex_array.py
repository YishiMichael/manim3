__all__ = ["VertexArray"]


from dataclasses import dataclass
from functools import reduce
import operator as op
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
    TexturePlaceholders,
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
        texture_placeholders: list[TexturePlaceholders] | None = None,
        uniform_blocks: list[UniformBlockBuffer] | None = None,
        indexed_attributes_buffer: IndexedAttributesBuffer | None = None
    ) -> None:
        super().__init__()
        self._shader_filename_ = shader_filename
        if custom_macros is not None:
            self._custom_macros_ = tuple(custom_macros)
        if texture_placeholders is not None:
            self._texture_placeholders_.add(*texture_placeholders)
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
    def _texture_placeholders_(cls) -> list[TexturePlaceholders]:
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
                        (-1.0, -1.0, 0.0),
                        (1.0, -1.0, 0.0),
                        (1.0, 1.0, 0.0),
                        (-1.0, 1.0, 0.0)
                    )),
                    "in_uv": np.array((
                        (0.0, 0.0),
                        (1.0, 0.0),
                        (1.0, 1.0),
                        (0.0, 1.0)
                    ))
                }
            ),
            index_buffer=IndexBuffer(
                data=np.array((0, 1, 2, 3), dtype=np.uint32)
            ),
            mode=moderngl.TRIANGLE_FAN
        )

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _dynamic_array_lens_(
        cls,
        texture_placeholders__dynamic_array_lens: list[tuple[tuple[str, int], ...]],
        uniform_blocks__dynamic_array_lens: list[tuple[tuple[str, int], ...]],
        indexed_attributes_buffer__attributes_buffer__dynamic_array_lens: list[tuple[str, int]]
    ) -> tuple[tuple[str, int], ...]:
        dynamic_array_lens: dict[str, int] = {}
        for texture_placeholder_dynamic_array_lens in texture_placeholders__dynamic_array_lens:
            dynamic_array_lens.update(dict(texture_placeholder_dynamic_array_lens))
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
    def _texture_placeholder_shapes_(
        cls,
        _texture_placeholders_: list[TexturePlaceholders]
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (texture_placeholder._field_name_.value, texture_placeholder._shape_.value)
            for texture_placeholder in _texture_placeholders_
        )

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _program_data_(
        cls,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        dynamic_array_lens: tuple[tuple[str, int], ...],
        texture_placeholder_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    ) -> ProgramData:

        def construct_moderngl_program(
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

        def set_texture_bindings(
            program: moderngl.Program,
            texture_placeholder_shapes: tuple[tuple[str, tuple[int, ...]], ...]
        ) -> dict[str, int]:
            texture_placeholder_shape_dict = dict(texture_placeholder_shapes)
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
                texture_placeholder_name = match_obj.group("texture_name")
                texture_placeholder_shape = texture_placeholder_shape_dict[texture_placeholder_name]
                if texture_placeholder_name not in texture_binding_offset_dict:
                    texture_binding_offset_dict[texture_placeholder_name] = binding_offset
                    binding_offset += cls._int_prod(texture_placeholder_shape)
                multi_index = tuple(
                    int(index_match.group(1))
                    for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
                )
                if not texture_placeholder_shape:
                    assert not multi_index
                    uniform_size = 1
                    local_offset = 0
                else:
                    *dims, uniform_size = texture_placeholder_shape
                    local_offset = np.ravel_multi_index(multi_index, dims) * uniform_size
                assert member.array_length == uniform_size
                offset = texture_binding_offset_dict[texture_placeholder_name] + local_offset
                member.value = offset if uniform_size == 1 else list(range(offset, offset + uniform_size))
            return texture_binding_offset_dict

        def set_uniform_block_bindings(
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

        with ConfigSingleton().path.shaders_dir.joinpath(f"{shader_filename}.glsl").open() as shader_file:
            shader_str = shader_file.read()
        program = construct_moderngl_program(shader_str, custom_macros, dynamic_array_lens)
        texture_binding_offset_dict = set_texture_bindings(program, texture_placeholder_shapes)
        uniform_block_binding_dict = set_uniform_block_bindings(program)
        return ProgramData(
            program=program,
            texture_binding_offset_dict=texture_binding_offset_dict,
            uniform_block_binding_dict=uniform_block_binding_dict
        )

    @_program_data_.finalizer
    @classmethod
    def _program_data_finalizer(
        cls,
        program_data: ProgramData
    ) -> None:
        program_data.program.release()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _vertex_array_(
        cls,
        program_data: ProgramData,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> moderngl.VertexArray | None:

        def validate_attributes_buffer(
            attributes_buffer: AttributesBuffer,
            program_attributes: dict[str, moderngl.Attribute]
        ) -> None:
            vertex_dtype = attributes_buffer._vertex_dtype_.value
            for attribute_name, attribute in program_attributes.items():
                field_dtype = vertex_dtype[attribute_name]
                assert attribute.array_length == cls._int_prod(field_dtype.shape)
                assert attribute.dimension == cls._int_prod(field_dtype.base.shape) * cls._int_prod(field_dtype.base["_"].shape)
                assert attribute.shape == field_dtype.base["_"].base.kind.replace("u", "I")

        def get_buffer_format(
            vertex_dtype: np.dtype,
            attribute_name_tuple: tuple[str, ...]
        ) -> tuple[str, list[str]]:
            # TODO: This may require refactory.
            #vertex_dtype = self._vertex_dtype_.value
            vertex_fields = vertex_dtype.fields
            assert vertex_fields is not None
            dtype_stack: list[tuple[np.dtype, int]] = []
            attribute_names: list[str] = []
            for field_name, (field_dtype, field_offset, *_) in vertex_fields.items():
                if field_name not in attribute_name_tuple:
                    continue
                dtype_stack.append((field_dtype, field_offset))
                attribute_names.append(field_name)

            components: list[str] = []
            current_offset = 0
            while dtype_stack:
                dtype, offset = dtype_stack.pop(0)
                dtype_size = cls._int_prod(dtype.shape)
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
        validate_attributes_buffer(attributes_buffer, program_attributes)
        buffer_format, attribute_names = get_buffer_format(attributes_buffer._vertex_dtype_.value, tuple(program_attributes))
        return Context.vertex_array(
            program=moderngl_program,
            attributes_buffer=attributes_buffer._buffer_.value,
            buffer_format=buffer_format,
            attribute_names=attribute_names,
            index_buffer=index_buffer._buffer_.value,
            mode=mode
        )

    @_vertex_array_.finalizer
    @classmethod
    def _vertex_array_finalizer(
        cls,
        vertex_array: moderngl.VertexArray | None
    ) -> None:
        if vertex_array is not None:
            vertex_array.release()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _texture_binding_items_(
        cls,
        _texture_placeholders_: list[TexturePlaceholders],
        program_data: ProgramData
    ) -> dict[str, tuple[tuple[int, ...], int]]:
        texture_placeholder_dict = {
            texture_placeholder._field_name_.value: texture_placeholder
            for texture_placeholder in _texture_placeholders_
        }
        texture_binding_item: dict[str, tuple[tuple[int, ...], int]] = {}
        for texture_placeholder_name, binding_offset in program_data.texture_binding_offset_dict.items():
            texture_placeholder = texture_placeholder_dict[texture_placeholder_name]
            assert not texture_placeholder._is_empty_.value
            texture_binding_item[texture_placeholder_name] = (texture_placeholder._shape_.value, binding_offset)
        return texture_binding_item

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _uniform_block_bindings_(
        cls,
        _uniform_blocks_: list[UniformBlockBuffer],
        program_data: ProgramData
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:

        def validate_uniform_block(
            uniform_block: UniformBlockBuffer,
            program_uniform_block: moderngl.UniformBlock
        ) -> None:
            assert program_uniform_block.name == uniform_block._field_name_.value
            assert program_uniform_block.size == uniform_block._itemsize_.value

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
            validate_uniform_block(uniform_block, program_uniform_block)
            uniform_block_bindings.append((uniform_block._buffer_.value, binding))
        return tuple(uniform_block_bindings)

    @classmethod
    def _int_prod(
        cls,
        shape: tuple[int, ...]
    ) -> int:
        return reduce(op.mul, shape, 1)

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
        for texture_placeholder_name, (shape, binding_offset) in self._texture_binding_items_.value.items():
            texture_array = texture_array_dict[texture_placeholder_name]
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
