import itertools as it
import re
from dataclasses import dataclass

import moderngl
import numpy as np

from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from ..toplevel.toplevel import Toplevel
from ..utils.path import PathUtils
from .buffer_formats.atomic_buffer_format import AtomicBufferFormat
from .buffer_formats.buffer_format import BufferFormat
from .buffer_formats.structured_buffer_format import StructuredBufferFormat
from .buffers.attributes_buffer import AttributesBuffer
from .buffers.omitted_index_buffer import OmittedIndexBuffer
from .buffers.texture_buffer import TextureBuffer
from .buffers.transform_feedback_buffer import TransformFeedbackBuffer
from .buffers.uniform_block_buffer import UniformBlockBuffer
from .framebuffers.framebuffer import Framebuffer
from .indexed_attributes_buffer import IndexedAttributesBuffer
from .mgl_enums import PrimitiveMode


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProgramAttributeInfo:
    array_length: int
    dimension: int
    shape: str
    #__slots__ = (
    #    "_array_length",
    #    "_dimension",
    #    "_shape"
    #)

    #def __init__(
    #    self,
    #    attribute: moderngl.Attribute
    #) -> None:
    #    super().__init__()
    #    self._array_length: int = attribute.array_length
    #    self._dimension: int = attribute.dimension
    #    self._shape: str = attribute.shape

    def verify_buffer_format(
        self,
        buffer_format: BufferFormat
    ) -> bool:
        assert isinstance(buffer_format, AtomicBufferFormat)
        assert not buffer_format._is_empty_
        assert buffer_format._size_ == self.array_length
        assert buffer_format._n_col_ * buffer_format._n_row_ == self.dimension
        assert buffer_format._base_char_.replace("u", "I") == self.shape
        return True


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProgramUniformInfo:
    array_length: int
    shape: tuple[int, ...]
    binding: int
    #__slots__ = (
    #    "_array_length",
    #    "_shape",
    #    #"_size",
    #    "_binding"
    #)

    #def __init__(
    #    self,
    #    uniform_dict: dict[tuple[int, ...], moderngl.Uniform],
    #    binding: int
    #) -> None:
    #    assert uniform_dict
    #    unique_array_lengths = list(dict.fromkeys(
    #        uniform.array_length
    #        for uniform in uniform_dict.values()
    #    ))
    #    array_length = unique_array_lengths.pop()
    #    assert not unique_array_lengths
    #    shape = tuple(
    #        max(indices) + 1
    #        for indices in zip(*uniform_dict)
    #    )

    #    value = binding
    #    for multi_index in it.product(*(range(n) for n in shape)):
    #        uniform = uniform_dict[multi_index]
    #        assert uniform.dimension == 1
    #        uniform.value = value if array_length == 1 else list(range(value, value + array_length))
    #        value += array_length

    #    #size = reduce(op.mul, shape, 1)
    #    #assert size == len(uniform_dict)
    #    super().__init__()
    #    self._array_length: int = array_length
    #    self._shape: tuple[int, ...] = shape
    #    #self._size: int = size
    #    self._binding: int = binding

    def verify_buffer_format(
        self,
        buffer_format: BufferFormat
    ) -> bool:
        assert not buffer_format._is_empty_
        if (buffer_format_shape := buffer_format._shape_):
            *shape, array_length = buffer_format_shape
        else:
            shape = ()
            array_length = 1
        assert shape == self.shape
        assert array_length == self.array_length
        return True


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProgramUniformBlockInfo:
    size: int
    binding: int
    #__slots__ = (
    #    "_size",
    #    "_binding"
    #)

    #def __init__(
    #    self,
    #    size: int,
    #    binding: int
    #) -> None:
    #    super().__init__()
    #    self._size: int = size
    #    self._binding: int = binding

    def verify_buffer_format(
        self,
        buffer_format: BufferFormat
    ) -> bool:
        assert not buffer_format._is_empty_
        assert buffer_format._nbytes_ == self.size
        return True


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProgramInfo:
    program: moderngl.Program
    attribute_info_dict: dict[str, ProgramAttributeInfo]
    uniform_info_dict: dict[str, ProgramUniformInfo]
    uniform_block_info_dict: dict[str, ProgramUniformBlockInfo]
    #texture_binding_offset_dict: dict[str, int]
    #uniform_block_binding_dict: dict[str, int]


class VertexArray(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        shader_filename: str,
        custom_macros: list[str] | None = None,
        texture_buffers: list[TextureBuffer] | None = None,
        uniform_block_buffers: list[UniformBlockBuffer] | None = None,
        indexed_attributes_buffer: IndexedAttributesBuffer | None = None,
        transform_feedback_buffer: TransformFeedbackBuffer | None = None
    ) -> None:
        super().__init__()
        self._shader_filename_ = shader_filename
        if custom_macros is not None:
            self._custom_macros_ = tuple(custom_macros)
        if texture_buffers is not None:
            self._texture_buffers_.reset(texture_buffers)
        if uniform_block_buffers is not None:
            self._uniform_block_buffers_.reset(uniform_block_buffers)
        if indexed_attributes_buffer is not None:
            self._indexed_attributes_buffer_ = indexed_attributes_buffer
        if transform_feedback_buffer is not None:
            self._transform_feedback_buffer_ = transform_feedback_buffer

    @Lazy.variable_hashable
    @classmethod
    def _shader_filename_(cls) -> str:
        return ""

    @Lazy.variable_hashable
    @classmethod
    def _custom_macros_(cls) -> tuple[str, ...]:
        return ()

    @Lazy.variable_collection
    @classmethod
    def _texture_buffers_(cls) -> list[TextureBuffer]:
        return []

    @Lazy.variable_collection
    @classmethod
    def _uniform_block_buffers_(cls) -> list[UniformBlockBuffer]:
        return []

    @Lazy.variable
    @classmethod
    def _indexed_attributes_buffer_(cls) -> IndexedAttributesBuffer:
        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=[],
                num_vertex=0,
                data={}
            ),
            mode=PrimitiveMode.POINTS
        )

    @Lazy.variable
    @classmethod
    def _transform_feedback_buffer_(cls) -> TransformFeedbackBuffer:
        return TransformFeedbackBuffer(
            fields=[],
            num_vertex=0
        )

    @Lazy.property_hashable
    @classmethod
    def _array_len_items_(
        cls,
        texture_buffers__array_len_items: list[tuple[tuple[str, int], ...]],
        uniform_block_buffers__array_len_items: list[tuple[tuple[str, int], ...]],
        indexed_attributes_buffer__attributes_buffer__array_len_items: tuple[tuple[str, int], ...],
        transform_feedback_buffer__array_len_items: tuple[tuple[str, int], ...]
    ) -> tuple[tuple[str, int], ...]:
        return tuple(
            (array_len_name, array_len)
            for array_len_name, array_len in it.chain(
                it.chain.from_iterable(texture_buffers__array_len_items),
                it.chain.from_iterable(uniform_block_buffers__array_len_items),
                indexed_attributes_buffer__attributes_buffer__array_len_items,
                transform_feedback_buffer__array_len_items
            )
            if not re.fullmatch(r"__\w+__", array_len_name)
        )

    @Lazy.property_external
    @classmethod
    def _program_info_(
        cls,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...],
        #texture_buffers__buffer_format: list[BufferFormat],
        transform_feedback_buffer__np_buffer_pointer_keys: tuple[str, ...]
    ) -> ProgramInfo:

        def read_shader_with_includes_replaced(
            filename: str
        ) -> str:
            with PathUtils.shaders_dir.joinpath(filename).open() as shader_file:
                shader_str = shader_file.read()
            return re.sub(
                r"#include \"(.+?)\"",
                lambda match_obj: read_shader_with_includes_replaced(match_obj.group(1)),
                shader_str
            )

        #def construct_moderngl_program(
        #    shader_str: str,
        #    custom_macros: tuple[str, ...],
        #    array_len_items: tuple[tuple[str, int], ...],
        #    varyings: tuple[str, ...]
        #) -> moderngl.Program:
        #    version_string = f"#version {Toplevel.context.version_code} core"
        #    array_len_macros = [
        #        f"#define {array_len_name} {array_len}"
        #        for array_len_name, array_len in array_len_items
        #    ]
        #    shaders = {
        #        shader_type: "\n".join([
        #            version_string,
        #            "\n",
        #            f"#define {shader_type}",
        #            *custom_macros,
        #            *array_len_macros,
        #            "\n",
        #            shader_str
        #        ])
        #        for shader_type in (
        #            "VERTEX_SHADER",
        #            "FRAGMENT_SHADER",
        #            "GEOMETRY_SHADER",
        #            "TESS_CONTROL_SHADER",
        #            "TESS_EVALUATION_SHADER"
        #        )
        #        if re.search(rf"\b{shader_type}\b", shader_str, flags=re.MULTILINE) is not None
        #    }
        #    return Toplevel.context.program(
        #        vertex_shader=shaders["VERTEX_SHADER"],
        #        fragment_shader=shaders.get("FRAGMENT_SHADER"),
        #        geometry_shader=shaders.get("GEOMETRY_SHADER"),
        #        tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
        #        tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER"),
        #        varyings=varyings
        #    )

        #def set_texture_bindings(
        #    program: moderngl.Program
        #    #texture_buffer_format_dict: dict[str, BufferFormat]
        #) -> dict[str, int]:
        #    texture_binding_offset_dict: dict[str, int] = {}
        #    binding_offset = 1
        #    texture_uniform_match_pattern = re.compile(r"""
        #        (?P<name>\w+?)
        #        (?P<multi_index>(\[\d+?\])*)
        #    """, flags=re.VERBOSE)
        #    for raw_name in program:
        #        member = program[raw_name]
        #        if not isinstance(member, moderngl.Uniform):
        #            continue
        #        # Used as a `sampler2D`.
        #        assert member.dimension == 1
        #        match_obj = texture_uniform_match_pattern.fullmatch(raw_name)
        #        assert match_obj is not None
        #        name = match_obj.group("name")
        #        #texture_buffer_format = texture_buffer_format_dict[name]
        #        offset = texture_binding_offset_dict.setdefault(name, binding_offset)
        #        #if name not in texture_binding_offset_dict:
        #        #    texture_binding_offset_dict[name] = binding_offset
        #            #binding_offset += texture_buffer_format._size_
        #        #multi_index = tuple(
        #        #    int(index_match.group(1))
        #        #    for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
        #        #)
        #        array_length = member.array_length
        #        binding_offset += array_length
        #        #if not (shape := texture_buffer_format._shape_):
        #        #    assert not multi_index
        #        #    uniform_size = 1
        #        #    local_offset = 0
        #        #else:
        #        #    *dims, uniform_size = shape
        #        #    local_offset = np.ravel_multi_index(multi_index, dims) * uniform_size
        #        #assert member.array_length == uniform_size
        #        #offset = texture_binding_offset_dict[name] + local_offset
        #        member.value = offset if array_length == 1 else list(range(offset, offset + array_length))
        #    return texture_binding_offset_dict

        #def set_uniform_block_bindings(
        #    program: moderngl.Program
        #) -> dict[str, int]:
        #    uniform_block_binding_dict: dict[str, int] = {}
        #    binding = 0
        #    for name in program:
        #        member = program[name]
        #        if not isinstance(member, moderngl.UniformBlock):
        #            continue
        #        # Ensure the name doesn't contain `[]`.
        #        assert re.fullmatch(r"\w+", name) is not None
        #        uniform_block_binding_dict[name] = binding
        #        member.binding = binding
        #        binding += 1
        #    return uniform_block_binding_dict

        shader_str = read_shader_with_includes_replaced(f"{shader_filename}.glsl")
        shaders = {
            shader_type: "\n".join((
                f"#version {Toplevel.context.version_code} core",
                "\n",
                f"#define {shader_type}",
                *custom_macros,
                *(
                    f"#define {array_len_name} {array_len}"
                    for array_len_name, array_len in array_len_items
                ),
                "\n",
                shader_str
            ))
            for shader_type in (
                "VERTEX_SHADER",
                "FRAGMENT_SHADER",
                "GEOMETRY_SHADER",
                "TESS_CONTROL_SHADER",
                "TESS_EVALUATION_SHADER"
            )
            if re.search(rf"\b{shader_type}\b", shader_str, flags=re.MULTILINE) is not None
        }
        program = Toplevel.context.program(
            vertex_shader=shaders["VERTEX_SHADER"],
            fragment_shader=shaders.get("FRAGMENT_SHADER"),
            geometry_shader=shaders.get("GEOMETRY_SHADER"),
            tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
            tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER"),
            varyings=transform_feedback_buffer__np_buffer_pointer_keys
        )

        attributes: dict[str, moderngl.Attribute] = {}
        uniforms: dict[str, dict[tuple[int, ...], moderngl.Uniform]] = {}
        uniform_blocks: dict[str, moderngl.UniformBlock] = {}
        name_pattern = re.compile(r"""
            (?P<name>\w+?)
            (?P<multi_index>(\[\d+?\])*)
        """, flags=re.VERBOSE)
        for raw_name in program:
            match_obj = name_pattern.fullmatch(raw_name)
            assert match_obj is not None
            name = match_obj.group("name")
            multi_index = tuple(
                int(index_match.group(1))
                for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
            )
            member = program[name]
            if isinstance(member, moderngl.Attribute):
                assert not multi_index
                assert name not in attributes
                attributes[name] = member
            elif isinstance(member, moderngl.Uniform):
                uniforms.setdefault(name, {})[multi_index] = member
            elif isinstance(member, moderngl.UniformBlock):
                assert not multi_index
                assert name not in uniform_blocks
                uniform_blocks[name] = member

        attribute_info_dict = {
            name: ProgramAttributeInfo(
                array_length=attribute.array_length,
                dimension=attribute.dimension,
                shape=attribute.shape
            )
            for name, attribute in attributes.items()
        }

        uniform_info_dict: dict[str, ProgramUniformInfo] = {}
        texture_binding = 1
        for name, uniform_dict in uniforms.items():
            unique_array_lengths = list(dict.fromkeys(
                uniform.array_length
                for uniform in uniform_dict.values()
            ))
            array_length = unique_array_lengths.pop()
            assert not unique_array_lengths
            shape = tuple(
                max(indices) + 1
                for indices in zip(*uniform_dict)
            )
            uniform_info_dict[name] = ProgramUniformInfo(
                array_length=array_length,
                shape=shape,
                binding=texture_binding
            )

            for multi_index in it.product(*(range(n) for n in shape)):
                uniform = uniform_dict[multi_index]
                # Used as a `sampler2D`.
                assert uniform.dimension == 1
                uniform.value = texture_binding if array_length == 1 else \
                    list(range(texture_binding, texture_binding + array_length))
                texture_binding += array_length

        uniform_block_info_dict: dict[str, ProgramUniformBlockInfo] = {}
        uniform_block_binding = 0
        for name, uniform_block in uniform_blocks.items():
            uniform_block_info_dict[name] = ProgramUniformBlockInfo(
                size=uniform_block.size,
                binding=uniform_block_binding
            )
            uniform_block.binding = uniform_block_binding
            uniform_block_binding += 1

        #uniform_block_info_dict: dict[str, ProgramUniformBlockInfo] = {
        #    name: ProgramUniformBlockInfo(
        #        size=uniform_block.size,
        #        binding=uniform_block_binding
        #    )
        #    for uniform_block_binding, (name, uniform_block) in enumerate(uniform_blocks.items())
        #}

        #program = construct_moderngl_program(shader_str, custom_macros, array_len_items, varyings)
        #texture_binding_offset_dict = set_texture_bindings(program)
        #uniform_block_binding_dict = set_uniform_block_bindings(program)

        return ProgramInfo(
            program=program,
            attribute_info_dict=attribute_info_dict,
            uniform_info_dict=uniform_info_dict,
            uniform_block_info_dict=uniform_block_info_dict
            #texture_binding_offset_dict=texture_binding_offset_dict,
            #uniform_block_binding_dict=uniform_block_binding_dict
        )

    @_program_info_.finalizer
    @classmethod
    def _program_info_finalizer(
        cls,
        program_info: ProgramInfo
    ) -> None:
        program_info.program.release()

    @Lazy.property_external
    @classmethod
    def _vertex_array_(
        cls,
        program_info: ProgramInfo,
        indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> moderngl.VertexArray | None:
        #def get_item_components(
        #    child: AtomicBufferFormat
        #) -> list[str]:
        #    components = [f"{child._n_col_}{child._base_char_}{child._base_itemsize_}"]
        #    if padding_n_col := child._n_col_pseudo_ - child._n_col_:
        #        components.append(f"{padding_n_col}x{child._base_itemsize_}")
        #    return components * child._n_row_

        attributes_buffer = indexed_attributes_buffer._attributes_buffer_
        index_buffer = indexed_attributes_buffer._index_buffer_
        mode = indexed_attributes_buffer._mode_
        assert isinstance(attributes_buffer_format := attributes_buffer._buffer_format_, StructuredBufferFormat)
        use_index_buffer = not isinstance(index_buffer, OmittedIndexBuffer)

        if attributes_buffer_format._is_empty_ or use_index_buffer and index_buffer._buffer_format_._is_empty_:
            return None

        attribute_items = [
            (child, offset)
            for child, offset in zip(attributes_buffer_format._children_, attributes_buffer_format._offsets_, strict=True)
            if (attribute_info := program_info.attribute_info_dict.get(child._name_)) is not None
            and isinstance(child, AtomicBufferFormat)
            and attribute_info.verify_buffer_format(child)
        ]

        components: list[str] = []
        current_stop = 0
        for child, offset in attribute_items:
            if current_stop != offset:
                components.append(f"{offset - current_stop}x")
            element_components = [f"{child._n_col_}{child._base_char_}{child._base_itemsize_}"]
            if padding_n_col := child._n_col_pseudo_ - child._n_col_:
                element_components.append(f"{padding_n_col}x{child._base_itemsize_}")
            components.extend(element_components * (child._n_row_ * child._size_))
            current_stop = offset + child._nbytes_
        if current_stop != attributes_buffer_format._itemsize_:
            components.append(f"{attributes_buffer_format._itemsize_ - current_stop}x")
        components.append("/v")
        #program = self._info_.program
        #attribute_names: list[str] = []
        #components: list[str] = []
        #current_stop: int = 0
        #for child, offset in zip(attributes_buffer_format._children_, attributes_buffer_format._offsets_, strict=True):
        #    assert isinstance(child, AtomicBufferFormat)
        #    name = child._name_
        #    if (attribute_info := attribute_info_dict.get(name)) is None:
        #        continue
        #    if not attribute_info.verify_buffer_format(child):
        #        continue
        #    #assert isinstance(attribute, moderngl.Attribute)
        #    #assert not child._is_empty_
        #    #assert attribute.array_length == child._size_
        #    #assert attribute.dimension == child._n_col_ * child._n_row_
        #    #assert attribute.shape == child._base_char_.replace("u", "I")
        #    attribute_names.append(name)
        #    if current_stop != offset:
        #        components.append(f"{offset - current_stop}x")
        #    components.extend(get_item_components(child) * child._size_)
        #    current_stop = offset + child._nbytes_
        #if current_stop != attributes_buffer_format._itemsize_:
        #    components.append(f"{attributes_buffer_format._itemsize_ - current_stop}x")
        #components.append("/v")

        return Toplevel.context.vertex_array(
            program=program_info.program,
            attributes_buffer=attributes_buffer._buffer_,
            buffer_format_str=" ".join(components),
            attribute_names=[child._name_ for child, _ in attribute_items],
            index_buffer=index_buffer._buffer_ if use_index_buffer else None,
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

    @Lazy.property_external
    @classmethod
    def _texture_bindings_(
        cls,
        program_info: ProgramInfo,
        texture_buffers: list[TextureBuffer]
    ) -> tuple[tuple[moderngl.Texture, int], ...]:
        return tuple(
            (texture, binding)
            for texture_buffer in texture_buffers
            if (uniform_info := program_info.uniform_info_dict.get(texture_buffer._buffer_format_._name_)) is not None
            and uniform_info.verify_buffer_format(texture_buffer._buffer_format_)
            for binding, texture in enumerate(texture_buffer._texture_array_.flatten(), start=uniform_info.binding)
        )

    @Lazy.property_external
    @classmethod
    def _uniform_block_bindings_(
        cls,
        program_info: ProgramInfo,
        uniform_block_buffers: list[UniformBlockBuffer]
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:
        return tuple(
            (uniform_block_buffer._buffer_, uniform_block_info.binding)
            for uniform_block_buffer in uniform_block_buffers
            if (uniform_block_info := program_info.uniform_block_info_dict.get(uniform_block_buffer._buffer_format_._name_)) is not None
            and uniform_block_info.verify_buffer_format(uniform_block_buffer._buffer_format_)
        )

    def render(
        self,
        framebuffer: Framebuffer
    ) -> None:
        if (vertex_array := self._vertex_array_) is None:
            return

        with Toplevel.context.scope(
            framebuffer=framebuffer._framebuffer_,
            textures=self._texture_bindings_,
            uniform_buffers=self._uniform_block_bindings_
        ):
            Toplevel.context.set_state(framebuffer._context_state_)
            vertex_array.render()

    def transform(self) -> dict[str, np.ndarray]:
        transform_feedback_buffer = self._transform_feedback_buffer_
        with transform_feedback_buffer.temporary_buffer() as buffer:
            if (vertex_array := self._vertex_array_) is not None:
                with Toplevel.context.scope(
                    uniform_buffers=self._uniform_block_bindings_
                ):
                    vertex_array.transform(buffer=buffer)
            data_dict = transform_feedback_buffer.read(buffer)
        return data_dict
