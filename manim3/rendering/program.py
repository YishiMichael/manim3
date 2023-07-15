from dataclasses import dataclass
import re

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
from .buffers.omitted_index_buffer import OmittedIndexBuffer
from .buffers.uniform_block_buffer import UniformBlockBuffer
from .indexed_attributes_buffer import IndexedAttributesBuffer


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProgramInfo:
    program: moderngl.Program
    texture_binding_offset_dict: dict[str, int]
    uniform_block_binding_dict: dict[str, int]


class Program(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...],
        texture_id_buffer_formats: list[BufferFormat],
        varyings: tuple[str, ...]
    ) -> None:
        super().__init__()
        self._shader_filename_ = shader_filename
        self._custom_macros_ = custom_macros
        self._array_len_items_ = array_len_items
        self._texture_id_buffer_formats_.extend(texture_id_buffer_formats)
        self._varyings_ = varyings

    @Lazy.variable_hashable
    @classmethod
    def _shader_filename_(cls) -> str:
        return ""

    @Lazy.variable_hashable
    @classmethod
    def _custom_macros_(cls) -> tuple[str, ...]:
        return ()

    @Lazy.variable_hashable
    @classmethod
    def _array_len_items_(cls) -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.variable_collection
    @classmethod
    def _texture_id_buffer_formats_(cls) -> list[BufferFormat]:
        return []

    @Lazy.variable_hashable
    @classmethod
    def _varyings_(cls) -> tuple[str, ...]:
        return ()

    @Lazy.property_external
    @classmethod
    def _info_(
        cls,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...],
        texture_id_buffer_formats: list[BufferFormat],
        varyings: tuple[str, ...]
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

        def construct_moderngl_program(
            shader_str: str,
            custom_macros: tuple[str, ...],
            array_len_items: tuple[tuple[str, int], ...]
        ) -> moderngl.Program:
            version_string = f"#version {Toplevel.context.version_code} core"
            array_len_macros = [
                f"#define {array_len_name} {array_len}"
                for array_len_name, array_len in array_len_items
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
            return Toplevel.context.program(
                vertex_shader=shaders["VERTEX_SHADER"],
                fragment_shader=shaders.get("FRAGMENT_SHADER"),
                geometry_shader=shaders.get("GEOMETRY_SHADER"),
                tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
                tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER"),
                varyings=varyings
            )

        def set_texture_bindings(
            program: moderngl.Program,
            texture_id_buffer_format_dict: dict[str, BufferFormat]
        ) -> dict[str, int]:
            texture_binding_offset_dict: dict[str, int] = {}
            binding_offset = 1
            texture_uniform_match_pattern = re.compile(r"""
                (?P<name>\w+?)
                (?P<multi_index>(\[\d+?\])*)
            """, flags=re.VERBOSE)
            for raw_name in program:
                member = program[raw_name]
                if not isinstance(member, moderngl.Uniform):
                    continue
                # Used as a `sampler2D`.
                assert member.dimension == 1
                match_obj = texture_uniform_match_pattern.fullmatch(raw_name)
                assert match_obj is not None
                name = match_obj.group("name")
                texture_id_buffer_format = texture_id_buffer_format_dict[name]
                if name not in texture_binding_offset_dict:
                    texture_binding_offset_dict[name] = binding_offset
                    binding_offset += texture_id_buffer_format._size_
                multi_index = tuple(
                    int(index_match.group(1))
                    for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
                )
                if not (shape := texture_id_buffer_format._shape_):
                    assert not multi_index
                    uniform_size = 1
                    local_offset = 0
                else:
                    *dims, uniform_size = shape
                    local_offset = np.ravel_multi_index(multi_index, dims) * uniform_size
                assert member.array_length == uniform_size
                offset = texture_binding_offset_dict[name] + local_offset
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

        shader_str = read_shader_with_includes_replaced(f"{shader_filename}.glsl")
        program = construct_moderngl_program(shader_str, custom_macros, array_len_items)
        texture_binding_offset_dict = set_texture_bindings(program, {
            buffer_format._name_: buffer_format
            for buffer_format in texture_id_buffer_formats
        })
        uniform_block_binding_dict = set_uniform_block_bindings(program)

        return ProgramInfo(
            program=program,
            texture_binding_offset_dict=texture_binding_offset_dict,
            uniform_block_binding_dict=uniform_block_binding_dict
        )

    @_info_.finalizer
    @classmethod
    def _info_finalizer(
        cls,
        info: ProgramInfo
    ) -> None:
        info.program.release()

    def _get_vertex_array(
        self,
        indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> moderngl.VertexArray | None:
        attributes_buffer = indexed_attributes_buffer._attributes_buffer_
        index_buffer = indexed_attributes_buffer._index_buffer_
        mode = indexed_attributes_buffer._mode_
        assert isinstance(attributes_buffer_format := attributes_buffer._buffer_format_, StructuredBufferFormat)
        use_index_buffer = not isinstance(index_buffer, OmittedIndexBuffer)

        if attributes_buffer_format._is_empty_ or use_index_buffer and index_buffer._buffer_format_._is_empty_:
            return None

        def get_item_components(
            child: AtomicBufferFormat
        ) -> list[str]:
            components = [f"{child._n_col_}{child._base_char_}{child._base_itemsize_}"]
            if padding_n_col := child._n_col_pseudo_ - child._n_col_:
                components.append(f"{padding_n_col}x{child._base_itemsize_}")
            return components * child._n_row_

        program = self._info_.program
        attribute_names: list[str] = []
        components: list[str] = []
        current_stop: int = 0
        for child, offset in zip(attributes_buffer_format._children_, attributes_buffer_format._offsets_, strict=True):
            assert isinstance(child, AtomicBufferFormat)
            name = child._name_
            if (attribute := program.get(name, None)) is None:
                continue
            assert isinstance(attribute, moderngl.Attribute)
            assert not child._is_empty_
            assert attribute.array_length == child._size_
            assert attribute.dimension == child._n_col_ * child._n_row_
            assert attribute.shape == child._base_char_.replace("u", "I")
            attribute_names.append(name)
            if current_stop != offset:
                components.append(f"{offset - current_stop}x")
            components.extend(get_item_components(child) * child._size_)
            current_stop = offset + child._nbytes_
        if current_stop != attributes_buffer_format._itemsize_:
            components.append(f"{attributes_buffer_format._itemsize_ - current_stop}x")
        components.append("/v")

        return Toplevel.context.vertex_array(
            program=program,
            attributes_buffer=attributes_buffer.get_buffer(),
            buffer_format_str=" ".join(components),
            attribute_names=attribute_names,
            index_buffer=index_buffer.get_buffer() if use_index_buffer else None,
            mode=mode
        )

    def _get_uniform_block_bindings(
        self,
        uniform_block_buffers: list[UniformBlockBuffer]
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:
        program = self._info_.program
        uniform_block_binding_dict = self._info_.uniform_block_binding_dict
        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
        for uniform_block_buffer in uniform_block_buffers:
            uniform_block_buffer_format = uniform_block_buffer._buffer_format_
            name = uniform_block_buffer_format._name_
            if (uniform_block := program.get(name, None)) is None:
                continue
            assert isinstance(uniform_block, moderngl.UniformBlock)
            assert not uniform_block_buffer_format._is_empty_
            assert uniform_block.size == uniform_block_buffer_format._nbytes_
            uniform_block_bindings.append(
                (uniform_block_buffer.get_buffer(), uniform_block_binding_dict[name])
            )
        return tuple(uniform_block_bindings)

    def _get_texture_bindings(
        self,
        texture_array_dict: dict[str, np.ndarray]
    ) -> tuple[tuple[moderngl.Texture, int], ...]:
        texture_binding_offset_dict = self._info_.texture_binding_offset_dict
        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_id_buffer_format in self._texture_id_buffer_formats_:
            if texture_id_buffer_format._is_empty_:
                continue
            name = texture_id_buffer_format._name_
            if (binding_offset := texture_binding_offset_dict.get(name)) is None:
                continue
            texture_array = texture_array_dict[name]
            assert texture_id_buffer_format._shape_ == texture_array.shape
            texture_bindings.extend(
                (texture, binding)
                for binding, texture in enumerate(texture_array.flat, start=binding_offset)
            )
        return tuple(texture_bindings)
