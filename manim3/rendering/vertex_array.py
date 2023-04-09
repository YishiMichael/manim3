__all__ = ["VertexArray"]


from dataclasses import dataclass
#from functools import reduce
import itertools as it
#import operator as op
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
    AtomicBufferFormat,
    AttributesBuffer,
    BufferFormat,
    IndexBuffer,
    StructuredBufferFormat,
    TextureIDBuffer,
    TransformFeedbackBuffer,
    UniformBlockBuffer
)


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
        self._texture_id_buffer_formats_.add(*texture_id_buffer_formats)
        self._varyings_ = varyings

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _shader_filename_(cls) -> str:
        return ""

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _custom_macros_(cls) -> tuple[str, ...]:
        return ()

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _array_len_items_(cls) -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _texture_id_buffer_formats_(cls) -> list[BufferFormat]:
        return []

    @Lazy.variable(LazyMode.SHARED)
    @classmethod
    def _varyings_(cls) -> tuple[str, ...]:
        return ()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _info_(
        cls,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...],
        _texture_id_buffer_formats_: list[BufferFormat],
        varyings: tuple[str, ...]
    ) -> ProgramInfo:
        
        def construct_moderngl_program(
            shader_str: str,
            custom_macros: tuple[str, ...],
            array_len_items: tuple[tuple[str, int], ...]
        ) -> moderngl.Program:
            version_string = f"#version {Context.mgl_context.version_code} core"
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
            return Context.program(
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
                    binding_offset += texture_id_buffer_format._size_.value
                multi_index = tuple(
                    int(index_match.group(1))
                    for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
                )
                if not texture_id_buffer_format._shape_.value:
                    assert not multi_index
                    uniform_size = 1
                    local_offset = 0
                else:
                    *dims, uniform_size = texture_id_buffer_format._shape_.value
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

        with ConfigSingleton().path.shaders_dir.joinpath(f"{shader_filename}.glsl").open() as shader_file:
            shader_str = shader_file.read()
        program = construct_moderngl_program(shader_str, custom_macros, array_len_items)
        texture_binding_offset_dict = set_texture_bindings(program, {
            buffer_format._name_.value: buffer_format
            for buffer_format in _texture_id_buffer_formats_
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

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _program_(
    #    cls,
    #    info: ProgramInfo
    #) -> moderngl.Program:
    #    return info.program

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _texture_binding_offset_dict_(
    #    cls,
    #    info: ProgramInfo
    #) -> dict[str, int]:
    #    return info.texture_binding_offset_dict
        #return {
        #    name: (texture_id_buffer_format._shape_.value, binding_offset)
        #    for texture_id_buffer_format in _texture_id_buffer_formats_
        #    if (binding_offset := info.texture_binding_offset_dict.get(
        #        name := texture_id_buffer_format._name_.value
        #    )) is not None
        #}
        ##texture_id_buffer_format_dict = dict(texture_id_buffer_format_items)
        #texture_binding_info_dict: dict[str, tuple[tuple[int, ...], int]] = {}
        #for texture_id_buffer_format in _texture_id_buffer_formats_:
        #    name = texture_id_buffer_format._name_.value
        #    if (binding_offset := info.texture_binding_offset_dict.get(name)) is None:
        #        continue
        #    texture_binding_info_dict[name] = (texture_id_buffer_format._shape_.value, binding_offset)
        ##for texture_id_buffer_name, binding_offset in info.texture_binding_offset_dict.items():
        ##    texture_id_buffer_format = texture_id_buffer_format_dict[texture_id_buffer_name]
        ##    assert not texture_id_buffer_format._is_empty_.value
        ##    texture_binding_info_dict[texture_id_buffer_name] = (texture_id_buffer_format._shape_.value, binding_offset)
        #return texture_binding_info_dict

    #@Lazy.property(LazyMode.UNWRAPPED)
    #@classmethod
    #def _uniform_block_binding_dict_(
    #    cls,
    #    info: ProgramInfo
    #) -> dict[str, int]:
    #    return info.uniform_block_binding_dict



        #def validate_attributes_buffer(
        #    attributes_buffer_format: StructuredBufferFormat,
        #    program_attributes: dict[str, moderngl.Attribute]
        #) -> None:
        #    for child in attributes_buffer_format._children_:
        #        assert isinstance(child, AtomicBufferFormat)
        #        attribute = program_attributes[child._name_.value]
        #        assert attribute.array_length == child._size_.value
        #        assert attribute.dimension == child._n_col_.value * child._n_row_.value
        #        assert attribute.shape == child._base_char_.value.replace("u", "I")
            #vertex_dtype = attributes_buffer._vertex_dtype_.value
            #for attribute_name, attribute in program_attributes.items():
            #    field_dtype = vertex_dtype[attribute_name]
            #    assert attribute.array_length == cls._int_prod(field_dtype.shape)
            #    assert attribute.dimension == cls._int_prod(field_dtype.base.shape) * cls._int_prod(field_dtype.base["_"].shape)
            #    assert attribute.shape == field_dtype.base["_"].base.kind.replace("u", "I")

        #def get_buffer_format_str(
        #    vertex_dtype: np.dtype,
        #    attribute_name_tuple: tuple[str, ...]
        #) -> tuple[str, list[str]]:
        #    # TODO: This may require refactory.
        #    #vertex_dtype = self._vertex_dtype_.value
        #    vertex_fields = vertex_dtype.fields
        #    assert vertex_fields is not None
        #    dtype_stack: list[tuple[np.dtype, int]] = []
        #    attribute_names: list[str] = []
        #    for field_name, (field_dtype, field_offset, *_) in vertex_fields.items():
        #        if field_name not in attribute_name_tuple:
        #            continue
        #        dtype_stack.append((field_dtype, field_offset))
        #        attribute_names.append(field_name)

        #    components: list[str] = []
        #    current_offset = 0
        #    while dtype_stack:
        #        dtype, offset = dtype_stack.pop(0)
        #        dtype_size = cls._int_prod(dtype.shape)
        #        dtype_itemsize = dtype.base.itemsize
        #        if dtype.base.fields is not None:
        #            dtype_stack = [
        #                (child_dtype, offset + i * dtype_itemsize + child_offset)
        #                for i in range(dtype_size)
        #                for child_dtype, child_offset, *_ in dtype.base.fields.values()
        #            ] + dtype_stack
        #            continue
        #        if current_offset != offset:
        #            components.append(f"{offset - current_offset}x")
        #            current_offset = offset
        #        components.append(f"{dtype_size}{dtype.base.kind}{dtype_itemsize}")
        #        current_offset += dtype_size * dtype_itemsize
        #    if current_offset != vertex_dtype.itemsize:
        #        components.append(f"{vertex_dtype.itemsize - current_offset}x")
        #    components.append("/v")
        #    return " ".join(components), attribute_names

    def _get_vertex_array(
        self,
        indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> moderngl.VertexArray | None:
        attributes_buffer = indexed_attributes_buffer._attributes_buffer_
        assert isinstance(attributes_buffer_format := attributes_buffer._buffer_format_, StructuredBufferFormat)
        index_buffer = indexed_attributes_buffer._index_buffer_
        mode = indexed_attributes_buffer._mode_.value

        if attributes_buffer_format._is_empty_.value or index_buffer._buffer_format_._is_empty_.value:
            return None

        def get_item_components(
            child: AtomicBufferFormat
        ) -> list[str]:
            components = [f"{child._n_col_.value}{child._base_char_.value}{child._base_itemsize_.value}"]
            if padding_factor := child._row_itemsize_factor_.value - child._n_col_.value:
                components.append(f"{padding_factor}x{child._base_itemsize_.value}")
            return components * child._n_row_.value

        program = self._info_.value.program
        #program_attributes = {
        #    name: member
        #    for name in program
        #    if isinstance(member := program[name], moderngl.Attribute)
        #}
        attribute_names: list[str] = []
        components: list[str] = []
        current_stop: int = 0
        for child, offset in zip(attributes_buffer_format._children_, attributes_buffer_format._offsets_.value, strict=True):
            assert isinstance(child, AtomicBufferFormat)
            name = child._name_.value
            if (attribute := program.get(name, None)) is None:
                continue
            assert isinstance(attribute, moderngl.Attribute)
            assert not child._is_empty_.value
            assert attribute.array_length == child._size_.value
            assert attribute.dimension == child._n_col_.value * child._n_row_.value
            assert attribute.shape == child._base_char_.value.replace("u", "I")
            attribute_names.append(name)
            if current_stop != offset:
                components.append(f"{offset - current_stop}x")
            components.extend(get_item_components(child) * child._size_.value)
            current_stop = offset + child._nbytes_.value
        if current_stop != attributes_buffer_format._itemsize_.value:
            components.append(f"{attributes_buffer_format._itemsize_.value - current_stop}x")
        components.append("/v")
        #validate_attributes_buffer(attributes_buffer_format, program_attributes)
        #buffer_format_str, attribute_names = get_buffer_format_str(attributes_buffer._vertex_dtype_.value, tuple(program_attributes))

        #print(" ".join(components))
        #print(attributes_buffer_format._itemsize_.value)
        #print(attributes_buffer.read())
        return Context.vertex_array(
            program=program,
            attributes_buffer=attributes_buffer.get_buffer(),
            buffer_format_str=" ".join(components),
            attribute_names=attribute_names,
            index_buffer=index_buffer.get_buffer(),
            mode=mode
        )

    def _get_uniform_block_bindings(
        self,
        uniform_block_buffers: list[UniformBlockBuffer]
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:
        #def validate_uniform_block_buffer(
        #    uniform_block_buffer: UniformBlockBuffer,
        #    program_uniform_block: moderngl.UniformBlock
        #) -> None:
        #    assert program_uniform_block.name == uniform_block_buffer._name_.value
        #    assert program_uniform_block.size == uniform_block_buffer._nbytes_.value

        #uniform_block_dict = {
        #    uniform_block_buffer._name_.value: uniform_block_buffer
        #    for uniform_block_buffer in _uniform_block_buffers_
        #}
        program = self._info_.value.program
        uniform_block_binding_dict = self._info_.value.uniform_block_binding_dict
        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
        for uniform_block_buffer in uniform_block_buffers:
            uniform_block_buffer_format = uniform_block_buffer._buffer_format_
            #if uniform_block_buffer_format._is_empty_.value:
            #    continue
            name = uniform_block_buffer_format._name_.value
            if (uniform_block := program.get(name, None)) is None:
                continue
            assert isinstance(uniform_block, moderngl.UniformBlock)
            assert not uniform_block_buffer_format._is_empty_.value
            #program_uniform_block = program[name]
            #assert isinstance(program_uniform_block, moderngl.UniformBlock)
            assert uniform_block.size == uniform_block_buffer_format._nbytes_.value
            uniform_block_bindings.append(
                (uniform_block_buffer.get_buffer(), uniform_block_binding_dict[name])
            )
        #for uniform_block_name, binding in info.uniform_block_binding_dict.items():
        #    uniform_block_buffer = uniform_block_dict[uniform_block_name]
        #    assert not uniform_block_buffer._is_empty_.value
        #    program_uniform_block = info.program[uniform_block_name]
        #    assert isinstance(program_uniform_block, moderngl.UniformBlock)
        #    validate_uniform_block_buffer(uniform_block_buffer, program_uniform_block)
        #    uniform_block_bindings.append((uniform_block_buffer.get_buffer(), binding))
        return tuple(uniform_block_bindings)

    def _get_texture_bindings(
        self,
        texture_array_dict: dict[str, np.ndarray]
    ) -> tuple[tuple[moderngl.Texture, int], ...]:
        texture_binding_offset_dict = self._info_.value.texture_binding_offset_dict
        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_id_buffer_format in self._texture_id_buffer_formats_:
            if texture_id_buffer_format._is_empty_.value:
                continue
            name = texture_id_buffer_format._name_.value
            texture_array = texture_array_dict[name]
            assert texture_id_buffer_format._shape_.value == texture_array.shape
            texture_bindings.extend(
                (texture, binding)
                for binding, texture in enumerate(texture_array.flat, start=texture_binding_offset_dict[name])
            )
        return tuple(texture_bindings)


class VertexArray(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        shader_filename: str,
        custom_macros: list[str] | None = None,
        texture_id_buffers: list[TextureIDBuffer] | None = None,
        uniform_block_buffers: list[UniformBlockBuffer] | None = None,
        indexed_attributes_buffer: IndexedAttributesBuffer | None = None,
        transform_feedback_buffer: TransformFeedbackBuffer | None = None
    ) -> None:
        super().__init__()
        self._shader_filename_ = shader_filename
        if custom_macros is not None:
            self._custom_macros_ = tuple(custom_macros)
        if texture_id_buffers is not None:
            self._texture_id_buffers_.add(*texture_id_buffers)
        if uniform_block_buffers is not None:
            self._uniform_block_buffers_.add(*uniform_block_buffers)
        if indexed_attributes_buffer is not None:
            self._indexed_attributes_buffer_ = indexed_attributes_buffer
        if transform_feedback_buffer is not None:
            self._transform_feedback_buffer_ = transform_feedback_buffer

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
    def _texture_id_buffers_(cls) -> list[TextureIDBuffer]:
        return []

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _uniform_block_buffers_(cls) -> list[UniformBlockBuffer]:
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

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _transform_feedback_buffer_(cls) -> TransformFeedbackBuffer:
        return TransformFeedbackBuffer(
            fields=[],
            num_vertex=0
        )

    #@Lazy.variable(LazyMode.SHARED)
    #@classmethod
    #def _varyings_(cls) -> tuple[str, ...]:
    #    return ()

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _array_len_items_(
        cls,
        texture_id_buffers__array_len_items: list[tuple[tuple[str, int], ...]],
        uniform_block_buffers__array_len_items: list[tuple[tuple[str, int], ...]],
        indexed_attributes_buffer__attributes_buffer__array_len_items: tuple[tuple[str, int], ...],
        transform_feedback_buffer__array_len_items: tuple[tuple[str, int], ...]
    ) -> tuple[tuple[str, int], ...]:
        #array_len_items: list[tuple[str, int]] = [
        #    *it.chain(*texture_id_buffers__array_len_items),
        #    *it.chain(*uniform_block_buffers__array_len_items),
        #    *indexed_attributes_buffer__attributes_buffer__array_len_items
        #]
        #for texture_id_buffer_array_len_items in texture_id_buffers__array_len_items:
        #    array_len_items.update(dict(texture_id_buffer_array_len_items))
        #for uniform_block_buffer_array_len_items in uniform_block_buffers__array_len_items:
        #    array_len_items.update(uniform_block_buffer_array_len_items)
        #array_len_items.update(dict(indexed_attributes_buffer__attributes_buffer__array_len_items))
        return tuple(
            (array_len_name, array_len)
            for array_len_name, array_len in (
                *it.chain(*texture_id_buffers__array_len_items),
                *it.chain(*uniform_block_buffers__array_len_items),
                *indexed_attributes_buffer__attributes_buffer__array_len_items,
                *transform_feedback_buffer__array_len_items
            )
            if not re.fullmatch(r"__\w+__", array_len_name)
        )

    #@Lazy.property(LazyMode.SHARED)
    #@classmethod
    #def _texture_id_buffer_format_items_(
    #    cls,
    #    _texture_id_buffers__buffer_format_: list[BufferFormat]
    #) -> tuple[tuple[str, BufferFormat], ...]:
    #    return tuple(
    #        (texture_id_buffer_format._name_.value, texture_id_buffer_format)
    #        for texture_id_buffer_format in _texture_id_buffers__buffer_format_
    #    )
    #    #return tuple(
    #    #    (texture_id_buffer._buffer_format_._name_.value, texture_id_buffer._buffer_format_._shape_.value)
    #    #    for texture_id_buffer in _texture_id_buffers_
    #    #)

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _program_(
        cls,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...],
        _texture_id_buffers__buffer_format_: list[BufferFormat],
        transform_feedback_buffer__np_buffer_pointer_keys: tuple[str, ...]
        #texture_id_buffer_format_items: tuple[tuple[str, BufferFormat], ...]
    ) -> Program:
        return Program(
            shader_filename=shader_filename,
            custom_macros=custom_macros,
            array_len_items=array_len_items,
            texture_id_buffer_formats=_texture_id_buffers__buffer_format_,
            varyings=transform_feedback_buffer__np_buffer_pointer_keys
        )

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _vertex_array_(
        cls,
        _program_: Program,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> moderngl.VertexArray | None:
        return _program_._get_vertex_array(_indexed_attributes_buffer_)

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
    def _uniform_block_bindings_(
        cls,
        _program_: Program,
        _uniform_block_buffers_: list[UniformBlockBuffer]
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:
        return _program_._get_uniform_block_bindings(_uniform_block_buffers_)

    #@classmethod
    #def _int_prod(
    #    cls,
    #    shape: tuple[int, ...]
    #) -> int:
    #    return reduce(op.mul, shape, 1)

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
        Context.set_state(context_state)
        with Context.mgl_context.scope(
            framebuffer=framebuffer,
            enable_only=context_state.enable_only,
            textures=self._program_._get_texture_bindings(texture_array_dict),
            uniform_buffers=self._uniform_block_bindings_.value
        ):
            vertex_array.render()

    def transform(self) -> dict[str, np.ndarray]:
        transform_feedback_buffer = self._transform_feedback_buffer_
        #transform_feedback_buffer_format = transform_feedback_buffer._buffer_format_
        with transform_feedback_buffer.temporary_buffer() as buffer:
            #print(buffer.read())
            if (vertex_array := self._vertex_array_.value) is not None:
                #print(buffer.read())
                vertex_array.transform(buffer=buffer)
                #print(buffer.read())
            data_dict = transform_feedback_buffer.read(buffer)
            #print(buffer.read())
            #print(data_dict)
        #with TemporaryBuffer(reserve=transform_feedback_buffer_format._nbytes_.value) as temporary_buffer:
        #    buffer = temporary_buffer.buffer
        #    varyings = tuple(transform_feedback_buffer_format.read(buffer))  # TODO
        #    self._varyings_ = varyings
        #    if (vertex_array := self._vertex_array_.value) is not None:
        #        vertex_array.transform(
        #            buffer=buffer,
        #            mode=mode
        #        )
        #    data_dict = transform_feedback_buffer_format.read(buffer)
        return data_dict
