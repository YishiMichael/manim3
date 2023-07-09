import itertools as it
import re

import moderngl
import numpy as np

from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from .buffer_formats.buffer_format import BufferFormat
from .buffers.attributes_buffer import AttributesBuffer
from .buffers.texture_id_buffer import TextureIdBuffer
from .buffers.transform_feedback_buffer import TransformFeedbackBuffer
from .buffers.uniform_block_buffer import UniformBlockBuffer
from .context import Context
from .framebuffers.framebuffer import Framebuffer
from .indexed_attributes_buffer import IndexedAttributesBuffer
from .mgl_enums import PrimitiveMode
from .program import Program


class VertexArray(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        shader_filename: str,
        custom_macros: list[str] | None = None,
        texture_id_buffers: list[TextureIdBuffer] | None = None,
        uniform_block_buffers: list[UniformBlockBuffer] | None = None,
        indexed_attributes_buffer: IndexedAttributesBuffer | None = None,
        transform_feedback_buffer: TransformFeedbackBuffer | None = None
    ) -> None:
        super().__init__()
        self._shader_filename_ = shader_filename
        if custom_macros is not None:
            self._custom_macros_ = tuple(custom_macros)
        if texture_id_buffers is not None:
            self._texture_id_buffers_.reset(texture_id_buffers)
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
    def _texture_id_buffers_(cls) -> list[TextureIdBuffer]:
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
        texture_id_buffers__array_len_items: list[tuple[tuple[str, int], ...]],
        uniform_block_buffers__array_len_items: list[tuple[tuple[str, int], ...]],
        indexed_attributes_buffer__attributes_buffer__array_len_items: tuple[tuple[str, int], ...],
        transform_feedback_buffer__array_len_items: tuple[tuple[str, int], ...]
    ) -> tuple[tuple[str, int], ...]:
        return tuple(
            (array_len_name, array_len)
            for array_len_name, array_len in it.chain(
                it.chain.from_iterable(texture_id_buffers__array_len_items),
                it.chain.from_iterable(uniform_block_buffers__array_len_items),
                indexed_attributes_buffer__attributes_buffer__array_len_items,
                transform_feedback_buffer__array_len_items
            )
            if not re.fullmatch(r"__\w+__", array_len_name)
        )

    @Lazy.property
    @classmethod
    def _program_(
        cls,
        shader_filename: str,
        custom_macros: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...],
        texture_id_buffers__buffer_format: list[BufferFormat],
        transform_feedback_buffer__np_buffer_pointer_keys: tuple[str, ...]
    ) -> Program:
        return Program(
            shader_filename=shader_filename,
            custom_macros=custom_macros,
            array_len_items=array_len_items,
            texture_id_buffer_formats=texture_id_buffers__buffer_format,
            varyings=transform_feedback_buffer__np_buffer_pointer_keys
        )

    @Lazy.property_external
    @classmethod
    def _vertex_array_(
        cls,
        program: Program,
        indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> moderngl.VertexArray | None:
        return program._get_vertex_array(indexed_attributes_buffer)

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
    def _uniform_block_bindings_(
        cls,
        program: Program,
        uniform_block_buffers: list[UniformBlockBuffer]
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:
        return program._get_uniform_block_bindings(uniform_block_buffers)

    def render(
        self,
        *,
        framebuffer: Framebuffer,
        # Note, redundant textures are currently not supported.
        texture_array_dict: dict[str, np.ndarray] | None = None
    ) -> None:
        if (vertex_array := self._vertex_array_) is None:
            return

        if texture_array_dict is None:
            texture_array_dict = {}
        with Context.scope(
            framebuffer=framebuffer.framebuffer,
            textures=self._program_._get_texture_bindings(texture_array_dict),
            uniform_buffers=self._uniform_block_bindings_
        ):
            Context.set_state(framebuffer.context_state)
            vertex_array.render()

    def transform(self) -> dict[str, np.ndarray]:
        transform_feedback_buffer = self._transform_feedback_buffer_
        with transform_feedback_buffer.temporary_buffer() as buffer:
            if (vertex_array := self._vertex_array_) is not None:
                with Context.scope(
                    uniform_buffers=self._uniform_block_bindings_
                ):
                    vertex_array.transform(buffer=buffer)
            data_dict = transform_feedback_buffer.read(buffer)
        return data_dict
