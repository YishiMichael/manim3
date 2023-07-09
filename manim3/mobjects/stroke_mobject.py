import numpy as np

from ..config import Config
from ..custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3f8,
    NP_xu4
)
from ..lazy.lazy import Lazy
from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.index_buffer import IndexBuffer
from ..rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from ..shape.shape import MultiLineString
from ..utils.space import SpaceUtils
from .mobject_style_meta import MobjectStyleMeta
from .renderable_mobject import RenderableMobject


class StrokeMobject(RenderableMobject):
    __slots__ = ()

    def __init__(
        self,
        multi_line_string: MultiLineString | None = None
    ) -> None:
        super().__init__()
        if multi_line_string is not None:
            self._multi_line_string_ = multi_line_string

    @MobjectStyleMeta.register(
        partial_method=MultiLineString.partial,
        interpolate_method=MultiLineString.interpolate,
        concatenate_method=MultiLineString.concatenate
    )
    @Lazy.variable
    @classmethod
    def _multi_line_string_(cls) -> MultiLineString:
        return MultiLineString()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_3f8
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return (1.0 - 2 ** (-32)) * np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _weight_(cls) -> NP_f8:
        return np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _width_(cls) -> NP_f8:
        return Config().style.stroke_width * np.ones(())

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(
        cls,
        multi_line_string__line_strings__points: list[NP_x3f8]
    ) -> NP_x3f8:
        if not multi_line_string__line_strings__points:
            return np.zeros((0, 3))
        return np.concatenate(multi_line_string__line_strings__points)

    @Lazy.property
    @classmethod
    def _stroke_uniform_block_buffer_(
        cls,
        color: NP_3f8,
        opacity: NP_f8,
        weight: NP_f8,
        width: NP_f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_stroke",
            fields=[
                "vec3 u_color",
                "float u_opacity",
                "float u_weight",
                "float u_width"
            ],
            data={
                "u_color": color,
                "u_opacity": opacity,
                "u_weight": weight,
                "u_width": width
            }
        )

    @Lazy.property
    @classmethod
    def _stroke_indexed_attributes_buffer_(
        cls,
        multi_line_string__line_strings__points: list[NP_x3f8],
        multi_line_string__line_strings__is_ring: list[bool]
    ) -> IndexedAttributesBuffer:

        def index_getter(
            points_len: int,
            is_ring: bool
        ) -> NP_xu4:
            arange = np.arange(points_len, dtype=np.uint32)
            index_pairs = np.vstack((arange, np.roll(arange, -1))).T
            if not is_ring:
                index_pairs = index_pairs[:-1]
            return index_pairs.flatten()

        if not multi_line_string__line_strings__points:
            index = np.zeros((0,), dtype=np.uint32)
            position = np.zeros((0, 3))
        else:
            points_lens = [len(points) for points in multi_line_string__line_strings__points]
            offsets = np.cumsum((0, *points_lens[:-1]))
            index = np.concatenate([
                index_getter(points_len, is_ring) + offset
                for points_len, is_ring, offset in zip(
                    points_lens,
                    multi_line_string__line_strings__is_ring,
                    offsets,
                    strict=True
                )
            ], dtype=np.uint32)
            position = np.concatenate(multi_line_string__line_strings__points)

        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=[
                    "vec3 in_position"
                ],
                num_vertex=len(position),
                data={
                    "in_position": position
                }
            ),
            index_buffer=IndexBuffer(
                data=index
            ),
            mode=PrimitiveMode.LINES
        )

    @Lazy.property
    @classmethod
    def _stroke_vertex_array_(
        cls,
        camera__camera_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        stroke_uniform_block_buffer: UniformBlockBuffer,
        stroke_indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="stroke",
            uniform_block_buffers=[
                camera__camera_uniform_block_buffer,
                model_uniform_block_buffer,
                stroke_uniform_block_buffer
            ],
            indexed_attributes_buffer=stroke_indexed_attributes_buffer
        )

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        self._stroke_vertex_array_.render(
            framebuffer=target_framebuffer
        )
