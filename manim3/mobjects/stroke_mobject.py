import numpy as np

from ..constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3f8,
    NP_xi4
)
from ..lazy.lazy import Lazy
from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.index_buffer import IndexBuffer
from ..rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from ..toplevel.toplevel import Toplevel
from ..utils.space import SpaceUtils
from .mobject.mobject_style_meta import MobjectStyleMeta
from .mobject.shape.stroke import Stroke
from .renderable_mobject import RenderableMobject


class StrokeMobject(RenderableMobject):
    __slots__ = ()

    def __init__(
        self,
        stroke: Stroke | None = None
    ) -> None:
        super().__init__()
        if stroke is not None:
            self._stroke_ = stroke

    @MobjectStyleMeta.register(
        partial_method=Stroke.partial,
        interpolate_method=Stroke.interpolate,
        concatenate_method=Stroke.concatenate
    )
    @Lazy.variable
    @classmethod
    def _stroke_(cls) -> Stroke:
        return Stroke()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return (1.0 - 2 ** (-32)) * np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _weight_(cls) -> NP_f8:
        return np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _width_(cls) -> NP_f8:
        return Toplevel.config.stroke_width * np.ones(())

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(
        cls,
        stroke__points: NP_x3f8
    ) -> NP_x3f8:
        return stroke__points

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
        stroke__points: NP_x3f8,
        stroke__disjoints: NP_xi4
    ) -> IndexedAttributesBuffer:
        segment_indices = np.delete(np.arange(len(stroke__points)), stroke__disjoints)[1:]
        index = np.vstack((segment_indices - 1, segment_indices)).T.flatten()

        #def index_getter(
        #    points_len: int
        #) -> NP_xu4:
        #    arange = np.arange(points_len, dtype=np.uint32)
        #    return np.vstack((arange[:-1], arange[1:])).T.flatten()


        #if not multi_line_string__line_strings__points:
        #    index = np.zeros((0,), dtype=np.uint32)
        #    position = np.zeros((0, 3))
        #else:
        #    points_lens = [len(points) for points in multi_line_string__line_strings__points]
        #    offsets = np.cumsum((0, *points_lens[:-1]))
        #    index = np.concatenate([
        #        index_getter(points_len) + offset
        #        for points_len, offset in zip(
        #            points_lens,
        #            offsets,
        #            strict=True
        #        )
        #    ], dtype=np.uint32)
        #    position = np.concatenate(multi_line_string__line_strings__points)
        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=[
                    "vec3 in_position"
                ],
                num_vertex=len(stroke__points),
                data={
                    "in_position": stroke__points
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
