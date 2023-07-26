import numpy as np

from ...constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3f8,
    NP_x2i4
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.attributes_buffer import AttributesBuffer
from ...rendering.buffers.index_buffer import IndexBuffer
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ...rendering.mgl_enums import PrimitiveMode
from ...rendering.vertex_array import VertexArray
from ...toplevel.toplevel import Toplevel
from ..mobject.operation_handlers.lerp_interpolate_handler import LerpInterpolateHandler
from ..mobject.style_meta import StyleMeta
from ..renderable_mobject import RenderableMobject
from .graphs.graph import Graph
from .graphs.graph_concatenate_handler import GraphConcatenateHandler
from .graphs.graph_interpolate_handler import GraphInterpolateHandler
from .graphs.graph_partial_handler import GraphPartialHandler


class GraphMobject(RenderableMobject):
    __slots__ = ()

    def __init__(
        self,
        graph: Graph | None = None
    ) -> None:
        super().__init__()
        if graph is not None:
            self._graph_ = graph

    @StyleMeta.register(
        partial_operation=GraphPartialHandler,
        interpolate_operation=GraphInterpolateHandler,
        concatenate_operation=GraphConcatenateHandler
    )
    @Lazy.variable
    @classmethod
    def _graph_(cls) -> Graph:
        return Graph()

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return np.ones(())

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _weight_(cls) -> NP_f8:
        return np.ones(())

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _width_(cls) -> NP_f8:
        return Toplevel.config.graph_width * np.ones(())

    @Lazy.property_array
    @classmethod
    def _local_sample_positions_(
        cls,
        graph__positions: NP_x3f8,
        graph__indices: NP_x2i4
    ) -> NP_x3f8:
        return graph__positions[graph__indices.flatten()]

    @Lazy.property
    @classmethod
    def _graph_uniform_block_buffer_(
        cls,
        color: NP_3f8,
        opacity: NP_f8,
        weight: NP_f8,
        width: NP_f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_graph",
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
    def _graph_indexed_attributes_buffer_(
        cls,
        graph__positions: NP_x3f8,
        graph__indices: NP_x2i4
    ) -> IndexedAttributesBuffer:
        #segment_indices = np.delete(np.arange(len(stroke__points)), stroke__disjoints)[1:]
        #index = np.vstack((segment_indices - 1, segment_indices)).T.flatten()

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
                num_vertex=len(graph__positions),
                data={
                    "in_position": graph__positions
                }
            ),
            index_buffer=IndexBuffer(
                data=graph__indices.flatten()
            ),
            mode=PrimitiveMode.LINES
        )

    @Lazy.property
    @classmethod
    def _graph_vertex_array_(
        cls,
        camera__camera_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        graph_uniform_block_buffer: UniformBlockBuffer,
        graph_indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="graph",
            uniform_block_buffers=[
                camera__camera_uniform_block_buffer,
                model_uniform_block_buffer,
                graph_uniform_block_buffer
            ],
            indexed_attributes_buffer=graph_indexed_attributes_buffer
        )

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        self._graph_vertex_array_.render(
            framebuffer=target_framebuffer
        )
