from __future__ import annotations


from typing import Self

from ...animatables.animatable.animatable import AnimatableActions
from ...animatables.arrays.animatable_color import AnimatableColor
from ...animatables.arrays.animatable_float import AnimatableFloat
from ...animatables.graph import Graph
from ...animatables.model import ModelActions
from ...constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x2i4,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.attributes_buffer import AttributesBuffer
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...rendering.mgl_enums import PrimitiveMode
from ...rendering.vertex_array import VertexArray
from ...toplevel.toplevel import Toplevel
from ...utils.path_utils import PathUtils
from ..mobject import Mobject


class GraphMobject(Mobject):
    __slots__ = ()

    def __init__(
        self: Self,
        graph: Graph | None = None
    ) -> None:
        super().__init__()
        if graph is not None:
            self._graph_ = graph

    @AnimatableActions.interpolate.register_descriptor()
    @AnimatableActions.piecewise.register_descriptor()
    @Lazy.volatile()
    @staticmethod
    def _graph_() -> Graph:
        return Graph()

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableColor)
    @Lazy.volatile()
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor(Toplevel.config.default_color)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _opacity_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.default_opacity)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _weight_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.default_weight)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _width_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.graph_width)

    @Lazy.property()
    @staticmethod
    def _local_sample_positions_(
        graph__positions: NP_x3f8,
        graph__edges: NP_x2i4
    ) -> NP_x3f8:
        return graph__positions[graph__edges.flatten()]

    @Lazy.property()
    @staticmethod
    def _graph_uniform_block_buffer_(
        color__array: NP_3f8,
        opacity__array: NP_f8,
        weight__array: NP_f8,
        width__array: NP_f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_graph",
            field_declarations=(
                "vec3 u_color",
                "float u_opacity",
                "float u_weight",
                "float u_width"
            ),
            data_dict={
                "u_color": color__array,
                "u_opacity": opacity__array,
                "u_weight": weight__array,
                "u_width": width__array
            }
        )

    @Lazy.property()
    @staticmethod
    def _graph_attributes_buffer_(
        graph__positions: NP_x3f8,
        graph__edges: NP_x2i4
    ) -> AttributesBuffer:
        return AttributesBuffer(
            field_declarations=(
                "vec3 in_position",
            ),
            data_dict={
                "in_position": graph__positions
            },
            index=graph__edges.flatten(),
            primitive_mode=PrimitiveMode.LINES,
            num_vertices=len(graph__positions)
        )

    @Lazy.property()
    @staticmethod
    def _graph_vertex_array_(
        camera__camera_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        graph_uniform_block_buffer: UniformBlockBuffer,
        graph_attributes_buffer: AttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_path=PathUtils.shaders_dir.joinpath("graph.glsl"),
            uniform_block_buffers=(
                camera__camera_uniform_block_buffer,
                model_uniform_block_buffer,
                graph_uniform_block_buffer
            ),
            attributes_buffer=graph_attributes_buffer
        )

    def _render(
        self: Self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        self._graph_vertex_array_.render(target_framebuffer)
