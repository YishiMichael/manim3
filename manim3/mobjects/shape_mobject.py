__all__ = [
    "ShapeMobject",
    "ShapeStrokeMobject"
]


from typing import Callable

import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Real,
    Vec4T
)
from ..geometries.shape_geometry import ShapeGeometry
from ..geometries.shape_stroke_geometry import ShapeStrokeGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..render_passes.copy_pass import CopyPass
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    AttributesBuffer,
    ContextState,
    Framebuffer,
    IntermediateDepthTextures,
    IntermediateFramebuffer,
    IntermediateTextures,
    RenderStep,
    Renderable,
    UniformBlockBuffer
)
from ..utils.scene_config import SceneConfig
from ..utils.shape import Shape


class ShapeMobject(MeshMobject):
    def __init__(self, shape: Shape):
        super().__init__()
        self._shape_ = shape

    @lazy_property_initializer_writable
    @staticmethod
    def _shape_() -> Shape:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _geometry_(shape: Shape) -> ShapeGeometry:
        return ShapeGeometry(shape)

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_only_() -> int:
        return moderngl.BLEND

    def set_local_fill(self, color: ColorType | Callable[..., Vec4T]):
        self._color_ = color
        return self

    def set_fill(
        self,
        color: ColorType | Callable[..., Vec4T],
        *,
        broadcast: bool = True
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject.set_local_fill(color=color)
        return self


class ShapeStrokeMobject(ShapeMobject):
    def __init__(
        self,
        shape: Shape,
        width: Real,
        round_end: bool = True,
        single_sided: bool = False
    ):
        super().__init__(shape)
        self._width_ = width
        self._round_end_ = round_end
        self._single_sided_ = single_sided

    @lazy_property_initializer_writable
    @staticmethod
    def _distance_dilate_() -> Real:
        return 0.0

    @lazy_property_initializer
    @staticmethod
    def _ub_stroke_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_stroke", [
            "float u_distance_dilate"
        ])

    @lazy_property
    @staticmethod
    def _ub_stroke_(
        ub_stroke_o: UniformBlockBuffer,
        distance_dilate: Real
    ) -> UniformBlockBuffer:
        ub_stroke_o.write({
            "u_distance_dilate": np.array(distance_dilate)
        })
        return ub_stroke_o

    @lazy_property_initializer_writable
    @staticmethod
    def _width_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _round_end_() -> bool:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _single_sided_() -> bool:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _geometry_(shape: Shape, width: Real, round_end: bool, single_sided: bool) -> ShapeStrokeGeometry:
        return ShapeStrokeGeometry(shape, width, round_end, single_sided)

    @lazy_property_initializer
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 a_position",
            "vec3 a_normal",
            "vec2 a_uv",
            "vec4 a_color",
            "float a_distance"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        geometry: ShapeStrokeGeometry,
        color: ColorType | Callable[..., Vec4T]
    ) -> AttributesBuffer:
        position = geometry._position_
        normal = geometry._normal_
        uv = geometry._uv_
        distance = geometry._distance_
        color_array = MeshMobject._calculate_color_array(color, position, normal, uv)
        attributes_o.write({
            "a_position": position,
            "a_normal": normal,
            "a_uv": uv,
            "a_color": color_array,
            "a_distance": distance
        })
        return attributes_o

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        with IntermediateTextures.register_n(1) as textures:
            with IntermediateDepthTextures.register_n(1) as depth_textures:
                intermediate_framebuffer = IntermediateFramebuffer(textures, depth_textures[0])
                self._render_by_step(RenderStep(
                    shader_str=Renderable._read_shader("shape_stroke"),
                    texture_storages=[
                        self._u_color_maps_
                    ],
                    uniform_blocks=[
                        scene_config._camera_._ub_camera_matrices_,
                        self._ub_model_matrices_,
                        scene_config._ub_lights_,
                        self._ub_stroke_
                    ],
                    attributes=self._attributes_,
                    subroutines={},
                    index_buffer=self._index_buffer_,
                    framebuffer=intermediate_framebuffer,
                    enable_only=self._enable_only_,
                    context_state=ContextState(
                        blend_func=moderngl.ADDITIVE_BLENDING,
                        blend_equation=moderngl.MAX
                    ),
                    mode=moderngl.TRIANGLES
                ))
                CopyPass()._render(
                    input_framebuffer=intermediate_framebuffer,
                    output_framebuffer=target_framebuffer,
                    mobject=self,
                    scene_config=scene_config
                )
