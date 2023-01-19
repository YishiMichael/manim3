__all__ = ["ShapeMobject"]


from dataclasses import dataclass
from typing import Callable

import moderngl

from ..custom_typing import (
    ColorType,
    Mat4T,
    Real,
    Vec4T
)
from ..geometries.shape_geometry import ShapeGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.stroke_mobject import StrokeMobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import Framebuffer
from ..utils.scene_config import SceneConfig
from ..utils.shape import Shape


@dataclass
class StrokeConfig:
    width: Real | None
    color: ColorType | None
    dilate: Real | None
    single_sided: bool | None


class ShapeMobject(MeshMobject):
    def __init__(self, shape: Shape):
        super().__init__()
        self._set_shape(shape)

    @lazy_property_initializer_writable
    @staticmethod
    def _shape_() -> Shape:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _geometry_(shape: Shape) -> ShapeGeometry:
        return ShapeGeometry(shape)

    @lazy_property_initializer
    @staticmethod
    def _stroke_mobjects_() -> list[StrokeMobject]:
        return []

    def _set_shape(self, shape: Shape):
        self._shape_ = shape
        for stroke in self._stroke_mobjects_:
            stroke._multi_line_string_ = shape._multi_line_string_3d_
        return self

    def _set_model_matrix(self, matrix: Mat4T):
        super()._set_model_matrix(matrix)
        for stroke in self._stroke_mobjects_:
            stroke._set_model_matrix(matrix)
        return self

    #@lazy_property_initializer_writable
    #@staticmethod
    #def _enable_only_() -> int:
    #    return moderngl.BLEND

    #@lazy_property_initializer
    #@staticmethod
    #def _foreground_stroke_configs_() -> list[StrokeConfig]:
    #    return []

    #@lazy_property_initializer
    #@staticmethod
    #def _background_stroke_configs_() -> list[StrokeConfig]:
    #    return []



    #@lazy_property
    #@staticmethod
    #def _stroke_mobjects_(
    #    foreground_stroke_configs: list[StrokeConfig],
    #    shape: Shape,
    #    model_matrix: Mat4T
    #) -> list[StrokeMobject]:
    #    result: list[StrokeMobject] = []
    #    for line_string in shape._multi_line_string_._children_:
    #        if line_string._kind_ == "point":
    #            continue
    #        position = np.insert(line_string._coords_, 2, 0.0, axis=1)
    #        is_loop = line_string._kind_ == "linear_ring"
    #        if is_loop:
    #            position = position[:-1]
    #        for stroke_config in foreground_stroke_configs:
    #            stroke = StrokeMobject(position, is_loop)
    #            if (width := stroke_config.width) is not None:
    #                stroke._width_ = width
    #            if (color := stroke_config.color) is not None:
    #                stroke._color_ = color
    #            if (dilate := stroke_config.dilate) is not None:
    #                stroke._dilate_ = dilate
    #            if (single_sided := stroke_config.single_sided) is not None:
    #                stroke._single_sided_ = single_sided
    #            stroke.apply_transform_locally(model_matrix)
    #            result.append(stroke)
    #    return result

    #@lazy_property
    #@staticmethod
    #def _background_stroke_mobjects_(
    #    background_stroke_configs: list[StrokeConfig],
    #    shape: Shape,
    #    model_matrix: Mat4T
    #) -> list[StrokeMobject]:
    #    # TODO
    #    result: list[StrokeMobject] = []
    #    for line_string in shape._multi_line_string_._children_:
    #        if line_string._kind_ == "point":
    #            continue
    #        position = np.insert(line_string._coords_, 2, 0.0, axis=1)
    #        is_loop = line_string._kind_ == "linear_ring"
    #        if is_loop:
    #            position = position[:-1]
    #        for stroke_config in background_stroke_configs:
    #            stroke = StrokeMobject(position, is_loop)
    #            if (width := stroke_config.width) is not None:
    #                stroke._width_ = width
    #            if (color := stroke_config.color) is not None:
    #                stroke._color_ = color
    #            if (dilate := stroke_config.dilate) is not None:
    #                stroke._dilate_ = dilate
    #            if (single_sided := stroke_config.single_sided) is not None:
    #                stroke._single_sided_ = single_sided
    #            stroke.apply_transform_locally(model_matrix)
    #            result.append(stroke)
    #    return result

    def _set_fill_locally(self, color: ColorType | Callable[..., Vec4T]):
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
            mobject._set_fill_locally(color=color)
        return self

    @_stroke_mobjects_.updater
    def _add_stroke_locally(
        self,
        *,
        width: Real | None = None,
        color: ColorType | None = None,
        dilate: Real | None = None,
        single_sided: bool | None = None,
        #background: bool = False
    ):
        stroke = StrokeMobject(self._shape_._multi_line_string_3d_)
        if width is not None:
            stroke._width_ = width
        if color is not None:
            stroke._color_ = color
        if dilate is not None:
            stroke._dilate_ = dilate
        if single_sided is not None:
            stroke._single_sided_ = single_sided
        stroke._set_model_matrix(self._model_matrix_)
        #stroke_config = StrokeConfig(
        #    width=width,
        #    color=color,
        #    dilate=dilate,
        #    single_sided=single_sided
        #)
        #if not background:
        #    self._foreground_stroke_configs_.append(stroke_config)
        #else:
        self._stroke_mobjects_.append(stroke)
        return self

    def add_stroke(
        self,
        *,
        width: Real | None = None,
        color: ColorType | None = None,
        dilate: Real | None = None,
        single_sided: bool | None = None,
        #background: bool = False,
        broadcast: bool = True
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, ShapeMobject):
                continue
            mobject._add_stroke_locally(
                width=width,
                color=color,
                dilate=dilate,
                single_sided=single_sided,
                #background=background
            )
        return self

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        #for stroke in self._background_stroke_mobjects_:
        #    stroke._render(scene_config, target_framebuffer)
        super()._render(scene_config, target_framebuffer)
        for stroke in self._stroke_mobjects_:
            stroke._render(scene_config, target_framebuffer)


#class ShapeStrokeMobject(ShapeMobject):
#    def __init__(
#        self,
#        shape: Shape,
#        stroke_config: StrokeConfig
#    ):
#        super().__init__(shape)
#        self._width_ = stroke_config.width
#        self._round_end_ = stroke_config.round_end
#        self._single_sided_ = stroke_config.single_sided
#        self._distance_dilate_ = stroke_config.distance_dilate
#        if (color := stroke_config.color) is not None:
#            self._color_ = color

#    @lazy_property_initializer
#    @staticmethod
#    def _ub_stroke_o_() -> UniformBlockBuffer:
#        return UniformBlockBuffer("ub_stroke", [
#            "float u_distance_dilate"
#        ])

#    @lazy_property
#    @staticmethod
#    def _ub_stroke_(
#        ub_stroke_o: UniformBlockBuffer,
#        distance_dilate: Real
#    ) -> UniformBlockBuffer:
#        ub_stroke_o.write({
#            "u_distance_dilate": np.array(distance_dilate)
#        })
#        return ub_stroke_o

#    @lazy_property_initializer_writable
#    @staticmethod
#    def _width_() -> Real:
#        return NotImplemented

#    @lazy_property_initializer_writable
#    @staticmethod
#    def _round_end_() -> bool:
#        return NotImplemented

#    @lazy_property_initializer_writable
#    @staticmethod
#    def _single_sided_() -> bool:
#        return NotImplemented

#    @lazy_property_initializer_writable
#    @staticmethod
#    def _distance_dilate_() -> Real:
#        return NotImplemented

#    @lazy_property
#    @staticmethod
#    def _geometry_(shape: Shape, width: Real, round_end: bool, single_sided: bool) -> ShapeStrokeGeometry:
#        return ShapeStrokeGeometry(shape, width, round_end, single_sided)

#    @lazy_property_initializer
#    @staticmethod
#    def _attributes_o_() -> AttributesBuffer:
#        return AttributesBuffer([
#            "vec3 in_position",
#            "vec3 in_normal",
#            "vec2 in_uv",
#            "vec4 in_color",
#            "float in_distance"
#        ])

#    @lazy_property
#    @staticmethod
#    def _attributes_(
#        attributes_o: AttributesBuffer,
#        geometry: ShapeStrokeGeometry,
#        color: ColorType | Callable[..., Vec4T]
#    ) -> AttributesBuffer:
#        position = geometry._position_
#        normal = geometry._normal_
#        uv = geometry._uv_
#        distance = geometry._distance_
#        color_array = MeshMobject._calculate_color_array(color, position, normal, uv)
#        attributes_o.write({
#            "in_position": position,
#            "in_normal": normal,
#            "in_uv": uv,
#            "in_color": color_array,
#            "in_distance": distance
#        })
#        return attributes_o

#    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
#        with IntermediateTextures.register_n(1) as textures:
#            with IntermediateDepthTextures.register_n(1) as depth_textures:
#                intermediate_framebuffer = IntermediateFramebuffer(textures, depth_textures[0])
#                self._render_by_step(RenderStep(
#                    shader_str=Renderable._read_shader("shape_stroke"),
#                    texture_storages=[
#                        self._u_color_maps_
#                    ],
#                    uniform_blocks=[
#                        scene_config._camera_._ub_camera_,
#                        self._ub_model_,
#                        scene_config._ub_lights_,
#                        self._ub_stroke_
#                    ],
#                    attributes=self._attributes_,
#                    subroutines={},
#                    index_buffer=self._index_buffer_,
#                    framebuffer=intermediate_framebuffer,
#                    enable_only=self._enable_only_,
#                    context_state=ContextState(
#                        blend_func=moderngl.ADDITIVE_BLENDING,
#                        blend_equation=moderngl.MAX
#                    ),
#                    mode=moderngl.TRIANGLES
#                ))
#                CopyPass()._render(
#                    input_framebuffer=intermediate_framebuffer,
#                    output_framebuffer=target_framebuffer,
#                    mobject=self,
#                    scene_config=scene_config
#                )
