import numpy as np

from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from ..lighting.ambient_light import AmbientLight
from ..lighting.point_light import PointLight
from ..rendering.gl_buffer import UniformBlockBuffer


#class PointLight(LazyObject):
#    __slots__ = ()

#    @Lazy.variable_external
#    @classmethod
#    def _position_(cls) -> Vec3T:
#        return np.zeros(3)

#    @Lazy.variable_external
#    @classmethod
#    def _color_(cls) -> Vec3T:
#        return np.ones(3)

#    @Lazy.variable_external
#    @classmethod
#    def _opacity_(cls) -> float:
#        return 1.0

#    def set_style(
#        self,
#        *,
#        position: Vec3T | None = None,
#        color: ColorT | None = None,
#        opacity: float | None = None
#    ):
#        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
#        if position is not None:
#            self._position_ = position
#        if color_component is not None:
#            self._color_ = color_component
#        if opacity_component is not None:
#            self._opacity_ = opacity_component
#        return self


class Lighting(LazyObject):
    __slots__ = ()

    #@Lazy.variable_external
    #@classmethod
    #def _ambient_light_color_(cls) -> Vec3T:
    #    return np.ones(3)

    #@Lazy.variable_external
    #@classmethod
    #def _ambient_light_opacity_(cls) -> float:
    #    return 1.0

    @Lazy.variable_collection
    @classmethod
    def _ambient_lights_(cls) -> list[AmbientLight]:
        return []

    @Lazy.variable_collection
    @classmethod
    def _point_lights_(cls) -> list[PointLight]:
        return []

    @Lazy.property
    @classmethod
    def _lighting_uniform_block_buffer_(
        cls,
        _ambient_lights_: list[AmbientLight],
        _point_lights_: list[PointLight]
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_lighting",
            fields=[
                "AmbientLight u_ambient_lights[NUM_U_AMBIENT_LIGHTS]",
                "PointLight u_point_lights[NUM_U_POINT_LIGHTS]"
            ],
            child_structs={
                "AmbientLight": [
                    "vec4 color"
                ],
                "PointLight": [
                    "vec3 position",
                    "vec4 color"
                ]
            },
            array_lens={
                "NUM_U_AMBIENT_LIGHTS": len(_ambient_lights_),
                "NUM_U_POINT_LIGHTS": len(_point_lights_)
            },
            data={
                "u_ambient_lights.color": np.array([
                    ambient_light._color_vec4_.value
                    for ambient_light in _ambient_lights_
                ]),
                "u_point_lights.position": np.array([
                    point_light._position_.value
                    for point_light in _point_lights_
                ]),
                "u_point_lights.color": np.array([
                    point_light._color_vec4_.value
                    for point_light in _point_lights_
                ])
            }
        )

    #def set_ambient_light(
    #    self,
    #    *,
    #    color: ColorT | None = None,
    #    opacity: float | None = None
    #):
    #    color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
    #    if color_component is not None:
    #        self._ambient_light_color_ = color_component
    #    if opacity_component is not None:
    #        self._ambient_light_opacity_ = opacity_component
    #    return self

    #def add_point_light(
    #    self,
    #    *,
    #    position: Vec3T | None = None,
    #    color: ColorT | None = None,
    #    opacity: float | None = None
    #):
    #    self._point_lights_.append(PointLight().set_style(
    #        position=position,
    #        color=color,
    #        opacity=opacity
    #    ))
    #    return self

    #def set_point_light(
    #    self,
    #    *,
    #    index: int | None = None,
    #    position: Vec3T | None = None,
    #    color: ColorT | None = None,
    #    opacity: float | None = None
    #):
    #    if self._point_lights_:
    #        if index is None:
    #            index = 0
    #        self._point_lights_[index].set_style(
    #            position=position,
    #            color=color,
    #            opacity=opacity
    #        )
    #    else:
    #        if index is not None:
    #            raise IndexError
    #        if any(param is not None for param in (
    #            position,
    #            color,
    #            opacity
    #        )):
    #            self.add_point_light(
    #                position=position,
    #                color=color,
    #                opacity=opacity
    #            )
    #    return self
