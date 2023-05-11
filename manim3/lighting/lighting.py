import numpy as np

from ..custom_typing import (
    ColorT,
    Vec3T
)
from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.color import ColorUtils


class PointLight(LazyObject):
    __slots__ = ()

    @Lazy.variable_external
    @classmethod
    def _position_(cls) -> Vec3T:
        return np.zeros(3)

    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    def set_style(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        if position is not None:
            self._position_ = position
        if color_component is not None:
            self._color_ = color_component
        if opacity_component is not None:
            self._opacity_ = opacity_component
        return self


class Lighting(LazyObject):
    __slots__ = ()

    @Lazy.variable_external
    @classmethod
    def _ambient_light_color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable_external
    @classmethod
    def _ambient_light_opacity_(cls) -> float:
        return 1.0

    @Lazy.variable_external
    @classmethod
    def _ambient_strength_(cls) -> float:
        return 1.0

    @Lazy.variable_external
    @classmethod
    def _specular_strength_(cls) -> float:
        return 0.5

    @Lazy.variable_external
    @classmethod
    def _shininess_(cls) -> float:
        return 32.0

    @Lazy.variable_collection
    @classmethod
    def _point_lights_(cls) -> list[PointLight]:
        return []

    @Lazy.property
    @classmethod
    def _lighting_uniform_block_buffer_(
        cls,
        ambient_light_color: Vec3T,
        ambient_light_opacity: float,
        ambient_strength: float,
        specular_strength: float,
        shininess: float,
        _point_lights_: list[PointLight]
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_lighting",
            fields=[
                "vec4 u_ambient_light_color",
                "float u_ambient_strength",
                "float u_specular_strength",
                "float u_shininess",
                "PointLight u_point_lights[NUM_U_POINT_LIGHTS]"
            ],
            child_structs={
                "PointLight": [
                    "vec3 position",
                    "vec4 color"
                ]
            },
            array_lens={
                "NUM_U_POINT_LIGHTS": len(_point_lights_)
            },
            data={
                "u_ambient_light_color": np.append(ambient_light_color, ambient_light_opacity),
                "u_ambient_strength": np.array(ambient_strength),
                "u_specular_strength": np.array(specular_strength),
                "u_shininess": np.array(shininess),
                "u_point_lights.position": np.array([
                    point_light._position_.value
                    for point_light in _point_lights_
                ]),
                "u_point_lights.color": np.array([
                    np.append(point_light._color_.value, point_light._opacity_.value)
                    for point_light in _point_lights_
                ])
            }
        )

    def set_ambient_light(
        self,
        *,
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        if color_component is not None:
            self._ambient_light_color_ = color_component
        if opacity_component is not None:
            self._ambient_light_opacity_ = opacity_component
        return self

    def add_point_light(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        self._point_lights_.append(PointLight().set_style(
            position=position,
            color=color,
            opacity=opacity
        ))
        return self

    def set_point_light(
        self,
        *,
        index: int | None = None,
        position: Vec3T | None = None,
        color: ColorT | None = None,
        opacity: float | None = None
    ):
        if self._point_lights_:
            if index is None:
                index = 0
            self._point_lights_[index].set_style(
                position=position,
                color=color,
                opacity=opacity
            )
        else:
            if index is not None:
                raise IndexError
            if any(param is not None for param in (
                position,
                color,
                opacity
            )):
                self.add_point_light(
                    position=position,
                    color=color,
                    opacity=opacity
                )
        return self
