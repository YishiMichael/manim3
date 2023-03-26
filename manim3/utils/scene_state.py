__all__ = ["SceneState"]


import numpy as np

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..custom_typing import (
    ColorType,
    Vec3T
)
from ..lazy.core import LazyObject
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.color import ColorUtils


class PointLight(LazyObject):
    __slots__ = ()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _position_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    def set_style(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorType | None = None,
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


class SceneState(LazyObject):
    __slots__ = ()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _background_color_(cls) -> Vec3T:
        return np.zeros(3)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _background_opacity_(cls) -> float:
        return 1.0

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _ambient_light_color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _ambient_light_opacity_(cls) -> float:
        return 1.0

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _point_lights_(cls) -> list[PointLight]:
        return []

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_lights_(
        cls,
        ambient_light_color: Vec3T,
        ambient_light_opacity: float,
        _point_lights_: list[PointLight]
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_lights",
            fields=[
                "vec4 u_ambient_light_color",
                "PointLight u_point_lights[NUM_U_POINT_LIGHTS]"
            ],
            child_structs={
                "PointLight": [
                    "vec3 position",
                    "vec4 color"
                ]
            },
            dynamic_array_lens={
                "NUM_U_POINT_LIGHTS": len(_point_lights_)
            },
            data={
                "u_ambient_light_color": np.append(ambient_light_color, ambient_light_opacity),
                "u_point_lights": {
                    "position": np.array([
                        point_light._position_.value
                        for point_light in _point_lights_
                    ]),
                    "color": np.array([
                        np.append(point_light._color_.value, point_light._opacity_.value)
                        for point_light in _point_lights_
                    ])
                }
            }
        )

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _camera_(cls) -> Camera:
        return PerspectiveCamera()

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        self._camera_.set_view(
            eye=eye,
            target=target,
            up=up
        )
        return self

    def set_background(
        self,
        *,
        color: ColorType | None = None,
        opacity: float | None = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        if color_component is not None:
            self._background_color_ = color_component
        if opacity_component is not None:
            self._background_opacity_ = opacity_component
        return self

    def set_ambient_light(
        self,
        *,
        color: ColorType | None = None,
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
        color: ColorType | None = None,
        opacity: float | None = None
    ):
        self._point_lights_.add(PointLight().set_style(
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
        color: ColorType | None = None,
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

    def set_style(
        self,
        *,
        background_color: ColorType | None = None,
        background_opacity: float | None = None,
        ambient_light_color: ColorType | None = None,
        ambient_light_opacity: float | None = None,
        point_light_position: Vec3T | None = None,
        point_light_color: ColorType | None = None,
        point_light_opacity: float | None = None
    ):
        self.set_background(
            color=background_color,
            opacity=background_opacity
        )
        self.set_ambient_light(
            color=ambient_light_color,
            opacity=ambient_light_opacity
        )
        self.set_point_light(
            index=None,
            position=point_light_position,
            color=point_light_color,
            opacity=point_light_opacity
        )
        return self
