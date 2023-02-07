__all__ = ["SceneConfig"]


from typing import ClassVar
import numpy as np

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..custom_typing import (
    ColorType,
    Real,
    Vec3T
)
from ..rendering.render_procedure import UniformBlockBuffer
from ..utils.color import ColorUtils
from ..utils.lazy import (
    LazyBase,
    LazyData,
    lazy_basedata,
    lazy_property,
    lazy_slot
)


class SceneConfig(LazyBase):
    __slots__ = ()

    _POINT_LIGHT_DTYPE: ClassVar[np.dtype] = np.dtype([
        ("position", (np.float_, (3,))),
        ("color", (np.float_, (3,))),
        ("opacity", (np.float_)),
    ])

    @lazy_basedata
    @staticmethod
    def _background_color_() -> Vec3T:
        return np.zeros(3)

    @lazy_basedata
    @staticmethod
    def _background_opacity_() -> Real:
        return 1.0

    @lazy_basedata
    @staticmethod
    def _ambient_light_color_() -> Vec3T:
        return np.ones(3)

    @lazy_basedata
    @staticmethod
    def _ambient_light_opacity_() -> Real:
        return 1.0

    @lazy_basedata
    @staticmethod
    def _point_lights_() -> np.ndarray:
        return np.zeros(0, dtype=SceneConfig._POINT_LIGHT_DTYPE)

    @lazy_property
    @staticmethod
    def _ub_lights_(
        ambient_light_color: Vec3T,
        ambient_light_opacity: Real,
        point_lights: np.ndarray
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
                "NUM_U_POINT_LIGHTS": len(point_lights)
            },
            data={
                "u_ambient_light_color": np.append(ambient_light_color, ambient_light_opacity),
                "u_point_lights": {
                    "position": point_lights["position"],
                    "color": np.append(point_lights["color"], point_lights["opacity"][:, None], axis=1)
                }
            }
        )

    @lazy_slot
    @staticmethod
    def _camera() -> Camera:
        return PerspectiveCamera()

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        self._camera.set_view(
            eye=eye,
            target=target,
            up=up
        )
        return self

    def set_background(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        if color_component is not None:
            self._background_color_ = LazyData(color_component)
        if opacity_component is not None:
            self._background_opacity_ = LazyData(opacity_component)
        return self

    def set_ambient_light(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        if color_component is not None:
            self._ambient_light_color_ = LazyData(color_component)
        if opacity_component is not None:
            self._ambient_light_opacity_ = LazyData(opacity_component)
        return self

    def add_point_light(
        self,
        *,
        position: Vec3T | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        point_light = np.array((
            position if position is not None else np.zeros(3),
            color_component if color_component is not None else np.ones(3),
            opacity_component if opacity_component is not None else 1.0
        ), dtype=self._POINT_LIGHT_DTYPE)
        self._point_lights_ = LazyData(np.append(self._point_lights_, point_light))
        return self

    def set_point_light(
        self,
        *,
        index: int | None = None,
        position: Vec3T | None = None,
        color: ColorType | None = None,
        opacity: Real | None = None
    ):
        if self._point_lights_:
            if index is None:
                index = 0
            point_lights = self._point_lights_
            color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
            point_light = np.array((
                position if position is not None else point_lights[index]["position"],
                color_component if color_component is not None else point_lights[index]["color"],
                opacity_component if opacity_component is not None else point_lights[index]["opacity"]
            ), dtype=self._POINT_LIGHT_DTYPE)
            self._point_lights_ = LazyData(np.concatenate(
                (point_lights[:index], [point_light], point_lights[index + 1:])
            ))
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
        background_opacity: Real | None = None,
        ambient_light_color: ColorType | None = None,
        ambient_light_opacity: Real | None = None,
        point_light_position: Vec3T | None = None,
        point_light_color: ColorType | None = None,
        point_light_opacity: Real | None = None
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
