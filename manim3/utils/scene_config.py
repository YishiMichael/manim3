__all__ = ["SceneConfig"]


import numpy as np

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..custom_typing import (
    Vector3ArrayType,
    Vector4ArrayType,
    Vector4Type
)
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_initializer_writable
)
from ..utils.renderable import UniformBlockBuffer


class SceneConfig(LazyBase):
    @lazy_property_initializer_writable
    @staticmethod
    def _camera_() -> Camera:
        return PerspectiveCamera()

    @lazy_property_initializer_writable
    @staticmethod
    def _ambient_light_color_() -> Vector4Type:
        return np.ones(4)

    @lazy_property_initializer_writable
    @staticmethod
    def _point_light_positions_() -> Vector3ArrayType:
        return np.ones((3, 0))

    @lazy_property_initializer_writable
    @staticmethod
    def _point_light_colors_() -> Vector4ArrayType:
        return np.ones((4, 0))

    @lazy_property_initializer_writable
    @staticmethod
    def _ub_lights_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer({
            "u_ambient_light_color": "vec4",
            "u_point_light_positions": "vec3[NUM_U_POINT_LIGHT_POSITIONS]",
            "u_point_light_colors": "vec4[NUM_U_POINT_LIGHT_COLORS]"
        })

    @lazy_property
    @staticmethod
    def _ub_lights_(
        ub_lights_o: UniformBlockBuffer,
        ambient_light_color: Vector4Type,
        point_light_positions: Vector3ArrayType,
        point_light_colors: Vector4ArrayType
    ) -> UniformBlockBuffer:
        ub_lights_o.write({
            "u_ambient_light_color": ambient_light_color,
            "u_point_light_positions": point_light_positions,
            "u_point_light_colors": point_light_colors
        })
        return ub_lights_o
