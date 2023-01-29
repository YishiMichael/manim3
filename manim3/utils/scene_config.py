__all__ = ["SceneConfig"]


import numpy as np

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..custom_typing import (
    Real,
    Vec3sT,
    Vec4sT,
    Vec4T
)
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_writable
)
from ..utils.render_procedure import UniformBlockBuffer


class SceneConfig(LazyBase):
    @lazy_property_writable
    @staticmethod
    def _camera_() -> Camera:
        return PerspectiveCamera()

    @lazy_property_writable
    @staticmethod
    def _ambient_light_color_() -> Vec4T:
        return np.ones(4)

    @lazy_property_writable
    @staticmethod
    def _point_light_positions_() -> Vec3sT:
        return np.ones((0, 3))

    @lazy_property_writable
    @staticmethod
    def _point_light_colors_() -> Vec4sT:
        return np.ones((0, 4))

    @lazy_property_writable
    @staticmethod
    def _ub_lights_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_lights", [
            "vec4 u_ambient_light_color",
            "PointLight u_point_lights[NUM_U_POINT_LIGHTS]"
        ], {
            "PointLight": [
                "vec3 position",
                "vec4 color"
            ]
        })

    @lazy_property
    @staticmethod
    def _ub_lights_(
        ub_lights_o: UniformBlockBuffer,
        ambient_light_color: Vec4T,
        point_light_positions: Vec3sT,
        point_light_colors: Vec4sT
    ) -> UniformBlockBuffer:
        ub_lights_o.write({
            "u_ambient_light_color": ambient_light_color,
            "u_point_lights": {
                "position": point_light_positions,
                "color": point_light_colors
            }
        })
        return ub_lights_o
