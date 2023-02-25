__all__ = ["Camera"]


from typing import Self
import numpy as np

from ..constants import (
    ORIGIN,
    OUT,
    UP
)
from ..custom_typing import (
    Mat4T,
    Vec3T
)
from ..rendering.config import ConfigSingleton
from ..rendering.glsl_buffers import UniformBlockBuffer
from ..utils.lazy import (
    LazyObject,
    LazyWrapper,
    lazy_property,
    lazy_object_raw,
    lazy_property_raw
)
from ..utils.space import SpaceUtils


class Camera(LazyObject):
    __slots__ = ()

    @lazy_object_raw
    @staticmethod
    def _projection_matrix_() -> Mat4T:
        return NotImplemented

    @lazy_object_raw
    @staticmethod
    def _eye_() -> Vec3T:
        return ConfigSingleton().camera_altitude * OUT

    @lazy_object_raw
    @staticmethod
    def _target_() -> Vec3T:
        return ORIGIN

    @lazy_object_raw
    @staticmethod
    def _up_() -> Vec3T:
        return UP

    @lazy_property_raw
    @staticmethod
    def _view_matrix_(
        eye: Vec3T,
        target: Vec3T,
        up: Vec3T
    ) -> Mat4T:
        z = SpaceUtils.normalize(eye - target)
        x = SpaceUtils.normalize(np.cross(up, z))
        y = SpaceUtils.normalize(np.cross(z, x))
        rot_mat = np.vstack((x, y, z))

        m = np.identity(4)
        m[:3, :3] = rot_mat
        m[:3, 3] = -rot_mat @ eye
        return m

    @lazy_property
    @staticmethod
    def _ub_camera_(
        projection_matrix: Mat4T,
        view_matrix: Mat4T,
        eye: Vec3T
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_camera",
            fields=[
                "mat4 u_projection_matrix",
                "mat4 u_view_matrix",
                "vec3 u_view_position",
                "vec2 u_frame_radius"
            ],
            data={
                "u_projection_matrix": projection_matrix.T,
                "u_view_matrix": view_matrix.T,
                "u_view_position": eye,
                "u_frame_radius": np.array(ConfigSingleton().frame_size) / 2.0
            }
        )

    def set_view(
        self: Self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ) -> Self:
        if eye is not None:
            self._eye_ = LazyWrapper(eye)
        if target is not None:
            self._target_ = LazyWrapper(target)
        if up is not None:
            self._up_ = LazyWrapper(up)
        return self
