__all__ = ["Camera"]


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
from ..rendering.render_procedure import UniformBlockBuffer
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_writable
)
from ..utils.space_ops import SpaceOps


class Camera(LazyBase):
    @lazy_property
    @staticmethod
    def _projection_matrix_() -> Mat4T:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _eye_() -> Vec3T:
        return ConfigSingleton().camera_altitude * OUT

    @lazy_property_writable
    @staticmethod
    def _target_() -> Vec3T:
        return ORIGIN

    @lazy_property_writable
    @staticmethod
    def _up_() -> Vec3T:
        return UP

    @lazy_property
    @staticmethod
    def _view_matrix_(
        eye: Vec3T,
        target: Vec3T,
        up: Vec3T
    ) -> Mat4T:
        z = SpaceOps.normalize(eye - target)
        x = SpaceOps.normalize(np.cross(up, z))
        y = SpaceOps.normalize(np.cross(z, x))
        rot_mat = np.vstack((x, y, z))

        m = np.identity(4)
        m[:3, :3] = rot_mat
        m[:3, 3] = -rot_mat @ eye
        return m

    @lazy_property_writable
    @staticmethod
    def _ub_camera_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_camera", [
            "mat4 u_projection_matrix",
            "mat4 u_view_matrix",
            "vec3 u_view_position",
            "vec2 u_frame_radius"
        ])

    @lazy_property
    @staticmethod
    def _ub_camera_(
        ub_camera_o: UniformBlockBuffer,
        projection_matrix: Mat4T,
        view_matrix: Mat4T,
        eye: Vec3T
    ) -> UniformBlockBuffer:
        return ub_camera_o.write({
            "u_projection_matrix": projection_matrix.T,
            "u_view_matrix": view_matrix.T,
            "u_view_position": eye,
            "u_frame_radius": np.array(ConfigSingleton().frame_size) / 2.0
        })

    def set_view(
        self,
        *,
        eye: Vec3T | None = None,
        target: Vec3T | None = None,
        up: Vec3T | None = None
    ):
        if eye is not None:
            self._eye_ = eye
        if target is not None:
            self._target_ = target
        if up is not None:
            self._up_ = up
        return self
