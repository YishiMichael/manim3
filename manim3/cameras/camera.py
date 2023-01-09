__all__ = ["Camera"]


import numpy as np

from ..constants import (
    CAMERA_ALTITUDE,
    ORIGIN,
    OUT,
    UP
)
from ..custom_typing import (
    Mat4T,
    Vec3T
)
from ..utils.renderable import (
    Renderable,
    UniformBlockBuffer
)
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)


def normalize(array: Vec3T) -> Vec3T:
    return array / np.linalg.norm(array)


class Camera(Renderable):
    @lazy_property_initializer
    @staticmethod
    def _projection_matrix_() -> Mat4T:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _eye_() -> Vec3T:
        return CAMERA_ALTITUDE * OUT

    @lazy_property_initializer_writable
    @staticmethod
    def _target_() -> Vec3T:
        return ORIGIN

    @lazy_property_initializer_writable
    @staticmethod
    def _up_() -> Vec3T:
        return CAMERA_ALTITUDE * UP

    @lazy_property
    @staticmethod
    def _view_matrix_(
        eye: Vec3T,
        target: Vec3T,
        up: Vec3T
    ) -> Mat4T:
        z = normalize(eye - target)
        x = normalize(np.cross(up, z))
        y = normalize(np.cross(z, x))
        rot_mat = np.vstack((x, y, z))

        m = np.identity(4)
        m[:3, :3] = rot_mat.T
        m[3, :3] = -rot_mat @ eye
        return m

    @lazy_property_initializer_writable
    @staticmethod
    def _ub_camera_matrices_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_camera_matrices", [
            "mat4 u_projection_matrix",
            "mat4 u_view_matrix",
            "vec3 u_view_position"
        ])

    @lazy_property
    @staticmethod
    def _ub_camera_matrices_(
        ub_camera_matrices_o: UniformBlockBuffer,
        projection_matrix: Mat4T,
        view_matrix: Mat4T,
        eye: Vec3T
    ) -> UniformBlockBuffer:
        ub_camera_matrices_o.write({
            "u_projection_matrix": projection_matrix,
            "u_view_matrix": view_matrix,
            "u_view_position": eye
        })
        return ub_camera_matrices_o
