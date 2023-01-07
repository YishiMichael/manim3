__all__ = ["Camera"]


import numpy as np

from ..constants import (
    CAMERA_ALTITUDE,
    ORIGIN,
    OUT,
    UP
)
from ..custom_typing import (
    Matrix44Type,
    Vector3Type
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


def normalize(array: Vector3Type) -> Vector3Type:
    return array / np.linalg.norm(array)


class Camera(Renderable):
    @lazy_property_initializer
    @staticmethod
    def _projection_matrix_() -> Matrix44Type:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _eye_() -> Vector3Type:
        return CAMERA_ALTITUDE * OUT

    @lazy_property_initializer_writable
    @staticmethod
    def _target_() -> Vector3Type:
        return ORIGIN

    @lazy_property_initializer_writable
    @staticmethod
    def _up_() -> Vector3Type:
        return CAMERA_ALTITUDE * UP

    @lazy_property
    @staticmethod
    def _view_matrix_(
        eye: Vector3Type,
        target: Vector3Type,
        up: Vector3Type
    ) -> Matrix44Type:
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
        return UniformBlockBuffer({
            "u_projection_matrix": "mat4",
            "u_view_matrix": "mat4",
            "u_view_position": "vec3"
        })

    @lazy_property
    @staticmethod
    def _ub_camera_matrices_(
        ub_camera_matrices_o: UniformBlockBuffer,
        projection_matrix: Matrix44Type,
        view_matrix: Matrix44Type,
        eye: Vector3Type
    ) -> UniformBlockBuffer:
        ub_camera_matrices_o.write({
            "u_projection_matrix": projection_matrix,
            "u_view_matrix": view_matrix,
            "u_view_position": eye
        })
        return ub_camera_matrices_o
