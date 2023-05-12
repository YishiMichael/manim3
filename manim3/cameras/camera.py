import numpy as np

from ..constants import (
    ORIGIN,
    OUT,
    UP
)
from ..config import ConfigSingleton
from ..custom_typing import (
    Mat4T,
    Vec3T
)
from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.space import SpaceUtils


class Camera(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        width: float | None = None,
        height: float | None = None,
        near: float | None = None,
        far: float | None = None,
        altitude: float | None = None
    ) -> None:
        super().__init__()
        if width is not None:
            self._width_ = width
        if height is not None:
            self._height_ = height
        if near is not None:
            self._near_ = near
        if far is not None:
            self._far_ = far
        if altitude is not None:
            self._altitude_ = altitude
            self._eye_ = altitude * OUT

    @Lazy.variable_external
    @classmethod
    def _width_(cls) -> float:
        return ConfigSingleton().size.frame_width

    @Lazy.variable_external
    @classmethod
    def _height_(cls) -> float:
        return ConfigSingleton().size.frame_height

    @Lazy.variable_external
    @classmethod
    def _near_(cls) -> float:
        return ConfigSingleton().camera.near

    @Lazy.variable_external
    @classmethod
    def _far_(cls) -> float:
        return ConfigSingleton().camera.far

    @Lazy.variable_external
    @classmethod
    def _altitude_(cls) -> float:
        return ConfigSingleton().camera.altitude

    @Lazy.variable_external
    @classmethod
    def _eye_(cls) -> Vec3T:
        return ConfigSingleton().camera.altitude * OUT

    @Lazy.variable_external
    @classmethod
    def _target_(cls) -> Vec3T:
        return ORIGIN

    @Lazy.variable_external
    @classmethod
    def _up_(cls) -> Vec3T:
        return UP

    @Lazy.property_external
    @classmethod
    def _projection_matrix_(cls) -> Mat4T:
        # Implemented in subclasses.
        return np.identity(4)

    @Lazy.property_external
    @classmethod
    def _view_matrix_(
        cls,
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

    @Lazy.property
    @classmethod
    def _camera_uniform_block_buffer_(
        cls,
        projection_matrix: Mat4T,
        view_matrix: Mat4T,
        eye: Vec3T,
        width: float,
        height: float
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
                "u_frame_radius": np.array((width, height)) / 2.0
            }
        )

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
