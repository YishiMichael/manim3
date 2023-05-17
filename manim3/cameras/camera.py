import numpy as np

from ..constants import (
    DOWN,
    LEFT,
    ORIGIN,
    OUT,
    RIGHT,
    UP
)
from ..config import ConfigSingleton
from ..custom_typing import (
    Mat4T,
    Vec3T
)
from ..lazy.lazy import Lazy
from ..mobjects.mobject import Mobject
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.space import SpaceUtils


class Camera(Mobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        # Positions that bound to `model_matrix`:
        # `eye`: ORIGIN
        # `target`: IN
        # `up`: UP
        # Distances that bound to `model matrix`:
        # `width`: |RIGHT - LEFT|
        # `height`: |UP - DOWN|
        # `altitude`: |ORIGIN - IN|
        self.shift(OUT).scale(np.array((
            ConfigSingleton().size.frame_width / 2.0,
            ConfigSingleton().size.frame_height / 2.0,
            ConfigSingleton().camera.altitude
        )))
        #print(self._view_matrix_.value)
        #print(self._eye_.value)

    @Lazy.variable_external
    @classmethod
    def _near_(cls) -> float:
        return ConfigSingleton().camera.near

    @Lazy.variable_external
    @classmethod
    def _far_(cls) -> float:
        return ConfigSingleton().camera.far

    @Lazy.property_external
    @classmethod
    def _width_(
        cls,
        model_matrix: Mat4T
    ) -> float:
        return SpaceUtils.norm(
            SpaceUtils.apply_affine(model_matrix, RIGHT) - SpaceUtils.apply_affine(model_matrix, LEFT)
        )

    @Lazy.property_external
    @classmethod
    def _height_(
        cls,
        model_matrix: Mat4T
    ) -> float:
        return SpaceUtils.norm(
            SpaceUtils.apply_affine(model_matrix, UP) - SpaceUtils.apply_affine(model_matrix, DOWN)
        )

    @Lazy.property_external
    @classmethod
    def _altitude_(
        cls,
        model_matrix: Mat4T
    ) -> float:
        return SpaceUtils.norm(
            SpaceUtils.apply_affine(model_matrix, ORIGIN) - SpaceUtils.apply_affine(model_matrix, OUT)
        )

    @Lazy.property_external
    @classmethod
    def _eye_(
        cls,
        model_matrix: Mat4T
    ) -> Vec3T:
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)

    #@Lazy.variable_external
    #@classmethod
    #def _target_(cls) -> Vec3T:
    #    return ORIGIN

    #@Lazy.variable_external
    #@classmethod
    #def _up_(cls) -> Vec3T:
    #    return UP

    @Lazy.property_external
    @classmethod
    def _projection_matrix_(cls) -> Mat4T:
        # Implemented in subclasses.
        return np.identity(4)

    @Lazy.property_external
    @classmethod
    def _view_matrix_(
        cls,
        eye: Vec3T
        #eye: Vec3T,
        #target: Vec3T,
        #up: Vec3T
    ) -> Mat4T:
        return SpaceUtils.matrix_from_translation(-eye)
        #m = np.identity(4)
        #m[:3, 3] = model_matrix[:3, 3] * -1.0
        #return m

    #    z = SpaceUtils.normalize(eye - target)
    #    x = SpaceUtils.normalize(np.cross(up, z))
    #    y = SpaceUtils.normalize(np.cross(z, x))
    #    rot_mat = np.vstack((x, y, z))

    #    m = np.identity(4)
    #    m[:3, :3] = rot_mat
    #    m[:3, 3] = -rot_mat @ eye
    #    return m

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

    #def set_view(
    #    self,
    #    *,
    #    width: float | None = None,
    #    height: float | None = None,
    #    near: float | None = None,
    #    far: float | None = None,
    #    altitude: float | None = None,
    #    eye: Vec3T | None = None,
    #    target: Vec3T | None = None,
    #    up: Vec3T | None = None
    #):
    #    if width is not None:
    #        self._width_ = width
    #    if height is not None:
    #        self._height_ = height
    #    if near is not None:
    #        self._near_ = near
    #    if far is not None:
    #        self._far_ = far
    #    if altitude is not None:
    #        self._altitude_ = altitude
    #    if eye is not None:
    #        self._eye_ = eye
    #    if target is not None:
    #        self._target_ = target
    #    if up is not None:
    #        self._up_ = up
    #    return self
