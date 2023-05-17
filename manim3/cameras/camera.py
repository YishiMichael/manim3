import numpy as np

from ..config import ConfigSingleton
from ..constants import (
    ORIGIN,
    OUT,
    RIGHT,
    UP
)
from ..custom_typing import (
    Mat4T,
    Vec2T,
    Vec3T,
    Vec3sT
)
from ..lazy.lazy import Lazy
from ..mobjects.mobject import Mobject
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.space import SpaceUtils


class Camera(Mobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        # Positions bound to `model_matrix`:
        # `target`: ORIGIN
        # `eye`: OUT
        # Distances bound to `model_matrix`:
        # `frame_radii`: (|RIGHT - ORIGIN|, |UP - ORIGIN|)
        # `altitude`: |OUT - ORIGIN|
        self.scale(np.append(
            ConfigSingleton().size.frame_radii,
            ConfigSingleton().camera.altitude
        ))

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
    def _projection_matrix_(cls) -> Mat4T:
        # Implemented in subclasses.
        return np.identity(4)

    @Lazy.property_external
    @classmethod
    def _local_sample_points_(cls) -> Vec3sT:
        return np.array((OUT,))

    @Lazy.property_external
    @classmethod
    def _target_(
        cls,
        model_matrix: Mat4T
    ) -> Vec3T:
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)

    @Lazy.property_external
    @classmethod
    def _eye_(
        cls,
        model_matrix: Mat4T
    ) -> Vec3T:
        return SpaceUtils.apply_affine(model_matrix, OUT)

    @Lazy.property_external
    @classmethod
    def _frame_radii_(
        cls,
        model_matrix: Mat4T,
        target: Vec3T
    ) -> Vec2T:
        return np.array((
            SpaceUtils.norm(SpaceUtils.apply_affine(model_matrix, RIGHT) - target),
            SpaceUtils.norm(SpaceUtils.apply_affine(model_matrix, UP) - target)
        ))

    @Lazy.property_external
    @classmethod
    def _altitude_(
        cls,
        eye: Vec3T,
        target: Vec3T
    ) -> float:
        return SpaceUtils.norm(eye - target)

    @Lazy.property_external
    @classmethod
    def _view_matrix_(
        cls,
        eye: Vec3T
    ) -> Mat4T:
        return SpaceUtils.matrix_from_translation(-eye)

    @Lazy.property_external
    @classmethod
    def _projection_view_matrix_(
        cls,
        projection_matrix: Mat4T,
        view_matrix: Mat4T
    ) -> Mat4T:
        return projection_matrix @ view_matrix

    @Lazy.property
    @classmethod
    def _camera_uniform_block_buffer_(
        cls,
        projection_view_matrix: Mat4T,
        eye: Vec3T,
        frame_radii: Vec2T
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_camera",
            fields=[
                "mat4 u_projection_view_matrix",
                "vec3 u_view_position",
                "vec2 u_frame_radii"
            ],
            data={
                "u_projection_view_matrix": projection_view_matrix.T,
                "u_view_position": eye,
                "u_frame_radii": frame_radii
            }
        )
