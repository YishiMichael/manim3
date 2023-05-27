import numpy as np

from ...config import ConfigSingleton
from ...constants import (
    ORIGIN,
    OUT,
    RIGHT,
    UP
)
from ...custom_typing import (
    NP_44f8,
    NP_2f8,
    NP_3f8,
    NP_f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...rendering.gl_buffer import UniformBlockBuffer
from ...utils.space import SpaceUtils
from ..mobject import Mobject


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

    @Lazy.variable_array
    @classmethod
    def _near_(cls) -> NP_f8:
        return ConfigSingleton().camera.near * np.ones(())

    @Lazy.variable_array
    @classmethod
    def _far_(cls) -> NP_f8:
        return ConfigSingleton().camera.far * np.ones(())

    @Lazy.property_array
    @classmethod
    def _projection_matrix_(cls) -> NP_44f8:
        # Implemented in subclasses.
        return np.identity(4)

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(cls) -> NP_x3f8:
        return np.array((OUT,))

    @Lazy.property_array
    @classmethod
    def _target_(
        cls,
        model_matrix: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)

    @Lazy.property_array
    @classmethod
    def _eye_(
        cls,
        model_matrix: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply_affine(model_matrix, OUT)

    @Lazy.property_array
    @classmethod
    def _frame_radii_(
        cls,
        model_matrix: NP_44f8,
        target: NP_3f8
    ) -> NP_2f8:
        return np.array((
            SpaceUtils.norm(SpaceUtils.apply_affine(model_matrix, RIGHT) - target),
            SpaceUtils.norm(SpaceUtils.apply_affine(model_matrix, UP) - target)
        ))

    @Lazy.property_array
    @classmethod
    def _altitude_(
        cls,
        eye: NP_3f8,
        target: NP_3f8
    ) -> NP_f8:
        return SpaceUtils.norm(eye - target)

    @Lazy.property_array
    @classmethod
    def _view_matrix_(
        cls,
        eye: NP_3f8
    ) -> NP_44f8:
        return SpaceUtils.matrix_from_translation(-eye)

    @Lazy.property_array
    @classmethod
    def _projection_view_matrix_(
        cls,
        projection_matrix: NP_44f8,
        view_matrix: NP_44f8
    ) -> NP_44f8:
        return projection_matrix @ view_matrix

    @Lazy.property
    @classmethod
    def _camera_uniform_block_buffer_(
        cls,
        projection_view_matrix: NP_44f8,
        eye: NP_3f8,
        frame_radii: NP_2f8
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
