from __future__ import annotations


from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation


from ..constants.constants import (
    DL,
    ORIGIN,
    OUT,
    RIGHT,
    UP,
    UR
)
from ..constants.custom_typing import (
    NP_2f8,
    NP_3f8,
    NP_44f8,
    NP_f8,
    NP_x3f8
)
from ..lazy.lazy import Lazy
from ..rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ..toplevel.toplevel import Toplevel
from .animatable.animatable import Animatable
from .arrays.animatable_float import AnimatableFloat
from .arrays.model_matrix import ModelMatrix
from .model import Model


class Camera(Model):
    __slots__ = ()

    def __init__(
        self: Self,
        near: float | None = None,
        far: float | None = None
    ) -> None:
        super().__init__()
        # Positions bound to `model_matrix`:
        # `target`: ORIGIN
        # `eye`: OUT
        # Distances bound to `model_matrix`:
        # `frame_radii`: (|RIGHT - ORIGIN|, |UP - ORIGIN|)
        # `distance`: |OUT - ORIGIN|
        self.scale(np.append(
            Toplevel._get_config().frame_radii,
            Toplevel._get_config().camera_distance
        ))
        if near is not None:
            self._near_ = AnimatableFloat(near)
        if far is not None:
            self._far_ = AnimatableFloat(far)

    @Animatable.interpolate.register_descriptor()
    @Lazy.volatile()
    @staticmethod
    def _near_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel._get_config().camera_near)

    @Animatable.interpolate.register_descriptor()
    @Lazy.volatile()
    @staticmethod
    def _far_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel._get_config().camera_far)

    @Lazy.property()
    @staticmethod
    def _target_(
        model_matrix__array: NP_44f8
    ) -> NP_3f8:
        return ModelMatrix._apply(model_matrix__array, ORIGIN)

    @Lazy.property()
    @staticmethod
    def _eye_(
        model_matrix__array: NP_44f8
    ) -> NP_3f8:
        return ModelMatrix._apply(model_matrix__array, OUT)

    @Lazy.property()
    @staticmethod
    def _frame_radii_(
        model_matrix__array: NP_44f8,
        target: NP_3f8
    ) -> NP_2f8:
        return np.array((
            np.linalg.norm(ModelMatrix._apply(model_matrix__array, RIGHT) - target),
            np.linalg.norm(ModelMatrix._apply(model_matrix__array, UP) - target)
        ))

    @Lazy.property()
    @staticmethod
    def _distance_(
        eye: NP_3f8,
        target: NP_3f8
    ) -> NP_f8:
        return np.array(np.linalg.norm(eye - target))

    @Lazy.property()
    @staticmethod
    def _projection_matrix_(
        frame_radii: NP_2f8,
        near__array: NP_f8,
        far__array: NP_f8,
        distance: NP_f8
    ) -> NP_44f8:
        near = near__array
        far = far__array
        sx, sy = distance / frame_radii
        sz = -(far + near) / (far - near)
        tz = -2.0 * far * near / (far - near)
        return np.array((
            ( sx, 0.0,  0.0, 0.0),
            (0.0,  sy,  0.0, 0.0),
            (0.0, 0.0,   sz,  tz),
            (0.0, 0.0, -1.0, 0.0)
        ))

    @Lazy.property()
    @staticmethod
    def _view_matrix_(
        model_matrix__array: NP_44f8,
        eye: NP_3f8
    ) -> NP_44f8:
        model_basis = model_matrix__array[:3, :3]
        model_basis_normalized = model_basis / np.linalg.norm(model_basis, axis=0, keepdims=True)
        return (
            ModelMatrix._matrix_from_rotate(-Rotation.from_matrix(model_basis_normalized).as_rotvec())
            @ ModelMatrix._matrix_from_shift(-eye)
        )

    @Lazy.property()
    @staticmethod
    def _camera_uniform_block_buffer_(
        projection_matrix: NP_44f8,
        view_matrix: NP_44f8,
        frame_radii: NP_2f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_camera",
            field_declarations=(
                "mat4 u_projection_matrix",
                "mat4 u_view_matrix",
                "vec2 u_frame_radii"
            ),
            data_dict={
                "u_projection_matrix": projection_matrix.T,
                "u_view_matrix": view_matrix.T,
                "u_frame_radii": frame_radii
            }
        )

    @Lazy.property()
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        # Form a rectangle covering the screen,
        # so that aligning on border can be achieved by aligning with the camera.
        return np.array((UR, DL))
