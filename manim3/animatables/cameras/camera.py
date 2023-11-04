from __future__ import annotations


from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation


from ...constants.constants import (
    ORIGIN,
    OUT,
    RIGHT,
    UP
)
from ...constants.custom_typing import (
    NP_2f8,
    NP_3f8,
    NP_44f8,
    NP_f8
    #NP_x3f8
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...toplevel.toplevel import Toplevel
from ...utils.space_utils import SpaceUtils
from ..animatable.animatable import AnimatableMeta
from ..arrays.animatable_float import AnimatableFloat
from ..models.model import Model
#from ..mobject.remodel_handlers.rotate_remodel_handler import RotateRemodelHandler
#from ..mobject.remodel_handlers.shift_remodel_handler import ShiftRemodelHandler
#from ..mobject.mobject import Mobject


class Camera(Model):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        # Positions bound to `model_matrix`:
        # `target`: ORIGIN
        # `eye`: OUT
        # Distances bound to `model_matrix`:
        # `frame_radii`: (|RIGHT - ORIGIN|, |UP - ORIGIN|)
        # `distance`: |OUT - ORIGIN|
        self.scale(np.append(
            Toplevel.config.frame_radii,
            Toplevel.config.camera_distance
        ))

    @AnimatableMeta.register_descriptor()
    @AnimatableMeta.register_converter(AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _near_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.camera_near)

    @AnimatableMeta.register_descriptor()
    @AnimatableMeta.register_converter(AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _far_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.camera_far)

    @Lazy.property()
    @staticmethod
    def _projection_matrix_() -> NP_44f8:
        # Implemented in subclasses.
        return np.identity(4)

    #@Lazy.property()
    #@staticmethod
    #def _local_sample_positions_() -> NP_x3f8:
    #    return np.array((OUT,))

    @Lazy.property()
    @staticmethod
    def _target_(
        model_matrix__array: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply(model_matrix__array, ORIGIN)

    @Lazy.property()
    @staticmethod
    def _eye_(
        model_matrix__array: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply(model_matrix__array, OUT)

    @Lazy.property()
    @staticmethod
    def _frame_radii_(
        model_matrix__array: NP_44f8,
        target: NP_3f8
    ) -> NP_2f8:
        return np.array((
            SpaceUtils.norm(SpaceUtils.apply(model_matrix__array, RIGHT) - target),
            SpaceUtils.norm(SpaceUtils.apply(model_matrix__array, UP) - target)
        ))

    @Lazy.property()
    @staticmethod
    def _distance_(
        eye: NP_3f8,
        target: NP_3f8
    ) -> NP_f8:
        return np.array(SpaceUtils.norm(eye - target))

    @Lazy.property()
    @staticmethod
    def _view_matrix_(
        model_matrix__array: NP_44f8,
        eye: NP_3f8
    ) -> NP_44f8:
        model_basis = model_matrix__array[:3, :3]
        model_basis_normalized = model_basis / np.linalg.norm(model_basis, axis=0, keepdims=True)
        return (
            SpaceUtils.matrix_from_rotate(-Rotation.from_matrix(model_basis_normalized).as_rotvec())
            @ SpaceUtils.matrix_from_shift(-eye)
        )

    @Lazy.property()
    @staticmethod
    def _camera_uniform_block_buffer_(
        projection_matrix: NP_44f8,
        view_matrix: NP_44f8,
        #eye: NP_3f8,
        frame_radii: NP_2f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_camera",
            fields=[
                "mat4 u_projection_matrix",
                "mat4 u_view_matrix",
                #"vec3 u_view_position",
                "vec2 u_frame_radii"
            ],
            data={
                "u_projection_matrix": projection_matrix.T,
                "u_view_matrix": view_matrix.T,
                #"u_view_position": eye,
                "u_frame_radii": frame_radii
            }
        )
