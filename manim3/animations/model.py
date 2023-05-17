from scipy.spatial.transform import Rotation

from manim3.custom_typing import Vec3T

from ..animations.animation import Animation
from ..constants import (
    ORIGIN,
    OUT
)
from ..mobjects.mobject import Mobject
from ..utils.space import SpaceUtils


class Rotating(Animation):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        # The direction of `axis` defines the rotation axis; the length of `axis` defines the angular speed.
        axis: Vec3T = OUT,
        about_point: Vec3T = ORIGIN
    ) -> None:
        initial_model_matrix = mobject._model_matrix_.value

        def updater(
            alpha: float
        ) -> None:
            mobject._model_matrix_ = mobject.get_relative_transform_matrix(
                matrix=SpaceUtils.matrix_from_rotation(Rotation.from_rotvec(axis * alpha)),
                about_point=about_point
            ) @ initial_model_matrix

        super().__init__(
            updater=updater
        )
