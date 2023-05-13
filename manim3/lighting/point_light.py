import numpy as np

from ..constants import ORIGIN
from ..custom_typing import (
    Mat4T,
    Vec3T
)
from ..lazy.lazy import Lazy
from ..mobjects.mobject import (
    Mobject,
    MobjectMeta
)
from ..utils.space import SpaceUtils


class PointLight(Mobject):
    __slots__ = ()

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_vec3
    )
    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((Mobject._is_transparent_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @Lazy.property_external
    @classmethod
    def _position_(
        cls,
        model_matrix: Mat4T
    ) -> Vec3T:
        # The position is initially set at the origin and tracked by `model_matrix`.
        # We can then control the light position via methods like `shift()`.
        # Light-moving animations are automatically applicable.
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)

    @property
    def color(self) -> Vec3T:
        return self._color_.value

    @property
    def opacity(self) -> float:
        return self._opacity_.value
