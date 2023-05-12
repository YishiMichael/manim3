from ..constants import ORIGIN
from ..custom_typing import (
    Mat4T,
    Vec3T
)
from ..lazy.lazy import Lazy
from ..mobjects.mobject import Mobject
from ..utils.space import SpaceUtils


class PointLight(Mobject):
    __slots__ = ()

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
