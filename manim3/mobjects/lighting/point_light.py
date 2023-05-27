import numpy as np

from ...constants import ORIGIN
from ...custom_typing import (
    NP_44f8,
    NP_3f8,
    NP_f8
)
from ...lazy.lazy import Lazy
from ...utils.space import SpaceUtils
from ..mobject import (
    Mobject,
    MobjectStyleMeta
)


class PointLight(Mobject):
    __slots__ = ()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_3f8
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return np.ones(())

    @Lazy.property_array
    @classmethod
    def _position_(
        cls,
        model_matrix: NP_44f8
    ) -> NP_3f8:
        # The position is initially set at the origin and tracked by `model_matrix`.
        # We can then control the light position via methods like `shift()`.
        # Light-moving animations are automatically applicable.
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)
