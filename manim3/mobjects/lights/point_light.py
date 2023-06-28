import numpy as np

from ...constants import ORIGIN
from ...custom_typing import (
    NP_44f8,
    NP_3f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...utils.space import SpaceUtils
from ..mobject import (
    Mobject,
    StyleMeta
)


class PointLight(Mobject):
    __slots__ = ()

    @StyleMeta.register(
        interpolate_method=SpaceUtils.lerp_3f8
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(cls) -> NP_x3f8:
        return np.array((ORIGIN,))

    @Lazy.property_array
    @classmethod
    def _position_(
        cls,
        model_matrix: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)
