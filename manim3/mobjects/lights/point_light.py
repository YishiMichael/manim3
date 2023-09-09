import numpy as np

from ...constants.constants import ORIGIN
from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..mobject.mobject import Mobject
from ..mobject.mobject_attributes.color_attribute import ColorAttribute


class PointLight(Mobject):
    __slots__ = ()

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _color_() -> ColorAttribute:
        return ColorAttribute(np.ones((3,)))

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _local_sample_positions_() -> NP_x3f8:
        return np.array((ORIGIN,))

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _position_(
        model_matrix__array: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply_affine(model_matrix__array, ORIGIN)
