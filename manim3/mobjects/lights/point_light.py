import numpy as np

from ...constants.constants import ORIGIN
from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..mobject.operation_handlers.lerp_interpolate_handler import LerpInterpolateHandler
from ..mobject.mobject import Mobject
from ..mobject.style_meta import StyleMeta


class PointLight(Mobject):
    __slots__ = ()

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @Lazy.property_array
    @classmethod
    def _local_sample_positions_(cls) -> NP_x3f8:
        return np.array((ORIGIN,))

    @Lazy.property_array
    @classmethod
    def _position_(
        cls,
        model_matrix: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)
