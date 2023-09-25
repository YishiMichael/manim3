import numpy as np


from ...constants.constants import ORIGIN
from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8
    #NP_x3f8
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
#from ..mobject.mobject import Mobject
#from ..mobject.mobject_attributes.color_attribute import ColorAttribute
from ..arrays.animatable_color import AnimatableColor
from ..models.point import Point


class PointLight(Point):
    __slots__ = ()

    @Lazy.variable(freeze=False)
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor(np.ones((3,)))

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _position_(
        model_matrix: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply_affine(model_matrix, ORIGIN)
