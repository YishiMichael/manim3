from ...constants.constants import ORIGIN
from ...constants.custom_typing import NP_3f8
from ...lazy.lazy import Lazy
#from ...utils.space_utils import SpaceUtils
#from ..mobject.mobject import Mobject
#from ..mobject.mobject_attributes.color_attribute import ColorAttribute
from ..arrays.animatable_color import AnimatableColor
from ..models.model import ModelMatrix
from ..models.point import Point


class PointLight(Point):
    __slots__ = ()

    @Lazy.variable(freeze=False)
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor()

    @Lazy.property(hasher=Lazy.array_hasher)
    @staticmethod
    def _position_(
        model_matrix: ModelMatrix
    ) -> NP_3f8:
        return model_matrix._apply_affine(ORIGIN)
