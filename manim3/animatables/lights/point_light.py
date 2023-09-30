from ...constants.constants import ORIGIN
from ...constants.custom_typing import NP_3f8
from ...lazy.lazy import Lazy
from ..arrays.animatable_color import AnimatableColor
from ..arrays.model_matrix import AffineApplier
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
        model_matrix__applier: AffineApplier
    ) -> NP_3f8:
        return model_matrix__applier.apply(ORIGIN)
