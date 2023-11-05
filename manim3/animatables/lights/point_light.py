from __future__ import annotations


from ...constants.constants import ORIGIN
from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable.animatable import AnimatableMeta
from ..arrays.animatable_color import AnimatableColor
from ..point import Point


class PointLight(Point):
    __slots__ = ()

    @AnimatableMeta.register_descriptor()
    @AnimatableMeta.register_converter(AnimatableColor)
    @Lazy.volatile()
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor()

    @Lazy.property()
    @staticmethod
    def _position_(
        model_matrix__array: NP_44f8
    ) -> NP_3f8:
        return SpaceUtils.apply(model_matrix__array, ORIGIN)
