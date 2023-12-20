from __future__ import annotations


from ...constants.constants import ORIGIN
from ...constants.custom_typing import (
    NP_3f8,
    NP_44f8
)
from ...lazy.lazy import Lazy
from ...toplevel.toplevel import Toplevel
from ..animatable.animatable import Animatable
from ..arrays.animatable_color import AnimatableColor
from ..arrays.model_matrix import ModelMatrix
from ..model import Model
from ..point import Point


class PointLight(Point):
    __slots__ = ()

    @Animatable.interpolate.register_descriptor()
    @Model.set.register_descriptor(converter=AnimatableColor)
    @Lazy.volatile()
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor(Toplevel._get_config().default_color)

    @Lazy.property()
    @staticmethod
    def _position_(
        model_matrix__array: NP_44f8
    ) -> NP_3f8:
        return ModelMatrix._apply(model_matrix__array, ORIGIN)
