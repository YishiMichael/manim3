from __future__ import annotations

from ...lazy.lazy import Lazy
from ..animatable.animatable import AnimatableActions
from ..arrays.animatable_color import AnimatableColor
from ..model import ModelActions
from ..point import Point


class AmbientLight(Point):
    __slots__ = ()

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableColor)
    @Lazy.volatile()
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor()
