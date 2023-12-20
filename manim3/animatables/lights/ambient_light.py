from __future__ import annotations


from ...lazy.lazy import Lazy
from ...toplevel.toplevel import Toplevel
from ..animatable.animatable import Animatable
from ..arrays.animatable_color import AnimatableColor
from ..model import Model
from ..point import Point


class AmbientLight(Point):
    __slots__ = ()

    @Animatable.interpolate.register_descriptor()
    @Model.set.register_descriptor(converter=AnimatableColor)
    @Lazy.volatile()
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor(Toplevel._get_config().default_color)
