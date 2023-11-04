from __future__ import annotations


from ...lazy.lazy import Lazy
#from ..mobject.mobject import Mobject
#from ..mobject.mobject_attributes.color_attribute import ColorAttribute
from ..animatable.animatable import AnimatableMeta
from ..arrays.animatable_color import AnimatableColor
from ..models.point import Point


class AmbientLight(Point):
    __slots__ = ()

    @AnimatableMeta.register_descriptor()
    @AnimatableMeta.register_converter(AnimatableColor)
    @Lazy.volatile()
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor()
