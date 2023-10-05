from __future__ import annotations


from ...lazy.lazy import Lazy
#from ..mobject.mobject import Mobject
#from ..mobject.mobject_attributes.color_attribute import ColorAttribute
from ..animatable.animatable import Animatable
from ..arrays.animatable_color import AnimatableColor


class AmbientLight(Animatable):
    __slots__ = ()

    @Lazy.variable(freeze=False)
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor()
