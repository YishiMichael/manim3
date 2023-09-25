import numpy as np

from ...animatables.animatable import Animatable
from ...lazy.lazy import Lazy
#from ..mobject.mobject import Mobject
#from ..mobject.mobject_attributes.color_attribute import ColorAttribute
from ..arrays.animatable_color import AnimatableColor


class AmbientLight(Animatable):
    __slots__ = ()

    @Lazy.variable(freeze=False)
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor(np.ones((3,)))
