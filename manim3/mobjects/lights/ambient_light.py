import numpy as np

from ...lazy.lazy import Lazy
from ..mobject.mobject import Mobject
from ..mobject.mobject_attributes.color_attribute import ColorAttribute


class AmbientLight(Mobject):
    __slots__ = ()

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _color_() -> ColorAttribute:
        return ColorAttribute(np.ones((3,)))
