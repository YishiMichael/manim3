import numpy as np

from ...animatables.models.model import Model
from ...lazy.lazy import Lazy
#from ..mobject.mobject import Mobject
from ..mobject.mobject_attributes.color_attribute import ColorAttribute


class AmbientLight(Model):
    __slots__ = ()

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _color_() -> ColorAttribute:
        return ColorAttribute(np.ones((3,)))
