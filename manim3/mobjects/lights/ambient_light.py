import numpy as np

from ...constants.custom_typing import NP_3f8
from ...lazy.lazy import Lazy
from ...utils.space import SpaceUtils
from ..mobject.mobject import Mobject
from ..mobject.mobject_style_meta import MobjectStyleMeta


class AmbientLight(Mobject):
    __slots__ = ()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))
