import numpy as np

from ...custom_typing import (
    NP_3f8,
    NP_f8
)
from ...lazy.lazy import Lazy
from ...utils.space import SpaceUtils
from ..mobject import (
    Mobject,
    MobjectStyleMeta
)


class AmbientLight(Mobject):
    __slots__ = ()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_3f8
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return np.ones(())
