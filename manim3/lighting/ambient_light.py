import numpy as np

from ..custom_typing import Vec3T
from ..lazy.lazy import Lazy
from ..mobjects.mobject import (
    Mobject,
    MobjectStyleMeta
)
from ..utils.space import SpaceUtils


class AmbientLight(Mobject):
    __slots__ = ()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_vec3
    )
    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_float
    )
    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0
