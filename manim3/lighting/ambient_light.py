import numpy as np

from ..custom_typing import Vec3T
from ..lazy.lazy import Lazy
from ..mobjects.mobject import (
    Mobject,
    MobjectMeta
)
from ..utils.space import SpaceUtils


class AmbientLight(Mobject):
    __slots__ = ()

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_vec3
    )
    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((Mobject._is_transparent_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @property
    def color(self) -> Vec3T:
        return self._color_.value

    @property
    def opacity(self) -> float:
        return self._opacity_.value
