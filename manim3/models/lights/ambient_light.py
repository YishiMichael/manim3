import numpy as np

from ...custom_typing import NP_3f8
from ...lazy.lazy import Lazy
from ...utils.space import SpaceUtils
from ..model import (
    Model,
    StyleMeta
)


class AmbientLight(Model):
    __slots__ = ()

    @StyleMeta.register(
        interpolate_method=SpaceUtils.lerp_3f8
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))
