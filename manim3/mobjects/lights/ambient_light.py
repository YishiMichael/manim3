import numpy as np

from ...constants.custom_typing import NP_3f8
from ...lazy.lazy import Lazy
from ..mobject.operation_handlers.lerp_interpolate_handler import LerpInterpolateHandler
from ..mobject.mobject import Mobject
from ..mobject.style_meta import StyleMeta


class AmbientLight(Mobject):
    __slots__ = ()

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))
