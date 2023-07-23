import numpy as np

from ...constants.custom_typing import NP_3f8
from ...lazy.lazy import Lazy
from ..mobject.operation_handlers.mobject_operation import MobjectOperation
from ..mobject.operation_handlers.lerp_interpolate_handler import LerpInterpolateHandler
from ..mobject.mobject import Mobject


class AmbientLight(Mobject):
    __slots__ = ()

    @MobjectOperation.register(
        interpolate=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))
