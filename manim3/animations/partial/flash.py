import numpy as np

from ...mobjects.mobject.mobject import Mobject
from .partial_base import PartialBase


class Flash(PartialBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        flash_proportion: float = 1.0 / 16,
        backwards: bool = False
    ) -> None:

        def clip_proportion(
            alpha: float
        ) -> float:
            return float(np.clip(alpha, 0.0, 1.0))

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (
                clip_proportion(alpha * (1.0 + flash_proportion) - flash_proportion),
                clip_proportion(alpha * (1.0 + flash_proportion))
            )

        assert flash_proportion >= 0.0
        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards
        )

    async def timeline(self) -> None:
        self.scene.add(self._mobject)
        await super().timeline()
        self.scene.discard(self._mobject)
