from ...mobjects.mobject.mobject import Mobject
from .partial_base import PartialBase


class PartialCreate(PartialBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        backwards: bool = False
    ) -> None:

        def alpha_to_boundary_values(
            alpha: float
        ) -> tuple[float, float]:
            return (0.0, alpha)

        super().__init__(
            mobject=mobject,
            alpha_to_boundary_values=alpha_to_boundary_values,
            backwards=backwards
        )

    async def timeline(self) -> None:
        self.scene.add(self._mobject)
        await super().timeline()
