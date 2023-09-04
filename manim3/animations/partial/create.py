from ...mobjects.mobject.mobject import Mobject
from .partial_evenly import PartialEvenly


class Create(PartialEvenly):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        backwards: bool = False,
        n_segments: int = 1
    ) -> None:

        def alpha_to_boundaries(
            alpha: float
        ) -> tuple[float, float]:
            return (0.0, alpha)

        super().__init__(
            mobject=mobject,
            alpha_to_boundaries=alpha_to_boundaries,
            backwards=backwards,
            n_segments=n_segments,
            run_alpha=1.0
        )

    async def timeline(self) -> None:
        self.scene.add(self._mobject)
        await self.wait()

    #async def timeline(self) -> None:
    #    await self.wait()
