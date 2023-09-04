from ...mobjects.mobject.mobject import Mobject
from .partial_evenly import PartialEvenly


class Dashed(PartialEvenly):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        proportion: float = 1.0 / 2,
        n_segments: int = 16,
        backwards: bool = False
    ) -> None:
        assert 0.0 <= proportion <= 1.0

        def alpha_to_boundaries(
            alpha: float
        ) -> tuple[float, float]:
            return (
                alpha,
                alpha + proportion
            )

        super().__init__(
            mobject=mobject,
            alpha_to_boundaries=alpha_to_boundaries,
            backwards=backwards,
            n_segments=n_segments
        )

    async def timeline(self) -> None:
        await self.wait_forever()
