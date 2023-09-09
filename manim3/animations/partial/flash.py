from ...mobjects.mobject.mobject import Mobject
from .partial_evenly import PartialEvenly


class Flash(PartialEvenly):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        *,
        proportion: float = 1.0 / 16,
        n_segments: int = 1,
        backwards: bool = False
    ) -> None:
        assert 0.0 <= proportion <= 1.0

        def alpha_to_boundaries(
            alpha: float
        ) -> tuple[float, float]:
            return (
                max(alpha * (1.0 + proportion) - proportion, 0.0),
                min(alpha * (1.0 + proportion), 1.0)
            )

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
        self.scene.discard(self._mobject)
