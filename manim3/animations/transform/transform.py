from ...mobjects.mobject.mobject import Mobject
from .transform_base import TransformBase


class Transform(TransformBase):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=start_mobject.copy()
        )

    async def timeline(self) -> None:
        self.scene.discard(self._start_mobject)
        self.scene.add(self._intermediate_mobject)
        await super().timeline()
        self.scene.discard(self._intermediate_mobject)
        self.scene.add(self._stop_mobject)
