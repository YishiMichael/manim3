from ...mobjects.mobject.mobject import Mobject
from .transform_base import TransformBase


class TransformTo(TransformBase):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: Mobject,
        stop_mobject: Mobject
    ) -> None:
        super().__init__(
            start_mobject=start_mobject,
            stop_mobject=stop_mobject,
            intermediate_mobject=start_mobject
        )
