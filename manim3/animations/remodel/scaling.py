from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.remodel_handlers.scale_remodel_handler import ScaleRemodelHandler
from ...mobjects.mobject.mobject import Mobject
from .remodel_infinite_base import RemodelInfiniteBase


class Scaling(RemodelInfiniteBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        factor: float | NP_3f8,
        about: About | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            remodel_handler=ScaleRemodelHandler(factor),
            about=about
        )
