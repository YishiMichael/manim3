from ...constants.custom_typing import NP_3f8
from ...mobjects.mobject.abouts.about import About
from ...mobjects.mobject.remodel_handlers.rotate_remodel_handler import RotateRemodelHandler
from ...mobjects.mobject.mobject import Mobject
from .remodel_finite_base import RemodelFiniteBase


class Rotating(RemodelFiniteBase):
    __slots__ = ()

    def __init__(
        self,
        mobject: Mobject,
        rotvec: NP_3f8,
        about: About | None = None
    ) -> None:
        super().__init__(
            mobject=mobject,
            remodel_handler=RotateRemodelHandler(rotvec),
            about=about
        )
