from ...constants import ORIGIN
from ...custom_typing import NP_3f8
from ...utils.space import SpaceUtils
from ..mobject import (
    AboutABC,
    Mobject
)
from ..renderable_mobject import RenderableMobject


class AboutPoint(AboutABC):
    __slots__ = ("_point",)

    def __init__(
        self,
        point: NP_3f8
    ) -> None:
        super().__init__()
        self._point: NP_3f8 = point

    def _get_about_point(
        self,
        mobject: Mobject
    ) -> NP_3f8:
        return self._point


class AboutEdge(AboutABC):
    __slots__ = ("_edge",)

    def __init__(
        self,
        edge: NP_3f8
    ) -> None:
        super().__init__()
        self._edge: NP_3f8 = edge

    def _get_about_point(
        self,
        mobject: Mobject
    ) -> NP_3f8:
        return mobject.get_bounding_box_point(self._edge)


class AboutCenter(AboutEdge):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            edge=ORIGIN
        )


class AboutBorder(AboutABC):
    __slots__ = ("_border",)

    def __init__(
        self,
        border: NP_3f8
    ) -> None:
        super().__init__()
        self._border: NP_3f8 = border

    def _get_about_point(
        self,
        mobject: Mobject
    ) -> NP_3f8:
        assert isinstance(mobject, RenderableMobject)
        return SpaceUtils.apply_affine(mobject._camera_._model_matrix_, self._border)
