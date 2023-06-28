from ..constants import ORIGIN
from ..custom_typing import NP_3f8
from ..mobjects.mobject import Mobject
from ..utils.space import SpaceUtils
from .model import (
    AboutABC,
    Model
)


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
        model: Model
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
        model: Model
    ) -> NP_3f8:
        return model.get_bounding_box_point(self._edge)


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
        model: Model
    ) -> NP_3f8:
        assert isinstance(model, Mobject)
        return SpaceUtils.apply_affine(model._camera_._model_matrix_, self._border)
