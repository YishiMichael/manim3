from ....constants.custom_typing import NP_3f8
from ....utils.space import SpaceUtils
from ...renderable_mobject import RenderableMobject
from ..mobject import Mobject
from .about import About


class AboutBorder(About):
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
