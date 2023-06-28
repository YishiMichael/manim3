from abc import abstractmethod

from ..lazy.lazy import Lazy
from ..rendering.framebuffer import OITFramebuffer
from .cameras.camera import Camera
from .cameras.perspective_camera import PerspectiveCamera
from .mobject import (
    Mobject,
    StyleMeta
)


class RenderableMobject(Mobject):
    __slots__ = ()

    @StyleMeta.register()
    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:
        return PerspectiveCamera()

    @abstractmethod
    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass
