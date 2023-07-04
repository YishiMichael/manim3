from abc import abstractmethod

from ..lazy.lazy import Lazy
from ..rendering.framebuffer import OITFramebuffer
from .cameras.camera import Camera
from .cameras.perspective_camera import PerspectiveCamera
from .mobject import Mobject
from .mobject_style_meta import MobjectStyleMeta


class RenderableMobject(Mobject):
    __slots__ = ()

    @MobjectStyleMeta.register()
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
