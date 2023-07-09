from abc import abstractmethod

from ..animations.animation import Toplevel
from ..lazy.lazy import Lazy
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from .cameras.camera import Camera
from .mobject import Mobject
from .mobject_style_meta import MobjectStyleMeta


class RenderableMobject(Mobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        self._camera_ = Toplevel.get_scene()._camera

    @MobjectStyleMeta.register()
    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:
        return Camera()

    @abstractmethod
    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass
