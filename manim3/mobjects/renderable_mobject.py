from abc import abstractmethod

from ..lazy.lazy import Lazy
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..toplevel.toplevel import Toplevel
from .cameras.camera import Camera
from .mobject.mobject import Mobject
from .mobject.operation_handlers.mobject_operation import MobjectOperation


class RenderableMobject(Mobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        self._camera_ = Toplevel.scene._camera

    @MobjectOperation.register()
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
