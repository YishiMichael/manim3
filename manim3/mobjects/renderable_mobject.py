from abc import abstractmethod

from ..lazy.lazy import Lazy
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..toplevel.toplevel import Toplevel
from .cameras.camera import Camera
from .mobject.mobject import Mobject


class RenderableMobject(Mobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        self._camera_ = Toplevel.scene._camera

    @Lazy.variable(freeze=False)
    @staticmethod
    def _camera_() -> Camera:
        return Camera()

    @abstractmethod
    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass

    def bind_camera(
        self,
        camera: Camera
    ):
        self._camera_ = camera
        return self
