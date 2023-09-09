from abc import abstractmethod

from ..lazy.lazy import Lazy
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..toplevel.toplevel import Toplevel
from .cameras.camera_attribute import CameraAttribute
from .mobject.mobject import Mobject


class RenderableMobject(Mobject):
    __slots__ = ()

    @Lazy.variable(freeze=False)
    @staticmethod
    def _camera_() -> CameraAttribute:
        return CameraAttribute(Toplevel.scene._camera)

    @abstractmethod
    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        pass
