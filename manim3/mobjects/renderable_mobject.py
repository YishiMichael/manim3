from abc import abstractmethod

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..lazy.lazy import Lazy
from ..mobjects.mobject import (
    Mobject,
    MobjectMeta
)
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)


class RenderableMobject(Mobject):
    __slots__ = ()

    @MobjectMeta.register(
        interpolate_method=NotImplemented
    )
    @Lazy.variable_shared
    @classmethod
    def _is_transparent_(cls) -> bool:
        return False

    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:  # Keep updated with `Scene._camera`.
        return PerspectiveCamera()

    @abstractmethod
    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        pass

    @property
    def is_transparent(self) -> bool:
        return self._is_transparent_.value
