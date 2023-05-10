from abc import abstractmethod

import moderngl

from ..rendering.framebuffer import ColorFramebuffer
from ..utils.lazy import LazyObject


class RenderPass(LazyObject):
    __slots__ = ()

    @abstractmethod
    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: ColorFramebuffer
    ) -> moderngl.Texture:
        pass
