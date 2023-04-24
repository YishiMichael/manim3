from abc import abstractmethod

import moderngl

from ..lazy.core import LazyObject
from ..rendering.framebuffer import ColorFramebuffer


class RenderPass(LazyObject):
    __slots__ = ()

    @abstractmethod
    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: ColorFramebuffer
    ) -> moderngl.Texture:
        pass
