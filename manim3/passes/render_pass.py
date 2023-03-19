__all__ = ["RenderPass"]


from abc import abstractmethod

import moderngl

from ..lazy.core import LazyObject


class RenderPass(LazyObject):
    __slots__ = ()

    @abstractmethod
    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: moderngl.Framebuffer
    ) -> moderngl.Texture:
        pass
