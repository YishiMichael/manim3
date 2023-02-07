__all__ = ["RenderPass"]


from abc import abstractmethod

import moderngl

from ..utils.lazy import LazyBase


class RenderPass(LazyBase):
    __slots__ = ()

    @abstractmethod
    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> moderngl.Texture:
        pass
