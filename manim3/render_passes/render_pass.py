__all__ = ["RenderPass"]


from abc import abstractmethod

import moderngl


class RenderPass:
    @abstractmethod
    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> moderngl.Texture:
        pass
