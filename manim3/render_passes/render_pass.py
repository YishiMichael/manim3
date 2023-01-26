__all__ = ["RenderPass"]


from abc import abstractmethod

import moderngl


class RenderPass:
    @abstractmethod
    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> moderngl.Texture:
        # `PassesRenderProcedure` has already cleared the `target_framebuffer`,
        # so this function is not responsible for clearing it.
        pass
