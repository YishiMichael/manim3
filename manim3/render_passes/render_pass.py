__all__ = ["RenderPass"]


from abc import abstractmethod

from ..utils.renderable import (
    IntermediateFramebuffer,
    Framebuffer,
    RenderProcedure
)


class RenderPass(RenderProcedure):
    @abstractmethod
    def render(
        self,
        input_framebuffer: IntermediateFramebuffer,
        output_framebuffer: Framebuffer
    ):
        pass
