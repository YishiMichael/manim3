__all__ = ["RenderPass"]


from abc import abstractmethod

from ..utils.renderable import (
    IntermediateFramebuffer,
    Framebuffer,
    Renderable
)


class RenderPass(Renderable):
    @abstractmethod
    def _render(
        self,
        input_framebuffer: IntermediateFramebuffer,
        output_framebuffer: Framebuffer
    ):
        pass

    #@classmethod
    #def _render_by_routine(cls, render_routine: list[RenderStep]) -> None:
    #    for render_step in render_routine:
    #        cls._render_by_step(render_step)
