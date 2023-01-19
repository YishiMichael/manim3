__all__ = ["RenderPass"]


from abc import abstractmethod
from typing import Generic, TypeVar

from ..mobjects.mobject import Mobject
from ..utils.renderable import (
    IntermediateFramebuffer,
    Framebuffer,
    Renderable
)


_MobjectT = TypeVar("_MobjectT", bound="Mobject")


class RenderPass(Generic[_MobjectT], Renderable):
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
