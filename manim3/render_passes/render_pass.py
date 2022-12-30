__all__ = ["RenderPass"]


from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import moderngl

from ..mobjects.mobject import Mobject
from ..scene import Scene
from ..utils.renderable import RenderStep, Renderable


_MobjectT = TypeVar("_MobjectT", bound="Mobject")


class RenderPass(Generic[_MobjectT], ABC, Renderable):
    @abstractmethod
    def _render(
        self,
        input_texture: moderngl.Texture,
        input_depth_texture: moderngl.Texture,
        output_framebuffer: moderngl.Framebuffer,
        mobject: _MobjectT,
        scene: Scene,
    ):
        pass

    @classmethod
    def _render_by_routine(cls, render_routine: list[RenderStep]) -> None:
        for render_step in render_routine:
            cls._render_by_step(render_step)
