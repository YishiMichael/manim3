import numpy as np

from ..animations.animation import Scene
from ..rendering.framebuffer import (
    ColorFramebuffer,
    OITFramebuffer
)
from .textured_mobject import TexturedMobject


class ChildSceneMobject(TexturedMobject):
    __slots__ = ("_scene",)

    def __init__(
        self,
        scene: Scene
    ) -> None:
        super().__init__()
        self._scene: Scene = scene
        self.scale(np.append(scene.camera._frame_radii_, 1.0))

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        assert (color_map := self._color_map_) is not None
        framebuffer = ColorFramebuffer(
            color_texture=color_map
        )
        self._scene._root_mobject._render_scene(framebuffer)
        super()._render(target_framebuffer)
