import numpy as np

from ..rendering.framebuffer import (
    ColorFramebuffer,
    OITFramebuffer
)
from ..scene.scene import Scene
from .textured_mobject import TexturedMobject


class ChildSceneMobject(TexturedMobject):
    __slots__ = (
        "_scene",
        "_framebuffer"
    )

    def __init__(
        self,
        scene: Scene
    ) -> None:
        super().__init__()
        self._scene: Scene = scene
        self._framebuffer: ColorFramebuffer = ColorFramebuffer(
            color_texture=self._color_map_
        )
        self.scale(np.append(scene.camera._frame_radii_, 1.0))

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        self._scene._root_mobject._render_scene(self._framebuffer)
        super()._render(target_framebuffer)
