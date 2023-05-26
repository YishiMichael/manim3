import numpy as np

from ..animations.animation import Scene
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.framebuffer import (
    ColorFramebuffer,
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.texture import TextureFactory


class ChildSceneMobject(MeshMobject):
    __slots__ = ("_scene",)

    def __init__(
        self,
        scene: Scene
    ) -> None:
        super().__init__()
        self._scene: Scene = scene
        self._enable_phong_lighting_ = False
        self.scale(np.append(scene.camera._frame_radii_, 1.0))

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        with TextureFactory.texture() as color_texture:
            framebuffer = ColorFramebuffer(
                color_texture=color_texture
            )
            self._scene._scene_frame._render_scene(framebuffer)

            # Clear alpha with 1.0.
            # The frame is always opaque, regardless of the background color of `self._scene`.
            framebuffer.framebuffer.color_mask = (False, False, False, True)
            framebuffer.framebuffer.clear(alpha=1.0)
            framebuffer.framebuffer.color_mask = (True, True, True, True)

            self._color_map_ = color_texture
            super()._render(target_framebuffer)
