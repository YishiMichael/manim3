import numpy as np

from ..animations.animation import Scene
#from ..geometries.plane_geometry import PlaneGeometry
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
        #self._geometry_ = PlaneGeometry()
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
            self._color_map_ = color_texture
            super()._render(target_framebuffer)
