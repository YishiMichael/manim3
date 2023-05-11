import numpy as np

from ..config import ConfigSingleton
from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..lazy.lazy import Lazy
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.texture import TextureFactory
from ..scene.scene import Scene


class ChildSceneMobject(MeshMobject):
    __slots__ = ("_scene",)

    def __init__(
        self,
        scene: Scene
    ) -> None:
        super().__init__()
        self._scene: Scene = scene
        self.scale(np.array((
            ConfigSingleton().size.frame_width,
            ConfigSingleton().size.frame_height,
            1.0
        )))

    @Lazy.variable
    @classmethod
    def _geometry_(cls) -> Geometry:
        return PlaneGeometry()

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        scene = self._scene
        with TextureFactory.texture() as color_texture:
            scene._scene_frame._render_to_texture(color_texture)
            self._color_map_ = color_texture
            super()._render(target_framebuffer)
