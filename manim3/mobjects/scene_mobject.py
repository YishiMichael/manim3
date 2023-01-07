__all__ = ["SceneMobject"]


import moderngl
import numpy as np

from ..constants import (
    FRAME_HEIGHT,
    FRAME_WIDTH
)
from ..custom_typing import Real
from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.scene import Scene
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import IntermediateFramebuffer


class SceneMobject(MeshMobject):
    def __init__(
        self,
        scene: Scene
    ):
        super().__init__()
        assert scene._window is None  # TODO
        self._scene_ = scene
        self.stretch_to_fit_size(np.array((FRAME_WIDTH, FRAME_HEIGHT, 0.0)))

    @lazy_property_initializer_writable
    @staticmethod
    def _scene_() -> Scene:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    @lazy_property
    @staticmethod
    def _color_map_texture_(scene: Scene) -> moderngl.Texture | None:
        scene._render_scene()
        framebuffer = scene._framebuffer
        assert isinstance(framebuffer, IntermediateFramebuffer)
        return framebuffer.get_attachment(0)

    @_scene_.updater
    def _update_dt(self, dt: Real):
        super()._update_dt(dt)
        self._scene_._update_dt(dt)
