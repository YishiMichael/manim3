__all__ = ["SceneMobject"]


import moderngl
import numpy as np


from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.config import ConfigSingleton
from ..rendering.framebuffer_batches import ColorFramebufferBatch
from ..scenes.scene import Scene
from ..scenes.scene_config import SceneConfig
from ..utils.lazy import (
    NewData,
    lazy_basedata,
    lazy_slot
)


class SceneMobject(MeshMobject):
    __slots__ = ()

    def __new__(
        cls,
        scene_cls: type[Scene]
    ):
        instance = super().__new__(cls)
        instance._scene = scene_cls()
        instance.stretch_to_fit_size(np.array((*ConfigSingleton().frame_size, 0.0)))
        return instance

    @lazy_basedata
    @staticmethod
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    @lazy_slot
    @staticmethod
    def _scene() -> Scene:
        return NotImplemented

    #def _update_dt(self, dt: Real):
    #    super()._update_dt(dt)  # TODO
    #    self._scene._update_dt(dt)

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        with ColorFramebufferBatch() as batch:
            self._scene._render_with_passes(self._scene._scene_config, batch.framebuffer)
            self._color_map_ = NewData(batch.color_texture)
            super()._render(scene_config, target_framebuffer)
