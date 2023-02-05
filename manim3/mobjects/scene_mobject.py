__all__ = ["SceneMobject"]


import moderngl
import numpy as np

from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.config import ConfigSingleton
from ..rendering.render_procedure import RenderProcedure
from ..scenes.scene import Scene
from ..scenes.scene_config import SceneConfig
from ..utils.lazy import (
    LazyData,
    lazy_basedata
)


class SceneMobject(MeshMobject):
    def __new__(
        cls,
        scene_cls: type[Scene]
    ):
        instance = super().__new__(cls)
        instance._scene_ = LazyData(scene_cls())
        instance.stretch_to_fit_size(np.array((*ConfigSingleton().frame_size, 0.0)))
        return instance

    @lazy_basedata
    @staticmethod
    def _scene_() -> Scene:
        return NotImplemented

    @lazy_basedata
    @staticmethod
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    #def _update_dt(self, dt: Real):
    #    super()._update_dt(dt)  # TODO
    #    self._scene._update_dt(dt)

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        with RenderProcedure.texture() as color_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[color_texture],
                    depth_attachment=None
                ) as scene_framebuffer:
            self._scene_._render_with_passes(self._scene_._scene_config, scene_framebuffer)
            self._color_map_texture_ = LazyData(color_texture)
            super()._render(scene_config, target_framebuffer)
