__all__ = ["SceneMobject"]


import moderngl
import numpy as np

from ..custom_typing import Real
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.config import ConfigSingleton
from ..rendering.render_procedure import RenderProcedure
from ..scenes.scene import Scene
from ..scenes.scene_config import SceneConfig


class SceneMobject(MeshMobject):
    def __init__(
        self,
        scene_cls: type[Scene]
    ):
        super().__init__()
        self._scene: Scene = scene_cls()
        self._geometry_ = PlaneGeometry()
        self.stretch_to_fit_size(np.array((*ConfigSingleton().frame_size, 0.0)))

    def _update_dt(self, dt: Real):
        super()._update_dt(dt)
        self._scene._update_dt(dt)

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        with RenderProcedure.texture() as color_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[color_texture],
                    depth_attachment=None
                ) as scene_framebuffer:
            self._scene._render_with_passes(self._scene._scene_config, scene_framebuffer)
            self._color_map_texture_ = color_texture
            super()._render(scene_config, target_framebuffer)
