__all__ = ["SceneMobject"]


import moderngl
import numpy as np

from ..constants import (
    FRAME_HEIGHT,
    FRAME_WIDTH
)
from ..custom_typing import Real
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..scenes.child_scene import ChildScene
from ..utils.render_procedure import RenderProcedure
from ..utils.scene_config import SceneConfig


class SceneMobject(MeshMobject):
    def __init__(
        self,
        scene: ChildScene
    ):
        super().__init__()
        self._scene: ChildScene = scene
        self._geometry_ = PlaneGeometry()
        self.stretch_to_fit_size(np.array((FRAME_WIDTH, FRAME_HEIGHT, 0.0)))

    def _update_dt(self, dt: Real):
        super()._update_dt(dt)
        self._scene._update_dt(dt)

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        with RenderProcedure.texture() as color_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[color_texture],
                    depth_attachment=None
                ) as scene_framebuffer:
            self._scene._render_with_passes(self._scene._scene_config_, scene_framebuffer)
            self._color_map_texture_ = color_texture
            super()._render(scene_config, target_framebuffer)
