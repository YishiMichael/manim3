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
from ..mobjects.scene import (
    ChildScene,
    Scene
)
from ..utils.lazy import (
    lazy_property,
    lazy_property_writable
)
from ..utils.renderable import RenderProcedure


class SceneMobject(MeshMobject):
    def __init__(
        self,
        scene: ChildScene
    ):
        super().__init__()
        assert not isinstance(scene, Scene)  # TODO
        self._scene_ = scene
        self.stretch_to_fit_size(np.array((FRAME_WIDTH, FRAME_HEIGHT, 0.0)))

    @lazy_property_writable
    @staticmethod
    def _scene_() -> ChildScene:
        return NotImplemented

    @lazy_property_writable  # writable?
    @staticmethod
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    @lazy_property
    @staticmethod
    def _color_map_texture_(scene: ChildScene) -> moderngl.Texture:
        render_procedure = SceneMobjectRenderProcedure()
        scene._render_with_passes(scene._scene_config_, render_procedure._framebuffer_)
        #framebuffer = scene._framebuffer
        #assert isinstance(framebuffer, IntermediateFramebuffer)
        return render_procedure._color_texture_

    @_scene_.updater
    def _update_dt(self, dt: Real):
        super()._update_dt(dt)
        self._scene_._update_dt(dt)


class SceneMobjectRenderProcedure(RenderProcedure):
    @lazy_property
    @staticmethod
    def _color_texture_() -> moderngl.Texture:
        return RenderProcedure.construct_texture()

    @lazy_property
    @staticmethod
    def _framebuffer_(
        color_texture: moderngl.Texture
    ) -> moderngl.Framebuffer:
        return RenderProcedure.construct_framebuffer(
            color_attachments=[color_texture],
            depth_attachment=None
        )

    #def render(
    #    self,
    #    scene_mobject: SceneMobject,
    #    scene_config: SceneConfig,
    #    target_framebuffer: moderngl.Framebuffer
    #) -> None:
    #    scene_mobject._scene._render_with_passes(scene_mobject._scene._scene_config_, self._framebuffer_)
    #    self.render_by_step(self.render_step(
    #        shader_str=self.read_shader("mesh"),
    #        custom_macros=[],
    #        texture_storages=[
    #            scene_mobject._u_color_maps_.write(
    #                np.array(self._color_texture_)
    #            )
    #        ],
    #        uniform_blocks=[
    #            scene_config._camera_._ub_camera_,
    #            scene_mobject._ub_model_,
    #            scene_config._ub_lights_
    #        ],
    #        attributes=scene_mobject._attributes_,
    #        index_buffer=scene_mobject._index_buffer_,
    #        framebuffer=target_framebuffer,
    #        enable_only=scene_mobject._enable_only_,
    #        context_state=self.context_state(),
    #        mode=moderngl.TRIANGLES
    #    ))
