__all__ = ["LightPass"]


import moderngl

from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.mobject import SceneConfig
from ..render_passes.render_pass import RenderPass


class LightPass(RenderPass[MeshMobject]):
    def _render(
        self,
        input_texture: moderngl.Texture,
        input_depth_texture: moderngl.Texture,
        output_framebuffer: moderngl.Framebuffer,
        mobject: MeshMobject,
        scene_config: SceneConfig
    ):
        pass  # TODO
