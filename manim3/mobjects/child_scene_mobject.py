#import numpy as np

#from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
#from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
#from ..scene.scene import Scene
#from .mesh_mobject import MeshMobject


#class ChildSceneMobject(MeshMobject):
#    __slots__ = (
#        "_scene",
#        "_color_framebuffer"
#    )

#    def __init__(
#        self,
#        scene: Scene
#    ) -> None:
#        super().__init__()
#        #color_texture = Context.texture(components=3)
#        color_framebuffer = ColorFramebuffer()
#        self._color_maps_ = [color_framebuffer.color_texture]
#        self._scene: Scene = scene
#        self._color_framebuffer: ColorFramebuffer = color_framebuffer
#        #self._color_framebuffer: ColorFramebuffer = ColorFramebuffer(
#        #    color_texture=color_texture
#        #)
#        self.scale(np.append(scene.camera._frame_radii_, 1.0))

#    def _render(
#        self,
#        target_framebuffer: OITFramebuffer
#    ) -> None:
#        self._scene._root_mobject._render_scene(self._color_framebuffer)
#        super()._render(target_framebuffer)
