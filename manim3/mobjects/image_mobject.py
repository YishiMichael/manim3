__all__ = ["ImageMobject"]


import moderngl
import numpy as np
from PIL import Image

from ..constants import PIXEL_PER_UNIT
from ..custom_typing import Real
from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
#from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import lazy_property
from ..utils.render_procedure import RenderProcedure
from ..utils.scene_config import SceneConfig


class ImageMobject(MeshMobject):
    def __init__(
        self,
        image_path: str,
        *,
        width: Real | None = None,
        height: Real | None = 4.0,
        frame_scale: Real | None = None
    ):
        super().__init__()
        image: Image.Image = Image.open(image_path)
        self._image: Image.Image = image
        #self._color_map_texture_ = RenderProcedure.construct_texture(
        #    size=image.size,
        #    components=len(image.getbands()),
        #    data=image.tobytes()
        #)

        self._adjust_frame(
            image.width / PIXEL_PER_UNIT,
            image.height / PIXEL_PER_UNIT,
            width,
            height,
            frame_scale
        )
        self.scale(np.array((1.0, -1.0, 1.0)))  # flip y

    #@lazy_property_writable
    #@staticmethod
    #def _image_() -> Image.Image:
    #    return NotImplemented

    @lazy_property
    @staticmethod
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        #render_procedure = SceneMobjectRenderProcedure()
        #framebuffer = render_procedure._framebuffer_
        #framebuffer.clear()
        #self._scene._render_with_passes(self._scene._scene_config_, framebuffer)
        #self._color_map_texture_ = render_procedure._color_texture_
        # Prevent `target_framebuffer` from being polluted when rendering mobjects of the child scene.
        #target_framebuffer.clear()
        image = self._image
        with RenderProcedure.texture(size=image.size, components=len(image.getbands())) as color_texture:
            color_texture.write(image.tobytes())
            self._color_map_texture_ = color_texture
            super()._render(scene_config, target_framebuffer)

    #@lazy_property
    #@staticmethod
    #def _color_map_texture_(image: Image.Image) -> moderngl.Texture:
    #    return RenderProcedure.construct_texture(
    #        size=image.size,
    #        components=len(image.getbands()),
    #        data=image.tobytes()
    #    )
