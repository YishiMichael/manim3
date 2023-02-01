__all__ = ["ImageMobject"]


import moderngl
import numpy as np
from PIL import Image

from ..config import Config
from ..custom_typing import Real
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
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
        self._geometry_ = PlaneGeometry()

        self._adjust_frame(
            image.width / Config.pixel_per_unit,
            image.height / Config.pixel_per_unit,
            width,
            height,
            frame_scale
        )
        self.scale(np.array((1.0, -1.0, 1.0)))  # flip y

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        image = self._image
        with RenderProcedure.texture(size=image.size, components=len(image.getbands())) as color_texture:
            color_texture.write(image.tobytes())
            self._color_map_texture_ = color_texture
            super()._render(scene_config, target_framebuffer)
