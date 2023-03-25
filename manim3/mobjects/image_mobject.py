__all__ = ["ImageMobject"]


import moderngl
import numpy as np
from PIL import Image

from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.config import ConfigSingleton
from ..rendering.framebuffer_batch import ColorFramebufferBatch


class ImageMobject(MeshMobject):
    __slots__ = ("_image",)

    def __init__(
        self,
        image_path: str,
        *,
        frame_scale: float | None = None,
        width: float | None = None,
        height: float | None = 4.0
    ) -> None:
        super().__init__()
        image = Image.open(image_path)
        self._image: Image.Image = image

        x_scale, y_scale = self._get_frame_scale_vector(
            original_width=image.width / ConfigSingleton().pixel_per_unit,
            original_height=image.height / ConfigSingleton().pixel_per_unit,
            specified_frame_scale=frame_scale,
            specified_width=width,
            specified_height=height
        )
        self.scale(np.array((x_scale, -y_scale, 1.0)))  # flip y

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _geometry_(cls) -> Geometry:
        return PlaneGeometry()

    def _render(
        self,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        image = self._image
        with ColorFramebufferBatch() as batch:
            batch.color_texture.write(image.tobytes())
            self._color_map_ = batch.color_texture
            super()._render(target_framebuffer)
