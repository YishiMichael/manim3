__all__ = ["ImageMobject"]


import moderngl
import numpy as np
from PIL import Image


from ..custom_typing import Real
from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.config import ConfigSingleton
from ..rendering.framebuffer_batches import ColorFramebufferBatch
from ..scenes.scene_config import SceneConfig
from ..utils.lazy import (
    NewData,
    lazy_basedata,
    lazy_slot
)


class ImageMobject(MeshMobject):
    __slots__ = ()

    def __new__(
        cls,
        image_path: str,
        *,
        width: Real | None = None,
        height: Real | None = 4.0,
        frame_scale: Real | None = None
    ):
        instance = super().__new__(cls)
        image = Image.open(image_path)
        instance._image = image

        instance.adjust_frame(
            image.width / ConfigSingleton().pixel_per_unit,
            image.height / ConfigSingleton().pixel_per_unit,
            width,
            height,
            frame_scale
        )
        instance.scale(np.array((1.0, -1.0, 1.0)))  # flip y
        return instance

    @lazy_basedata
    @staticmethod
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    @lazy_slot
    @staticmethod
    def _image() -> Image.Image:
        return NotImplemented

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        image = self._image
        with ColorFramebufferBatch() as batch:
            batch.color_texture.write(image.tobytes())
            self._color_map_texture_ = NewData(batch.color_texture)
            super()._render(scene_config, target_framebuffer)
