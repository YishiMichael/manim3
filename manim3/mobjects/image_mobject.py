__all__ = ["ImageMobject"]


import moderngl
import numpy as np
from PIL import Image

from ..constants import PIXEL_PER_UNIT
from ..custom_typing import Real
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
#from ..utils.paint import Paint


class ImageMobject(MeshMobject):
    def __init__(
        self,
        image_path: str,
        #paint: Paint | None = None,
        *,
        width: Real | None = None,
        height: Real | None = 4.0,
        frame_scale: Real | None = None
    ):
        super().__init__()
        image: Image.Image = Image.open(image_path)
        self._image_ = image
        #self._paint_ = paint

        self._adjust_frame(
            image.width / PIXEL_PER_UNIT,
            image.height / PIXEL_PER_UNIT,
            width,
            height,
            frame_scale
        )
        self.scale(np.array((1.0, -1.0, 1.0)))  # flip y

    @lazy_property_initializer_writable
    @staticmethod
    def _image_() -> Image.Image:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _geometry_() -> PlaneGeometry:
        return PlaneGeometry()

    #@lazy_property_initializer_writable
    #@staticmethod
    #def _paint_() -> Paint | None:
    #    return None

    #@lazy_property_initializer_writable
    #@staticmethod
    #def _frame_() -> skia.Rect:
    #    return NotImplemented

    @lazy_property
    @staticmethod
    def _color_map_texture_(image: Image.Image) -> moderngl.Texture:
        return ContextSingleton().texture(
            size=image.size,
            components=len(image.getbands()),
            data=image.tobytes(),
        )
