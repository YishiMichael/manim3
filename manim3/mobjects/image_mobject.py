__all__ = ["ImageMobject"]


import moderngl
import numpy as np
from PIL import Image

from ..constants import PIXEL_PER_UNIT
from ..custom_typing import Real
from ..geometries.geometry import Geometry
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

        self._stretch_by_size(
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
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    def _stretch_by_size(
        self,
        original_width: Real,
        original_height: Real,
        specified_width: Real | None,
        specified_height: Real | None,
        specified_frame_scale: Real | None
    ):
        if specified_width is None and specified_height is None:
            width = original_width
            height = original_height
            if specified_frame_scale is not None:
                width *= specified_frame_scale
                height *= specified_frame_scale
        elif specified_width is not None and specified_height is None:
            width = specified_width
            height = specified_width / original_width * original_height
        elif specified_width is None and specified_height is not None:
            width = specified_height / original_height * original_width
            height = specified_height
        elif specified_width is not None and specified_height is not None:
            width = specified_width
            height = specified_height
        else:
            raise  # never
        self.stretch_to_fit_size(np.array((width, height, 0.0)))
        return self

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
    def _color_map_texture_(image: Image.Image) -> moderngl.Texture | None:
        return ContextSingleton().texture(
            size=image.size,
            components=len(image.getbands()),
            data=image.tobytes(),
        )
