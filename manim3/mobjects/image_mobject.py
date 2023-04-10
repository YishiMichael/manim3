__all__ = ["ImageMobject"]


#import moderngl
import numpy as np
from PIL import Image


from ..constants import X_AXIS
from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.config import ConfigSingleton
from ..rendering.framebuffer import (
    TransparentFramebuffer,
    OpaqueFramebuffer
)
from ..rendering.texture import TextureFactory
#from ..rendering.temporary_resource import ColorFramebufferBatch


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
            original_width=image.width / ConfigSingleton().size.pixel_per_unit,
            original_height=image.height / ConfigSingleton().size.pixel_per_unit,
            specified_frame_scale=frame_scale,
            specified_width=width,
            specified_height=height
        )
        self.scale(np.array((x_scale, y_scale, 1.0))).flip(X_AXIS)

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _geometry_(cls) -> Geometry:
        return PlaneGeometry()

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        image = self._image
        with TextureFactory.texture(size=image.size) as color_texture:
            color_texture.write(image.tobytes())
            self._color_map_ = color_texture
            super()._render(target_framebuffer)
