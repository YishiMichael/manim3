import numpy as np
from PIL import Image

from ..constants import X_AXIS
from ..config import ConfigSingleton
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.texture import TextureFactory


class ImageMobject(MeshMobject):
    __slots__ = ("_image",)

    def __init__(
        self,
        image_path: str,
        *,
        width: float | None = None,
        height: float | None = 4.0,
        frame_scale: float | None = None
    ) -> None:
        super().__init__()
        image = Image.open(image_path)
        self._image: Image.Image = image

        self._geometry_ = PlaneGeometry()
        pixel_per_unit = ConfigSingleton().size.pixel_per_unit
        x_scale, y_scale = self._get_frame_scale_vector(
            original_width=image.width / pixel_per_unit,
            original_height=image.height / pixel_per_unit,
            specified_width=width,
            specified_height=height,
            specified_frame_scale=frame_scale
        )
        self.scale(np.array((x_scale, y_scale, 1.0))).flip(X_AXIS)

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        image = self._image
        with TextureFactory.texture(size=image.size) as color_texture:
            color_texture.write(image.tobytes())
            self._color_map_ = color_texture
            super()._render(target_framebuffer)
