import numpy as np
from PIL import Image

from ..config import Config
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.texture import TextureFactory
from ..utils.space import SpaceUtils
from .mesh_mobject import MeshMobject


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

        pixel_per_unit = Config().size.pixel_per_unit
        scale_x, scale_y = SpaceUtils._get_frame_scale_vector(
            original_width=image.width / pixel_per_unit,
            original_height=image.height / pixel_per_unit,
            specified_width=width,
            specified_height=height,
            specified_frame_scale=frame_scale
        )
        self.scale(np.array((
            scale_x / 2.0,
            -scale_y / 2.0,  # Flip y.
            1.0
        )))

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        image = self._image
        with TextureFactory.texture(size=image.size) as color_texture:
            color_texture.write(image.tobytes())
            self._color_maps_ = [color_texture]
            super()._render(target_framebuffer)
