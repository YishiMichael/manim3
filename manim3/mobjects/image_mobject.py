import numpy as np
from PIL import Image

from ..config import Config
from ..utils.space import SpaceUtils
from .textured_mobject import TexturedMobject


class ImageMobject(TexturedMobject):
    __slots__ = ()

    def __init__(
        self,
        image_path: str,
        *,
        width: float | None = None,
        height: float | None = 4.0,
        frame_scale: float | None = None
    ) -> None:
        image = Image.open(image_path).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        super().__init__(size=image.size)
        self._color_map_.write(image.tobytes("raw", "RGB"))

        pixel_per_unit = Config().size.pixel_per_unit
        original_width = image.width / pixel_per_unit
        original_height = image.height / pixel_per_unit
        scale_x, scale_y = SpaceUtils._get_frame_scale_vector(
            original_width=original_width,
            original_height=original_height,
            specified_width=width,
            specified_height=height,
            specified_frame_scale=frame_scale
        )
        self.scale(np.array((
            scale_x * original_width / 2.0,
            scale_y * original_height / 2.0,
            1.0
        )))
