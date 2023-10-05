from __future__ import annotations


from typing import Self

import numpy as np
from PIL import Image

from ..toplevel.toplevel import Toplevel
from ..utils.space_utils import SpaceUtils
#from .mesh_mobjects.meshes.plane_mesh import PlaneMesh
from .mesh_mobjects.plane import Plane


class ImageMobject(Plane):
    __slots__ = ()

    def __init__(
        self: Self,
        image_path: str,
        *,
        width: float | None = None,
        height: float | None = 4.0,
        frame_scale: float | None = None
    ) -> None:
        super().__init__()
        image = Image.open(image_path).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image_texture = Toplevel.context.texture(size=image.size, components=3, dtype="f1")
        image_texture.write(image.tobytes("raw", "RGB"))
        self._color_maps_ = (image_texture,)

        pixel_per_unit = Toplevel.config.pixel_per_unit
        original_width = image.width / pixel_per_unit
        original_height = image.height / pixel_per_unit
        scale_x, scale_y = SpaceUtils.get_frame_scale_vector(
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
