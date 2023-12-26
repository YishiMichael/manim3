from __future__ import annotations


from typing import Self

import numpy as np
from PIL import Image

from ..toplevel.toplevel import Toplevel
from .mesh_mobjects.plane import Plane


class ImageMobject(Plane):
    __slots__ = ()

    def __init__(
        self: Self,
        image_filename: str,
        *,
        width: float | None = None,
        height: float | None = None,
        scale: float | None = None
    ) -> None:
        super().__init__()
        for image_dir in Toplevel._get_config().image_search_dirs:
            if (image_path := image_dir.joinpath(image_filename)).exists():
                break
        else:
            raise FileNotFoundError(image_filename)

        image = Image.open(image_path).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        image_texture = Toplevel._get_context().texture(size=image.size, components=3, samples=0, dtype="f1")
        image_texture.write(image.tobytes("raw", "RGB"))
        self._color_maps_ = (image_texture,)

        pixel_per_unit = Toplevel._get_config().pixel_per_unit
        original_width = image.width / pixel_per_unit
        original_height = image.height / pixel_per_unit
        scale_x, scale_y = type(self)._get_scale_vector(
            original_width=original_width,
            original_height=original_height,
            specified_width=width,
            specified_height=height,
            specified_scale=scale
        )
        self.scale(np.array((
            scale_x * original_width / 2.0,
            scale_y * original_height / 2.0,
            1.0
        )))

    @classmethod
    def _get_scale_vector(
        cls: type[Self],
        *,
        original_width: float,
        original_height: float,
        specified_width: float | None,
        specified_height: float | None,
        specified_scale: float | None
    ) -> tuple[float, float]:
        match specified_width, specified_height:
            case float(), float():
                return specified_width / original_width, specified_height / original_height
            case float(), None:
                scale = specified_width / original_width
            case None, float():
                scale = specified_height / original_height
            case _:
                scale = specified_scale if specified_scale is not None else 1.0
        return scale, scale
