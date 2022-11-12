import numpy as np
import skia

from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..typing import *


__all__ = ["SkiaMobject"]


class SkiaMobject(MeshMobject):
    def __init__(
        self: Self,
        frame: skia.Rect,
        resolution: tuple[int, int]
    ):
        super().__init__()
        self.enable_depth_test = False
        self.cull_face = "front_and_back"
        self.frame: skia.Rect = frame
        self.resolution: tuple[int, int] = resolution

        self.scale(np.array((frame.width() / 2.0, frame.height() / 2.0, 1.0)))
        self.shift(np.array((frame.centerX(), -frame.centerY(), 0.0)))

    @staticmethod
    def calculate_frame_by_aspect_ratio(
        width: Real | None,
        height: Real | None,
        aspect_ratio: Real
    ) -> skia.Rect:
        if width is None:
            if height is None:
                height = 4.0
            width = height * aspect_ratio
        elif height is None:
            height = width / aspect_ratio
        rx = width / 2.0
        ry = height / 2.0
        return skia.Rect(-rx, -ry, rx, ry)

    def init_geometry(self: Self) -> Geometry:
        return PlaneGeometry()

    def load_color_map(self: Self) -> TextureArrayType:
        px_width, px_height = self.resolution
        array = np.zeros((px_height, px_width, 4), dtype=np.uint8)

        # According to the documentation at `https://kyamagu.github.io/skia-python/tutorial`,
        # the default value of parameter `colorType` should be `skia.kRGBA_8888_ColorType`,
        # but it strangely defaults to `skia.kBGRA_8888_ColorType` in practice.
        # Passing in the parameter explicitly fixes this issue for now.
        with skia.Surface(
            array=array,
            colorType=skia.kRGBA_8888_ColorType,
            alphaType=skia.kUnpremul_AlphaType
        ) as canvas:
            self.draw(canvas)
        return array

    def draw(self: Self, canvas: skia.Canvas) -> None:
        pass
