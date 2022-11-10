from dataclasses import dataclass
from functools import reduce
import numpy as np
import pyrr
import skia

from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..constants import PIXEL_PER_UNIT
#from ..constants import DEFAULT_PIXEL_HEIGHT, DEFAULT_PIXEL_WIDTH
from ..typing import *


__all__ = [
    "BoundingBox2D",
    "SkiaBaseMobject"
]


@dataclass
class BoundingBox2D:
    origin: Vector2Type
    radius: Vector2Type


class SkiaBaseMobject(MeshMobject):
    def __init__(
        self: Self,
        #draw_range: BoundingBox2D,
        resolution: tuple[int, int],
        frame: BoundingBox2D | None = None,
        #canvas_matrix: Matrix33Type,
        #geometry: Geometry | None = None,
        #rectangle: BoundingBox2D,
        #resolution: tuple[int, int] = (DEFAULT_PIXEL_WIDTH, DEFAULT_PIXEL_HEIGHT)
    ):
        if frame is None:
            frame = BoundingBox2D(
                origin=np.array((0.0, 0.0)),
                radius=np.array(resolution) / PIXEL_PER_UNIT / 2.0
            )
        self.frame: BoundingBox2D = frame
        self.resolution: tuple[int, int] = resolution
        super().__init__()
        #self.scale(np.array((1.0, -1.0, 1.0)))  # flip y
        self.enable_depth_test = False
        self.cull_face = "front_and_back"
        #super().__init__(rectangle)

    def init_matrix(self: Self) -> pyrr.Matrix44:
        return reduce(pyrr.Matrix44.__matmul__, (
            self.matrix_from_scale(np.array((*self.frame.radius, 1.0))),
            self.matrix_from_translation(np.array((*self.frame.origin, 0.0)))
        ))

    def init_geometry(self: Self) -> Geometry:
        return PlaneGeometry()

    def load_color_map(self: Self) -> TextureArrayType:
        #px_width = int(self.width * PIXEL_PER_UNIT)
        #px_height = int(self.height * PIXEL_PER_UNIT)
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
            #canvas.scale(PIXEL_PER_UNIT, PIXEL_PER_UNIT)
            matrix = self.get_canvas_matrix()
            if matrix is not None:
                canvas.concat(skia.Matrix.MakeAll(*matrix.T.flatten()))
            #canvas.concat(skia.Matrix.MakeRectToRect(
            #    #skia.Rect(-1.0, -1.0, 1.0, 1.0),
            #    canvas_rect,
            #    skia.Rect(0, 0, px_width, px_height),
            #    skia.Matrix.kFill_ScaleToFit
            #))
            self.draw(canvas)
        return array

    @staticmethod
    def rect_to_rect_matrix(rect_0: BoundingBox2D, rect_1: BoundingBox2D) -> pyrr.Matrix33:
        # `skia.Matrix.MakeRectToRect` function cannot handle flipping,
        # so implement a more flexible one here.
        def matrix_from_translation(vector: Vector2Type) -> pyrr.Matrix33:
            result = pyrr.Matrix33.identity()
            result[2, 0:2] = vector
            return result

        def matrix_from_scale(factor_vector: Vector2Type) -> pyrr.Matrix33:
            return pyrr.Matrix33(np.diagflat((*factor_vector, 1.0)))

        return reduce(pyrr.Matrix33.__matmul__, (
            matrix_from_translation(-rect_0.origin),
            matrix_from_scale(rect_1.radius / rect_0.radius),
            matrix_from_translation(rect_1.origin),
        ))

    def get_canvas_matrix(self: Self) -> pyrr.Matrix33 | None:
        return None

    def draw(self: Self, canvas: skia.Canvas) -> None:
        pass
