#from dataclasses import dataclass
#from functools import reduce
import numpy as np
#import pyrr
import skia

from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..constants import PIXEL_PER_UNIT
#from ..constants import PIXEL_HEIGHT, PIXEL_WIDTH
from ..typing import *


__all__ = [
    #"BoundingBox2D",
    "SkiaMobject"
]


#@dataclass
#class BoundingBox2D:
#    origin: Vector2Type
#    radius: Vector2Type


class SkiaMobject(MeshMobject):
    def __init__(
        self: Self,
        #paints: list[skia.Paint] | None = None,
        frame: skia.Rect,
        resolution: tuple[int, int]
        #*,
        #width: Real | None = None,
        #height: Real | None = None
    ):
        #if frame is None:
        #    px_width, px_height = resolution
        #    aspect_ratio = px_width / px_height
        #    width, height = self.calculate_size_by_aspect_ratio(
        #        width, height, aspect_ratio
        #    )
        #    frame = BoundingBox2D(
        #        origin=np.array((0.0, 0.0)),
        #        radius=np.array((width, height)) / 2.0
        #    )
        #elif width is not None or height is not None:
        #    raise AttributeError("Cannot specify both parameters `frame` and `width` / `height`")

        super().__init__()
        self.enable_depth_test = False
        self.cull_face = "front_and_back"
        self.resolution: tuple[int, int] = resolution
        #self.frame: skia.Rect = frame

        self.scale(np.array((frame.width() / 2.0, frame.height() / 2.0, 1.0)))
        self.shift(np.array((frame.centerX(), -frame.centerY(), 0.0)))
        #frame_size = np.array((width, height))
        # TODO: remove the following comment
        """
        The `mode` attribute is set according to the arguments provided.

        If `resolution` is provided, `mode` is set to be `"resolution"`,
        and the coordinate system in `draw` method is the same as the canvas,
        i.e. the origin is at the top-left corner, with x-axis going rightwards
        and y-axis downwards.

        If `resolution` is not provided, `mode` is set to be `"frame"`,
        and the coordinate system in `draw` method is the same as the screen sapce,
        i.e. the origin is at the center, with x-axis going rightwards
        and y-axis upwards. Note that text-related drawing methods
        (i.e. `drawTextBlob`) will result in text drawings flipped upside down.
        """
        #if frame is None:
        #    if frame_size is None:
        #        if resolution is None:
        #            resolution_arr = np.array((FRAME_WIDTH, FRAME_HEIGHT))
        #        else:
        #            resolution_arr = np.array(resolution)
        #        frame_size = resolution_arr / PIXEL_PER_UNIT
        #    frame = BoundingBox2D(
        #        origin=np.array((0.0, 0.0)),
        #        radius=frame_size / 2.0
        #    )
        #elif frame_size is not None:
        #    raise AttributeError("Cannot specify both parameters `frame_size` and `frame`")

        #if resolution is None:
        #    resolution_arr = abs(2.0 * PIXEL_PER_UNIT * frame.radius)
        #    resolution = (int(resolution_arr[0]), int(resolution_arr[1]))
        #    mode = "frame"
        #else:
        #    mode = "resolution"

        #if frame is None:
        #    if resolution is None:
        #        resolution = (PIXEL_WIDTH, PIXEL_HEIGHT)
        #    frame = BoundingBox2D(
        #        origin=np.array((0.0, 0.0)),
        #        radius=np.array(resolution) / PIXEL_PER_UNIT / 2.0
        #    )
        #    mode = "resolution"
        #else:
        #    if resolution is None:
        #        arr = abs(2.0 * PIXEL_PER_UNIT * frame.radius)
        #        resolution = (int(arr[0]), int(arr[1]))
        #    mode = "frame"

        #self.paints: list[skia.Paint] = []
        #self.frame: BoundingBox2D = frame
        #self.resolution: tuple[int, int] = resolution
        #self.mode: str = mode
        #super().__init__()
        #self.scale(np.array((1.0, -1.0, 1.0)))  # flip y
        #self.enable_depth_test = False
        #self.cull_face = "front_and_back"

    #def init_matrix(self: Self) -> pyrr.Matrix44:
    #    return reduce(pyrr.Matrix44.__matmul__, (
    #        self.matrix_from_scale(np.array((*self.frame.radius, 1.0))),
    #        self.matrix_from_translation(np.array((*self.frame.origin, 0.0)))
    #    ))

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

    @staticmethod
    def calculate_resolution_by_frame(frame: skia.Rect) -> tuple[int, int]:
        return (
            int(frame.width() * PIXEL_PER_UNIT),
            int(frame.height() * PIXEL_PER_UNIT)
        )

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
            #canvas.scale(PIXEL_PER_UNIT, PIXEL_PER_UNIT)
            #matrix = self.get_canvas_matrix()
            #if matrix is not None:
            #    canvas.concat(skia.Matrix.MakeAll(*matrix.T.flatten()))
            self.draw(canvas)
        return array

    #@staticmethod
    #def rect_to_rect_matrix(rect_0: BoundingBox2D, rect_1: BoundingBox2D) -> pyrr.Matrix33:
    #    # `skia.Matrix.MakeRectToRect` function cannot handle flipping,
    #    # so implement a more flexible one here.
    #    def matrix_from_translation(vector: Vector2Type) -> pyrr.Matrix33:
    #        result = pyrr.Matrix33.identity()
    #        result[2, 0:2] = vector
    #        return result

    #    def matrix_from_scale(factor_vector: Vector2Type) -> pyrr.Matrix33:
    #        return pyrr.Matrix33(np.diagflat((*factor_vector, 1.0)))

    #    return reduce(pyrr.Matrix33.__matmul__, (
    #        matrix_from_translation(-rect_0.origin),
    #        matrix_from_scale(rect_1.radius / rect_0.radius),
    #        matrix_from_translation(rect_1.origin),
    #    ))

    #def get_canvas_matrix(self: Self) -> pyrr.Matrix33 | None:
    #    if self.mode == "resolution":
    #        return None

    #    width, height = self.resolution
    #    return self.rect_to_rect_matrix(
    #        self.frame,
    #        BoundingBox2D(
    #            origin=np.array((width, height)) / 2.0,
    #            radius=np.array((width, -height)) / 2.0  # flip y
    #        )
    #    )

    def draw(self: Self, canvas: skia.Canvas) -> None:
        pass
