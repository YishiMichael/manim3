from abc import abstractmethod
from functools import reduce
from typing import Callable

import numpy as np
import pyrr
import skia

from ..geometries.geometry import Geometry
from ..geometries.plane_geometry import PlaneGeometry
from ..mobjects.mesh_mobject import MeshMobject
from ..utils.lazy import lazy_property, lazy_property_initializer
from ..custom_typing import *


__all__ = ["SkiaMobject"]


class SkiaMobject(MeshMobject):
    def __init__(
        self
        #frame: skia.Rect,
        #resolution: tuple[int, int]
    ):
        super().__init__()
        #self.canvas_matrix: pyrr.Matrix44 = pyrr.Matrix44.identity()

        self._enable_depth_test_ = False
        self._cull_face_ = "front_and_back"
        #self.frame: skia.Rect = frame
        #self.resolution: tuple[int, int] = resolution

        #self.scale(np.array((frame.width() / 2.0, frame.height() / 2.0, 1.0)))
        #self.shift(np.array((frame.centerX(), -frame.centerY(), 0.0)))

    @lazy_property_initializer
    def _geometry_() -> Geometry:
        return PlaneGeometry()

    @lazy_property_initializer
    def _canvas_matrix_() -> pyrr.Matrix44:
        return pyrr.Matrix44.identity()

    #@_canvas_matrix.setter
    #def _canvas_matrix(self, arg: pyrr.Matrix44) -> None:
    #    pass

    @lazy_property
    def _color_map_(
        resolution: tuple[int, int],
        draw: Callable[[skia.Canvas], None]
    ) -> skia.Image:

        px_width, px_height = resolution
        #array = np.zeros((px_height, px_width, 4), dtype=np.uint8)

        # TODO: try using GPU rendering?
        #context = skia.GrDirectContext.MakeGL()
        #context.resetContext()
        #print(type(context))
        #surface = skia.Surface.MakeRenderTarget(
        #    context=skia_context,
        #    budgeted=skia.Budgeted.kYes,
        #    imageInfo=skia.ImageInfo.MakeN32Premul(width=px_width, height=px_height)
        #)

        # According to the documentation at `https://kyamagu.github.io/skia-python/tutorial`,
        # the default value of parameter `colorType` should be `skia.kRGBA_8888_ColorType`,
        # but it strangely defaults to `skia.kBGRA_8888_ColorType` in practice.
        # Passing in the parameter explicitly fixes this issue for now.
        surface = skia.Surface.MakeRaster(imageInfo=skia.ImageInfo.Make(
            width=px_width,
            height=px_height,
            ct=skia.kRGBA_8888_ColorType,
            at=skia.kUnpremul_AlphaType
        ))
        assert surface is not None

        #with skia.Surface(
        #    array=array,
        #    colorType=skia.kRGBA_8888_ColorType,
        #    alphaType=skia.kUnpremul_AlphaType
        #) as canvas:
        with surface as canvas:
            draw(canvas)

        #info = surface.imageInfo()
        #row_bytes = px_width * info.bytesPerPixel()
        #buffer = bytearray(row_bytes * px_height)
        #pixmap = skia.Pixmap(info=info, data=buffer, rowBytes=row_bytes)
        #surface.readPixels(pixmap)

        #surface.flushAndSubmit()
        #print(len(bytes(pixmap).strip(b"\x00")))
        #print(surface.readPixels(pixmap))
        #print(pixmap.info().bytesPerPixel())
        #print(pixmap.width())
        #print(pixmap.height())
        #print(pixmap.rowBytes())

        #image = skia.Image.MakeFromRaster(pixmap)
        #assert image is not None
        #image.save('skia_output.png', skia.kPNG)
        return surface.makeImageSnapshot()

    def render(self) -> None:
        frame = self._frame_
        new_canvas_matrix = reduce(pyrr.Matrix44.__matmul__, (
            self.matrix_from_scale(np.array((frame.width() / 2.0, -frame.height() / 2.0, 1.0))),
            self.matrix_from_translation(np.array((frame.centerX(), -frame.centerY(), 0.0)))
        ))
        self.preapply_raw_matrix(
            ~self._canvas_matrix_,
            broadcast=False
        )
        self.preapply_raw_matrix(
            new_canvas_matrix,
            broadcast=False
        )
        self._canvas_matrix_ = new_canvas_matrix
        super().render()

    @staticmethod
    def calculate_frame(
        original_width: Real,
        original_height: Real,
        specified_width: Real | None,
        specified_height: Real | None,
        specified_frame_scale: Real | None
    ) -> skia.Rect:
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
        #return width, height
        #    if specified_height is not None:

        #        height = 4.0
        #    width = height * aspect_ratio
        #elif height is None:
        #    height = width / aspect_ratio
        rx = width / 2.0
        ry = height / 2.0
        return skia.Rect(l=-rx, t=-ry, r=rx, b=ry)

    @lazy_property_initializer
    @abstractmethod
    def _frame_() -> skia.Rect:
        raise NotImplementedError

    @lazy_property_initializer
    @abstractmethod
    def _resolution_() -> tuple[int, int]:
        raise NotImplementedError

    @lazy_property_initializer
    @abstractmethod
    def _draw_() -> Callable[[skia.Canvas], None]:
        raise NotImplementedError
