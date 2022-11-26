from abc import abstractmethod
from functools import reduce

import numpy as np
import pyrr
import skia

from ..geometries.frame_geometry import FrameGeometry
from ..geometries.geometry import Geometry
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

        self._enable_depth_test_ = False
        self._cull_face_ = "front_and_back"
        #self.frame: skia.Rect = frame
        #self.resolution: tuple[int, int] = resolution

        #self.scale(np.array((frame.width() / 2.0, frame.height() / 2.0, 1.0)))
        #self.shift(np.array((frame.centerX(), -frame.centerY(), 0.0)))

    @lazy_property
    def _geometry_(cls, frame: skia.Rect) -> Geometry:
        return FrameGeometry(frame)

    @lazy_property_initializer
    @abstractmethod
    def _frame_() -> skia.Rect:
        raise NotImplementedError

    #@lazy_property_initializer
    #def _prev_frame_() -> skia.Rect:
    #    # Rect induced from FrameGeometry
    #    return skia.Rect(l=-1.0, t=1.0, r=1.0, b=-1.0)

    #@_prev_frame.setter
    #def _prev_frame(self, arg: pyrr.Matrix44) -> None:
    #    pass

    #@lazy_property
    #def _color_map_(
    #    resolution: tuple[int, int],
    #    draw: Callable[[skia.Canvas], None]
    #) -> skia.Image:

    #    px_width, px_height = resolution
    #    #array = np.zeros((px_height, px_width, 4), dtype=np.uint8)

    #    # TODO: try using GPU rendering?
    #    #context = skia.GrDirectContext.MakeGL()
    #    #context.resetContext()
    #    #print(type(context))
    #    #surface = skia.Surface.MakeRenderTarget(
    #    #    context=skia_context,
    #    #    budgeted=skia.Budgeted.kYes,
    #    #    imageInfo=skia.ImageInfo.MakeN32Premul(width=px_width, height=px_height)
    #    #)

    #    #with skia.Surface(
    #    #    array=array,
    #    #    colorType=skia.kRGBA_8888_ColorType,
    #    #    alphaType=skia.kUnpremul_AlphaType
    #    #) as canvas:
    #    with surface as canvas:
    #        draw(canvas)

    #    #info = surface.imageInfo()
    #    #row_bytes = px_width * info.bytesPerPixel()
    #    #buffer = bytearray(row_bytes * px_height)
    #    #pixmap = skia.Pixmap(info=info, data=buffer, rowBytes=row_bytes)
    #    #surface.readPixels(pixmap)

    #    #surface.flushAndSubmit()
    #    #print(len(bytes(pixmap).strip(b"\x00")))
    #    #print(surface.readPixels(pixmap))
    #    #print(pixmap.info().bytesPerPixel())
    #    #print(pixmap.width())
    #    #print(pixmap.height())
    #    #print(pixmap.rowBytes())

    #    #image = skia.Image.MakeFromRaster(pixmap)
    #    #assert image is not None
    #    #image.save('skia_output.png', skia.kPNG)
    #    return surface.makeImageSnapshot()

    #@_prev_frame_.updater
    #def render(self) -> None:
    #    # Calculate the matrix inverse could be expensive, so use the explicit formula
    #    #prev_frame = self._prev_frame_
    #    frame = self._frame_
    #    #print(
    #    #    prev_frame.centerX(),
    #    #    prev_frame.centerY(),
    #    #    prev_frame.width(),
    #    #    prev_frame.height()
    #    #)
    #    #print(
    #    #    frame.centerX(),
    #    #    frame.centerY(),
    #    #    frame.width(),
    #    #    frame.height()
    #    #)
    #    #prev_frame_matrix_inv = reduce(pyrr.Matrix44.__matmul__, (
    #    #    self.matrix_from_translation(np.array((-prev_frame.centerX(), prev_frame.centerY(), 0.0))),
    #    #    self.matrix_from_scale(np.array((2.0 / prev_frame.width(), -2.0 / prev_frame.height(), 1.0)))
    #    #))
    #    #print(prev_frame_matrix_inv)
    #    frame_matrix = reduce(pyrr.Matrix44.__matmul__, (
    #        self.matrix_from_scale(np.array((frame.width() / 2.0, -frame.height() / 2.0, 1.0))),
    #        self.matrix_from_translation(np.array((frame.centerX(), -frame.centerY(), 0.0)))
    #    ))
    #    #frame_matrix = self.matrix_from_translation(np.array((frame.centerX(), -frame.centerY(), 0.0)))
    #    #frame_matrix_inv = reduce(pyrr.Matrix44.__matmul__, (
    #    #    self.matrix_from_translation(np.array((-frame.centerX(), frame.centerY(), 0.0))),
    #    #    self.matrix_from_scale(np.array((2.0 / frame.width(), -2.0 / frame.height(), 1.0)))
    #    #))
    #    #print(frame_matrix)
    #    #print()
    #    #self.preapply_raw_matrix(
    #    #    prev_frame_matrix_inv,
    #    #    broadcast=False
    #    #)
    #    self.preapply_raw_matrix(
    #        frame_matrix,
    #        broadcast=False
    #    )
    #    #self._prev_frame_ = frame
    #    super().render()
    #    #self.preapply_raw_matrix(
    #    #    frame_matrix_inv,
    #    #    broadcast=False
    #    #)

    @classmethod
    def _calculate_frame(
        cls,
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

    @classmethod
    def _make_surface(cls, px_width: int, px_height: int) -> skia.Surface:
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
        return surface

    #@lazy_property_initializer
    #@abstractmethod
    #def _resolution_() -> tuple[int, int]:
    #    raise NotImplementedError

    #@lazy_property_initializer
    #@abstractmethod
    #def _draw_() -> Callable[[skia.Canvas], None]:
    #    raise NotImplementedError
