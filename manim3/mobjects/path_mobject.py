from typing import Callable

from colour import Color
import numpy as np
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..utils.lazy import expire_properties, lazy_property
from ..utils.paint import Paint
from ..utils.path import Path
from ..constants import ORIGIN
from ..constants import PIXEL_PER_UNIT
from ..custom_typing import *


__all__ = [
    "PathMobject",
    "PathGroup"
]


class PathMobject(SkiaMobject):
    def __init__(
        self: Self,
        path: Path | None = None,
        frame_buff: tuple[Real, Real] = (0.25, 0.25),
        flip_y: bool = True
    ):
        #frame = path.computeTightBounds().makeOutset(*frame_buff)
        super().__init__(
            #frame=frame,
            #resolution=(
            #    int(frame.width() * PIXEL_PER_UNIT),
            #    int(frame.height() * PIXEL_PER_UNIT)
            #)
        )
        self.frame_buff: tuple[Real, Real] = frame_buff
        if flip_y:
            self.scale(np.array((1.0, -1.0, 1.0)), about_point=ORIGIN)

        #print(type(path))
        if path is None:
            path = Path()
        self.path: Path = path

        #self.fill_color: Color = Color("black")
        #self.fill_opacity: Real = 0.0
        #self.stroke_color: Color = Color("white")
        #self.stroke_opacity: Real = 1.0
        #self.stroke_width: Real = 0.05
        #self.draw_stroke_behind_fill: bool = False
        self.fill_paint: Paint | None = Paint(
            anti_alias=True,
            style=skia.Paint.kFill_Style,
            color=Color("black"),
            opacity=0.0,
            image_filter=skia.ImageFilters.Dilate(0.01, 0.01)
        )
        self.stroke_paint: Paint | None = Paint(
            anti_alias=True,
            style=skia.Paint.kStroke_Style,
            color=Color("white"),
            opacity=1.0,
            stroke_width=0.05
        )

    @lazy_property
    def _path(self: Self) -> Path:
        return self.path

    @_path.setter
    def _path(self: Self, arg: Path) -> None:
        pass

    @lazy_property
    def _fill_paint(self: Self) -> Paint | None:
        return self.fill_paint

    @_fill_paint.setter
    def _fill_paint(self: Self, arg: Paint | None) -> None:
        pass

    @lazy_property
    def _stroke_paint(self: Self) -> Paint | None:
        return self.stroke_paint

    @_stroke_paint.setter
    def _stroke_paint(self: Self, arg: Paint | None) -> None:
        pass

    @lazy_property
    def draw_stroke_behind_fill(self: Self) -> bool:
        return False

    @draw_stroke_behind_fill.setter
    def draw_stroke_behind_fill(self: Self, arg: bool) -> None:
        pass

    @lazy_property
    def frame(self: Self) -> skia.Rect:
        return self._path.skia_path.computeTightBounds().makeOutset(*self.frame_buff)
        ##self.scale(np.array((frame.width() / 2.0, frame.height() / 2.0, 1.0)))
        #self.stretch_to_fit_width(frame.width())
        #self.stretch_to_fit_height(frame.height())
        #self.shift(np.array((frame.centerX(), -frame.centerY(), 0.0)))
        #return self

    @lazy_property
    def resolution(self: Self) -> tuple[int, int]:
        frame = self.frame
        return (
            int(frame.width() * PIXEL_PER_UNIT),
            int(frame.height() * PIXEL_PER_UNIT)
        )

    @lazy_property
    def draw(self: Self) -> Callable[[skia.Canvas], None]:
        def wrapped(canvas: skia.Canvas) -> None:
            #if self.fill_opacity > 0.0:
            #    paints.append(skia.Paint(
            #        Style=skia.Paint.kFill_Style,
            #        Color=self.to_argb_int(self.fill_color, self.fill_opacity)
            #    ))
            #if self.stroke_width > 0.0 and self.stroke_opacity > 0.0:
            #    paints.append(skia.Paint(
            #        Style=skia.Paint.kStroke_Style,
            #        Color=self.to_argb_int(self.stroke_color, self.stroke_opacity),
            #        StrokeWidth=self.stroke_width
            #    ))
            paints = [self._fill_paint, self._stroke_paint]
            if self.draw_stroke_behind_fill:
                paints.reverse()

            canvas.concat(skia.Matrix.MakeRectToRect(
                self.frame, skia.Rect.MakeWH(*self.resolution), skia.Matrix.kFill_ScaleToFit
            ))
            for paint in paints:
                if paint is not None:
                    canvas.drawPath(self._path.skia_path, paint)
        return wrapped

    @expire_properties("_path")
    def set_path(self: Self, path: Path) -> Self:
        self.path = path
        return self

    @expire_properties("_fill_paint")
    def _set_fill(self: Self, **kwargs) -> Self:
        if self.fill_paint is None:
            self.fill_paint = Paint()
        self.fill_paint.set(**kwargs)

    def set_fill(
        self: Self,
        *,
        broadcast: bool = True,
        **kwargs
    ) -> Self:
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject._set_fill(**kwargs)
        return self

    @expire_properties("_stroke_paint")
    def _set_stroke(self: Self, **kwargs) -> Self:
        if self.stroke_paint is None:
            self.stroke_paint = Paint()
        self.stroke_paint.set(**kwargs)

    def set_stroke(
        self: Self,
        *,
        broadcast: bool = True,
        **kwargs
    ) -> Self:
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject._set_stroke(**kwargs)
        return self

    def set_paint(
        self: Self,
        *,
        draw_stroke_behind_fill: bool | None = None,
        fill_color: ColorType | None = None,
        fill_opacity: Real | None = None,
        stroke_color: ColorType | None = None,
        stroke_opacity: Real | None = None,
        broadcast: bool = True,
        **kwargs
    ) -> Self:
        for mobject in self.get_descendents(broadcast=broadcast):
            if draw_stroke_behind_fill is not None:
                mobject.draw_stroke_behind_fill = draw_stroke_behind_fill
            if fill_color is not None:
                mobject.set_fill(color=fill_color, broadcast=False)
            if fill_opacity is not None:
                mobject.set_fill(opacity=fill_opacity, broadcast=False)
            if stroke_color is not None:
                mobject.set_stroke(color=stroke_color, broadcast=False)
            if stroke_opacity is not None:
                mobject.set_stroke(opacity=stroke_opacity, broadcast=False)
            mobject.set_fill(**kwargs, broadcast=False)
            mobject.set_stroke(**kwargs, broadcast=False)
        return self

    #@staticmethod
    #def to_rgb_int(color: Color) -> int:
    #    return int(color.red * 255.0) << 16 \
    #        | int(color.green * 255.0) << 8 \
    #        | int(color.blue * 255.0)

    #@staticmethod
    #def to_argb_int(color: Color, opacity: Real) -> int:
    #    return int(opacity * 255.0) << 24 | PathMobject.to_rgb_int(color)


class PathGroup(PathMobject):
    def __init__(self: Self, *mobjects: PathMobject):
        super().__init__()
        self.add(*mobjects)

    # TODO
    def _bind_child(self: Self, node: Self, index: int | None = None) -> Self:
        assert isinstance(node, PathMobject)
        super()._bind_child(node, index=index)
