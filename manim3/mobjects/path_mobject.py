from colour import Color
import numpy as np
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..utils.paint import Paint
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
        path: skia.Path,
        frame_buff: tuple[Real, Real] = (0.25, 0.25),
        flip_y: bool = True
    ):
        frame = path.computeTightBounds().makeOutset(*frame_buff)
        super().__init__(
            frame=frame,
            resolution=(
                int(frame.width() * PIXEL_PER_UNIT),
                int(frame.height() * PIXEL_PER_UNIT)
            )
        )
        if flip_y:
            self.scale(np.array((1.0, -1.0, 1.0)), about_point=ORIGIN)

        self.path: skia.Path = path

        #self.fill_color: Color = Color("black")
        #self.fill_opacity: Real = 0.0
        #self.stroke_color: Color = Color("white")
        #self.stroke_opacity: Real = 1.0
        #self.stroke_width: Real = 0.05
        self.fill_paint: Paint = Paint(
            anti_alias=True,
            style=skia.Paint.kFill_Style,
            color=Color("black"),
            opacity=0.0,
            image_filter=skia.ImageFilters.Dilate(0.01, 0.01)
        )
        self.stroke_paint: Paint = Paint(
            anti_alias=True,
            style=skia.Paint.kStroke_Style,
            color=Color("white"),
            opacity=1.0,
            stroke_width=0.05
        )
        self.draw_stroke_behind_fill: bool = False

    def draw(self: Self, canvas: skia.Canvas) -> None:
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
        paints = [self.fill_paint, self.stroke_paint]
        if self.draw_stroke_behind_fill:
            paints.reverse()

        canvas.concat(skia.Matrix.MakeRectToRect(
            self.frame, skia.Rect.MakeWH(*self.resolution), skia.Matrix.kFill_ScaleToFit
        ))
        for paint in paints:
            canvas.drawPath(self.path, paint)

    def set_fill(
        self: Self,
        *,
        broadcast: bool = True,
        **kwargs
    ) -> Self:
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject.fill_paint.set(**kwargs)
        return self

    def set_stroke(
        self: Self,
        *,
        broadcast: bool = True,
        **kwargs
    ) -> Self:
        for mobject in self.get_descendents(broadcast=broadcast):
            mobject.stroke_paint.set(**kwargs)
        return self

    def set_paint(
        self: Self,
        *,
        fill_color: ColorType | None = None,
        fill_opacity: Real | None = None,
        stroke_color: ColorType | None = None,
        stroke_opacity: Real | None = None,
        draw_stroke_behind_fill: bool | None = None,
        broadcast: bool = True,
        **kwargs
    ) -> Self:
        for mobject in self.get_descendents(broadcast=broadcast):
            if fill_color is not None:
                mobject.fill_paint.color = fill_color
            if fill_opacity is not None:
                mobject.fill_paint.opacity = fill_opacity
            if stroke_color is not None:
                mobject.stroke_paint.color = stroke_color
            if stroke_opacity is not None:
                mobject.stroke_paint.opacity = stroke_opacity
            if draw_stroke_behind_fill is not None:
                mobject.draw_stroke_behind_fill = draw_stroke_behind_fill
            mobject.set_fill(**kwargs)
            mobject.set_stroke(**kwargs)
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
        super().__init__(skia.Path())
        assert all(
            isinstance(mobject, PathMobject)
            for mobject in mobjects
        )
        self.add(*mobjects)
