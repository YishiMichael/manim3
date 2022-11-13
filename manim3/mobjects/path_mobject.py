from colour import Color
import numpy as np
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..constants import ORIGIN
from ..constants import PIXEL_PER_UNIT
from ..custom_typing import *


__all__ = [
    "PathMobject",
    "PathGroup"
]


class Paint(skia.Paint):
    def __init__(self: Self, **kwargs):
        super().__init__()
        self.set(**kwargs)

    def set(self: Self, **kwargs) -> Self:
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return self

    @property
    def anti_alias(self: Self) -> bool:
        return self.getAntiAlias()

    @anti_alias.setter
    def anti_alias(self: Self, anti_alias: bool) -> Self:
        self.setAntiAlias(anti_alias)
        return self

    @property
    def blend_mode(self: Self) -> skia.BlendMode:
        return self.getBlendMode()

    @blend_mode.setter
    def blend_mode(self: Self, blend_mode: skia.BlendMode) -> Self:
        self.setBlendMode(blend_mode)
        return self

    @property
    def color(self: Self) -> Color:
        r, g, b, _ = tuple(self.getColor4f())
        return Color(rgb=(r, g, b))

    @color.setter
    def color(self: Self, color: ColorType) -> Self:
        if isinstance(color, str):
            color = Color(color)
        self.setColor4f(skia.Color4f(*color.rgb, self.opacity))
        return self

    @property
    def color_filter(self: Self) -> skia.ColorFilter | None:
        return self.getColorFilter()

    @color_filter.setter
    def color_filter(self: Self, color_filter: skia.ColorFilter | None) -> Self:
        self.setcolor_filter(color_filter)
        return self

    @property
    def dither(self: Self) -> bool:
        return self.getDither()

    @dither.setter
    def dither(self: Self, dither: bool) -> Self:
        self.setDither(dither)
        return self

    @property
    def filter_quality(self: Self) -> skia.FilterQuality:
        return self.getFilterQuality()

    @filter_quality.setter
    def filter_quality(self: Self, filter_quality: skia.FilterQuality) -> Self:
        self.setFilterQuality(filter_quality)
        return self

    @property
    def image_filter(self: Self) -> skia.ImageFilter | None:
        return self.getImageFilter()

    @image_filter.setter
    def image_filter(self: Self, image_filter: skia.ImageFilter | None) -> Self:
        self.setImageFilter(image_filter)
        return self

    @property
    def mask_filter(self: Self) -> skia.MaskFilter | None:
        return self.getMaskFilter()

    @mask_filter.setter
    def mask_filter(self: Self, mask_filter: skia.MaskFilter | None) -> Self:
        self.setMaskFilter(mask_filter)
        return self

    @property
    def opacity(self: Self) -> float:
        return self.getAlphaf()

    @opacity.setter
    def opacity(self: Self, opacity: Real) -> Self:
        self.setAlphaf(opacity)
        return self

    @property
    def path_effect(self: Self) -> skia.PathEffect | None:
        return self.getPathEffect()

    @path_effect.setter
    def path_effect(self: Self, path_effect: skia.PathEffect | None) -> Self:
        self.setPathEffect(path_effect)
        return self

    @property
    def shader(self: Self) -> skia.Shader:
        return self.getShader()

    @shader.setter
    def shader(self: Self, shader: skia.Shader) -> Self:
        self.setShader(shader)
        return self

    @property
    def stroke_cap(self: Self) -> skia.Paint.Cap:
        return self.getStrokeCap()

    @stroke_cap.setter
    def stroke_cap(self: Self, stroke_cap: skia.Paint.Cap) -> Self:
        self.setStrokeCap(stroke_cap)
        return self

    @property
    def stroke_join(self: Self) -> skia.Paint.Join:
        return self.getStrokeJoin()

    @stroke_join.setter
    def stroke_join(self: Self, stroke_join: skia.Paint.Join) -> Self:
        self.setStrokeJoin(stroke_join)
        return self

    @property
    def stroke_miter(self: Self) -> float:
        return self.getStrokeMiter()

    @stroke_miter.setter
    def stroke_miter(self: Self, stroke_miter: Real) -> Self:
        self.setStrokeMiter(stroke_miter)
        return self

    @property
    def stroke_width(self: Self) -> float:
        return self.getStrokeWidth()

    @stroke_width.setter
    def stroke_width(self: Self, stroke_width: Real) -> Self:
        self.setStrokeWidth(stroke_width)
        return self

    @property
    def style(self: Self) -> skia.Paint.Style:
        return self.getStyle()

    @style.setter
    def style(self: Self, style: skia.Paint.Style) -> Self:
        self.setStyle(style)
        return self


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
            opacity=0.0
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
