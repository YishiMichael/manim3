from colour import Color
import skia

from ..custom_typing import *


__all__ = ["Paint"]


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
