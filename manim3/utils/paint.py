__all__ = ["Paint"]


from colour import Color
import skia

from ..custom_typing import *


class Paint(skia.Paint):
    def __init__(self, **kwargs):
        super().__init__()
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)
        return self

    @property
    def anti_alias(self) -> bool:
        return self.getAntiAlias()

    @anti_alias.setter
    def anti_alias(self, anti_alias: bool):
        self.setAntiAlias(anti_alias)
        return self

    @property
    def blend_mode(self) -> skia.BlendMode:
        return self.getBlendMode()

    @blend_mode.setter
    def blend_mode(self, blend_mode: skia.BlendMode):
        self.setBlendMode(blend_mode)
        return self

    @property
    def color(self) -> Color:
        r, g, b, _ = tuple(self.getColor4f())
        return Color(rgb=(r, g, b))

    @color.setter
    def color(self, color: ColorType):
        if isinstance(color, str):
            color = Color(color)
        self.setColor4f(skia.Color4f(*color.rgb, self.opacity))
        return self

    @property
    def color_filter(self) -> skia.ColorFilter | None:
        return self.getColorFilter()

    @color_filter.setter
    def color_filter(self, color_filter: skia.ColorFilter | None):
        self.setcolor_filter(color_filter)
        return self

    @property
    def dither(self) -> bool:
        return self.getDither()

    @dither.setter
    def dither(self, dither: bool):
        self.setDither(dither)
        return self

    @property
    def filter_quality(self) -> skia.FilterQuality:
        return self.getFilterQuality()

    @filter_quality.setter
    def filter_quality(self, filter_quality: skia.FilterQuality):
        self.setFilterQuality(filter_quality)
        return self

    @property
    def image_filter(self) -> skia.ImageFilter | None:
        return self.getImageFilter()

    @image_filter.setter
    def image_filter(self, image_filter: skia.ImageFilter | None):
        self.setImageFilter(image_filter)
        return self

    @property
    def mask_filter(self) -> skia.MaskFilter | None:
        return self.getMaskFilter()

    @mask_filter.setter
    def mask_filter(self, mask_filter: skia.MaskFilter | None):
        self.setMaskFilter(mask_filter)
        return self

    @property
    def opacity(self) -> float:
        return self.getAlphaf()

    @opacity.setter
    def opacity(self, opacity: Real):
        self.setAlphaf(opacity)
        return self

    @property
    def path_effect(self) -> skia.PathEffect | None:
        return self.getPathEffect()

    @path_effect.setter
    def path_effect(self, path_effect: skia.PathEffect | None):
        self.setPathEffect(path_effect)
        return self

    @property
    def shader(self) -> skia.Shader:
        return self.getShader()

    @shader.setter
    def shader(self, shader: skia.Shader):
        self.setShader(shader)
        return self

    @property
    def stroke_cap(self) -> skia.Paint.Cap:
        return self.getStrokeCap()

    @stroke_cap.setter
    def stroke_cap(self, stroke_cap: skia.Paint.Cap):
        self.setStrokeCap(stroke_cap)
        return self

    @property
    def stroke_join(self) -> skia.Paint.Join:
        return self.getStrokeJoin()

    @stroke_join.setter
    def stroke_join(self, stroke_join: skia.Paint.Join):
        self.setStrokeJoin(stroke_join)
        return self

    @property
    def stroke_miter(self) -> float:
        return self.getStrokeMiter()

    @stroke_miter.setter
    def stroke_miter(self, stroke_miter: Real):
        self.setStrokeMiter(stroke_miter)
        return self

    @property
    def stroke_width(self) -> float:
        return self.getStrokeWidth()

    @stroke_width.setter
    def stroke_width(self, stroke_width: Real):
        self.setStrokeWidth(stroke_width)
        return self

    @property
    def style(self) -> skia.Paint.Style:
        return self.getStyle()

    @style.setter
    def style(self, style: skia.Paint.Style):
        self.setStyle(style)
        return self
