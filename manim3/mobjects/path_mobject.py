from colour import Color
import moderngl
import numpy as np
import skia

from ..mobjects.skia_mobject import SkiaMobject
from ..utils.lazy import lazy_property, lazy_property_initializer
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
        self,
        path: Path | None = None,
        #frame_buff: tuple[Real, Real] = (0.25, 0.25),
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
        #self.frame_buff: tuple[Real, Real] = frame_buff
        if flip_y:
            self.scale(np.array((1.0, -1.0, 1.0)), about_point=ORIGIN)

        #print(type(path))
        if path is not None:
            self.set_path(path)

        #self.fill_color: Color = Color("black")
        #self.fill_opacity: Real = 0.0
        #self.stroke_color: Color = Color("white")
        #self.stroke_opacity: Real = 1.0
        #self.stroke_width: Real = 0.05
        #self.draw_stroke_behind_fill: bool = False
        #self.fill_paint: Paint | None = Paint(
        #    anti_alias=True,
        #    style=skia.Paint.kFill_Style,
        #    color=Color("black"),
        #    opacity=0.0,
        #    image_filter=skia.ImageFilters.Dilate(0.01, 0.01)
        #)
        #self.stroke_paint: Paint | None = Paint(
        #    anti_alias=True,
        #    style=skia.Paint.kStroke_Style,
        #    color=Color("white"),
        #    opacity=1.0,
        #    stroke_width=0.05
        #)

    @lazy_property_initializer
    def _path_() -> Path:
        return Path()

    #@_path.setter
    #def _path(self, arg: Path) -> None:
    #    pass

    @lazy_property_initializer
    def _fill_paint_() -> Paint | None:
        return Paint(
            anti_alias=True,
            style=skia.Paint.kFill_Style,
            color=Color("black"),
            opacity=0.0,
            image_filter=skia.ImageFilters.Dilate(0.01, 0.01)
        )

    #@_fill_paint.setter
    #def _fill_paint(self, arg: Paint | None) -> None:
    #    pass

    @lazy_property_initializer
    def _stroke_paint_() -> Paint | None:
        return Paint(
            anti_alias=True,
            style=skia.Paint.kStroke_Style,
            color=Color("white"),
            opacity=1.0,
            stroke_width=0.05
        )

    @lazy_property_initializer
    def _draw_stroke_behind_fill_() -> bool:
        return False

    @lazy_property_initializer
    def _frame_buff_() -> tuple[float, float]:
        return (0.25, 0.25)

    #@_stroke_paint.setter
    #def _stroke_paint(self, arg: Paint | None) -> None:
    #    pass

    #@lazy_property
    #def draw_stroke_behind_fill(self) -> bool:
    #    return False

    #@draw_stroke_behind_fill.setter
    #def draw_stroke_behind_fill(self, arg: bool) -> None:
    #    pass

    @lazy_property
    def _frame_(
        cls,
        path: Path,
        frame_buff: tuple[float, float]
    ) -> skia.Rect:
        return path._skia_path_.computeTightBounds().makeOutset(*frame_buff)
        ##self.scale(np.array((frame.width() / 2.0, frame.height() / 2.0, 1.0)))
        #self.stretch_to_fit_width(frame.width())
        #self.stretch_to_fit_height(frame.height())
        #self.shift(np.array((frame.centerX(), -frame.centerY(), 0.0)))
        #return self

    @lazy_property
    def _color_map_(
        cls,
        fill_paint: Paint | None,
        stroke_paint: Paint | None,
        draw_stroke_behind_fill: bool,
        frame: skia.Rect,
        path: Path
    ) -> moderngl.Texture:
        surface = cls._make_surface(
            int(frame.width() * PIXEL_PER_UNIT),
            int(frame.height() * PIXEL_PER_UNIT)
        )
        with surface as canvas:
            canvas.concat(skia.Matrix.MakeRectToRect(
                src=frame,
                dst=skia.Rect.Make(surface.imageInfo().bounds()),
                stf=skia.Matrix.kFill_ScaleToFit
            ))

            paints = [fill_paint, stroke_paint]
            if draw_stroke_behind_fill:
                paints.reverse()
            for paint in paints:
                if paint is not None:
                    canvas.drawPath(path=path._skia_path_, paint=paint)
        return cls._make_texture(surface.makeImageSnapshot())

    #@lazy_property
    #def _resolution_(frame: skia.Rect) -> tuple[int, int]:
    #    return (
    #        int(frame.width() * PIXEL_PER_UNIT),
    #        int(frame.height() * PIXEL_PER_UNIT)
    #    )

    #@lazy_property
    #def _draw_(
    #    fill_paint: Paint | None,
    #    stroke_paint: Paint | None,
    #    draw_stroke_behind_fill: bool,
    #    frame: skia.Rect,
    #    resolution: tuple[int, int],
    #    path: Path
    #) -> Callable[[skia.Canvas], None]:
    #    def draw(canvas: skia.Canvas) -> None:
    #        #if self.fill_opacity > 0.0:
    #        #    paints.append(skia.Paint(
    #        #        Style=skia.Paint.kFill_Style,
    #        #        Color=self.to_argb_int(self.fill_color, self.fill_opacity)
    #        #    ))
    #        #if self.stroke_width > 0.0 and self.stroke_opacity > 0.0:
    #        #    paints.append(skia.Paint(
    #        #        Style=skia.Paint.kStroke_Style,
    #        #        Color=self.to_argb_int(self.stroke_color, self.stroke_opacity),
    #        #        StrokeWidth=self.stroke_width
    #        #    ))
    #        paints = [fill_paint, stroke_paint]
    #        if draw_stroke_behind_fill:
    #            paints.reverse()

    #        canvas.concat(skia.Matrix.MakeRectToRect(
    #            frame, skia.Rect.MakeWH(*resolution), skia.Matrix.kFill_ScaleToFit
    #        ))
    #        for paint in paints:
    #            if paint is not None:
    #                canvas.drawPath(path._skia_path_, paint)
    #    return draw

    @_path_.updater
    def set_path(self, path: Path):
        self._path_ = path
        return self

    @_fill_paint_.updater
    def set_local_fill(self, disable: bool = False, **kwargs):
        if disable is False:
            if self._fill_paint_ is None:
                self._fill_paint_ = Paint()
            self._fill_paint_.set(**kwargs)
        else:
            self._fill_paint_ = None
        return self

    def set_fill(
        self,
        *,
        broadcast: bool = True,
        **kwargs
    ):
        for mobject in self.get_descendents(broadcast=broadcast):
            if not isinstance(mobject, PathMobject):
                continue
            mobject.set_local_fill(**kwargs)
        return self

    @_stroke_paint_.updater
    def set_local_stroke(self, disable: bool = False, **kwargs):
        if disable is False:
            if self._stroke_paint_ is None:
                self._stroke_paint_ = Paint()
            self._stroke_paint_.set(**kwargs)
        else:
            self._stroke_paint_ = None
        return self

    def set_stroke(
        self,
        *,
        broadcast: bool = True,
        **kwargs
    ):
        for mobject in self.get_descendents(broadcast=broadcast):
            if not isinstance(mobject, PathMobject):
                continue
            mobject.set_local_stroke(**kwargs)
        return self

    def set_paint(
        self,
        *,
        disable_fill: bool = False,
        disable_stroke: bool = False,
        draw_stroke_behind_fill: bool | None = None,
        fill_color: ColorType | None = None,
        fill_opacity: Real | None = None,
        stroke_color: ColorType | None = None,
        stroke_opacity: Real | None = None,
        broadcast: bool = True,
        **kwargs
    ):
        fill_kwargs = kwargs.copy()
        if disable_fill is not None:
            fill_kwargs["disable"] = disable_fill
        if fill_color is not None:
            fill_kwargs["color"] = fill_color
        if fill_color is not None:
            fill_kwargs["opacity"] = fill_opacity

        stroke_kwargs = kwargs.copy()
        if disable_stroke is not None:
            stroke_kwargs["disable"] = disable_stroke
        if stroke_color is not None:
            stroke_kwargs["color"] = stroke_color
        if stroke_color is not None:
            stroke_kwargs["opacity"] = stroke_opacity

        for mobject in self.get_descendents(broadcast=broadcast):
            if not isinstance(mobject, PathMobject):
                continue
            if draw_stroke_behind_fill is not None:
                self._draw_stroke_behind_fill_ = draw_stroke_behind_fill
            mobject.set_fill(**fill_kwargs, broadcast=False)
            mobject.set_stroke(**stroke_kwargs, broadcast=False)
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
    def __init__(self, *mobjects: PathMobject):
        super().__init__()
        self.add(*mobjects)

    # TODO
    def _bind_child(self, node, index: int | None = None):
        assert isinstance(node, PathMobject)
        super()._bind_child(node, index=index)
        return self
