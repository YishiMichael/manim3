__all__ = ["PathMobject"]


from colour import Color
import moderngl
import numpy as np
import skia

from ..constants import (
    ORIGIN,
    PIXEL_PER_UNIT
)
from ..custom_typing import (
    ColorType,
    Real
)
from ..mobjects.skia_mobject import SkiaMobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.paint import Paint
from ..utils.path import Path


class PathMobject(SkiaMobject):
    def __init__(
        self,
        path: Path | None = None,
        flip_y: bool = True
    ):
        super().__init__()
        if flip_y:
            self.scale(np.array((1.0, -1.0, 1.0)), about_point=ORIGIN)
        if path is not None:
            self.set_path(path)

    @lazy_property_initializer_writable
    @staticmethod
    def _path_() -> Path:
        return Path()

    @lazy_property_initializer_writable
    @staticmethod
    def _disable_fill_() -> bool:
        return False

    @lazy_property_initializer
    @staticmethod
    def _fill_paint_() -> Paint:
        return Paint(
            anti_alias=True,
            style=skia.Paint.kFill_Style,
            color=Color("black"),
            opacity=0.0,
            image_filter=skia.ImageFilters.Dilate(0.01, 0.01)
        )

    @lazy_property_initializer_writable
    @staticmethod
    def _disable_stroke_() -> bool:
        return False

    @lazy_property_initializer
    @staticmethod
    def _stroke_paint_() -> Paint:
        return Paint(
            anti_alias=True,
            style=skia.Paint.kStroke_Style,
            color=Color("white"),
            opacity=1.0,
            stroke_width=0.05
        )

    @lazy_property_initializer_writable
    @staticmethod
    def _draw_stroke_behind_fill_() -> bool:
        return False

    @lazy_property_initializer_writable
    @staticmethod
    def _frame_buff_() -> tuple[float, float]:
        return (0.25, 0.25)

    @lazy_property
    @staticmethod
    def _frame_(
        path: Path,
        frame_buff: tuple[float, float]
    ) -> skia.Rect:
        return path._skia_path_.computeTightBounds().makeOutset(*frame_buff)

    @lazy_property
    @staticmethod
    def _paints_(
        fill_paint: Paint,
        disable_fill: bool,
        stroke_paint: Paint,
        disable_stroke: bool,
        draw_stroke_behind_fill: bool,
    ) -> list[Paint]:
        paints = [(fill_paint, disable_fill), (stroke_paint, disable_stroke)]
        if draw_stroke_behind_fill:
            paints.reverse()
        return [paint for paint, disable in paints if not disable]

    @lazy_property
    @staticmethod
    def _color_map_texture_(
        paints: list[Paint],
        frame: skia.Rect,
        path: Path
    ) -> moderngl.Texture | None:
        surface = SkiaMobject._make_surface(
            int(frame.width() * PIXEL_PER_UNIT),
            int(frame.height() * PIXEL_PER_UNIT)
        )
        with surface as canvas:
            #canvas.clear(color=skia.Color4f.kTransparent)
            canvas.concat(skia.Matrix.MakeRectToRect(
                src=frame,
                dst=skia.Rect.Make(surface.imageInfo().bounds()),
                stf=skia.Matrix.kFill_ScaleToFit
            ))
            for paint in paints:
                canvas.drawPath(path=path._skia_path_, paint=paint)
        return SkiaMobject._make_texture(surface.makeImageSnapshot())

    #@SkiaMobject._update_model_matrix_by_refreshed_frame
    def set_path(self, path: Path):
        self._path_ = path
        return self

    @_fill_paint_.updater
    def set_local_fill(self, disable: bool | None = None, **kwargs):
        if disable is not None:
            self._disable_fill_ = disable
        self._fill_paint_.set(**kwargs)
        return self

    def set_fill(
        self,
        *,
        broadcast: bool = True,
        **kwargs
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, PathMobject):
                continue
            mobject.set_local_fill(**kwargs)
        return self

    @_stroke_paint_.updater
    def set_local_stroke(self, disable: bool | None = None, **kwargs):
        if disable is not None:
            self._disable_stroke_ = disable
        self._stroke_paint_.set(**kwargs)
        return self

    def set_stroke(
        self,
        *,
        broadcast: bool = True,
        **kwargs
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, PathMobject):
                continue
            mobject.set_local_stroke(**kwargs)
        return self

    def set_paint(
        self,
        *,
        disable_fill: bool | None = None,
        disable_stroke: bool | None = None,
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

        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, PathMobject):
                continue
            if draw_stroke_behind_fill is not None:
                self._draw_stroke_behind_fill_ = draw_stroke_behind_fill
            mobject.set_fill(**fill_kwargs, broadcast=False)
            mobject.set_stroke(**stroke_kwargs, broadcast=False)
        return self
