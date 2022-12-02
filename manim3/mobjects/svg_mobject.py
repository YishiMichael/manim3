__all__ = ["SVGMobject"]


from typing import Any
import warnings

import skia
import svgelements as se

from ..mobjects.path_mobject import PathGroup
from ..mobjects.path_mobject import PathMobject
from ..utils.path import Path
from ..custom_typing import *


class SVGMobject(PathGroup):
    def __init__(
        self,
        file_path: str,
        *,
        width: Real | None = None,
        height: Real | None = None,
        frame_scale: Real | None = None,
        paint_settings: dict[str, Any] | None = None
    ):
        svg = se.SVG.parse(file_path)
        svg_bbox = skia.Rect.MakeXYWH(*svg.bbox())
        svg_frame = self._calculate_frame(
            svg_bbox.width(),
            svg_bbox.height(),
            width,
            height,
            frame_scale
        )
        transform_matrix = skia.Matrix.MakeRectToRect(
            src=svg_bbox,
            dst=svg_frame,
            stf=skia.Matrix.kFill_ScaleToFit
        )

        mobjects = []
        for shape in svg.elements():
            if not isinstance(shape, se.Shape):
                continue
            path = self.shape_to_skia_path(shape)
            if path is None:
                continue
            if isinstance(shape, se.Transformable) and shape.apply:
                path.transform(self.convert_transform(shape.transform))
            path.transform(transform_matrix)
            mobject = PathMobject(path=Path(path), flip_y=False)
            if paint_settings is not None:
                mobject.set_paint(**paint_settings)
            mobject.set_paint(**self.get_paint_settings_from_shape(shape))
            mobjects.append(mobject)
        super().__init__(*mobjects)

    @classmethod
    def shape_to_skia_path(cls, shape: se.Shape) -> skia.Path | None:
        if isinstance(shape, (se.Group, se.Use)):
            return None
        if isinstance(shape, se.Path):
            return cls.path_to_skia_path(shape)
            #mob = self.path_to_mobject(shape)
        if isinstance(shape, se.SimpleLine):
            return None
            #mob = self.line_to_mobject(shape)
        if isinstance(shape, se.Rect):
            return None
            #mob = self.rect_to_mobject(shape)
        if isinstance(shape, (se.Circle, se.Ellipse)):
            return None
            #mob = self.ellipse_to_mobject(shape)
        if isinstance(shape, se.Polygon):
            return None
            #mob = self.polygon_to_mobject(shape)
        if isinstance(shape, se.Polyline):
            return None
            #mob = self.polyline_to_mobject(shape)
        if type(shape) == se.SVGElement:
            return None
        warnings.warn(f"Unsupported element type: {type(shape)}")
        return None

    @classmethod
    def convert_transform(cls, matrix: se.Matrix) -> skia.Matrix:
        return skia.Matrix.MakeAll(
            scaleX=matrix.a,
            skewX=matrix.c,
            transX=matrix.e,
            skewY=matrix.b,
            scaleY=matrix.d,
            transY=matrix.f,
            pers0=0.0,
            pers1=0.0,
            pers2=1.0
        )

    @classmethod
    def get_paint_settings_from_shape(cls, shape: se.GraphicObject) -> dict[str, Any]:
        return {
            "fill_color": None if shape.fill is None else shape.fill.hexrgb,
            "fill_opacity": None if shape.fill is None else shape.fill.opacity,
            "stroke_color": None if shape.stroke is None else shape.stroke.hexrgb,
            "stroke_opacity": None if shape.stroke is None else shape.stroke.opacity,
            # Don't know why, svgelements may parse stroke_width out of nothing...
            "stroke_width": shape.stroke_width
        }

    @classmethod
    def path_to_skia_path(cls, path_shape: se.Path) -> skia.Path:
        path_shape.approximate_arcs_with_cubics()
        path = skia.Path()
        for segment in path_shape.segments():
            if isinstance(segment, se.Move):
                path.moveTo(*segment.end)
            elif isinstance(segment, se.Close):
                path.close()
            elif isinstance(segment, se.Line):
                path.lineTo(*segment.end)
            elif isinstance(segment, se.QuadraticBezier):
                path.quadTo(*segment.control, *segment.end)
            elif isinstance(segment, se.CubicBezier):
                path.cubicTo(*segment.control1, *segment.control2, *segment.end)
            else:
                raise ValueError(f"Cannot handle path segment type: {type(segment)}")
        return path
