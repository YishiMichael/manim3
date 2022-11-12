import skia
import svgelements as se
import warnings

from ..mobjects.mobject import Group
from ..mobjects.skia_mobject import SkiaMobject
from ..typing import *


__all__ = [
    "SkiaPathMobject",
    "SVGMobject"
]


class SkiaPathMobject(SkiaMobject):
    def __init__(
        self: Self,
        path: skia.Path,
        transform_matrix: skia.Matrix,
        frame_buff: tuple[Real, Real] = (0.5, 0.5),
    ):
        inverse_matrix = skia.Matrix()
        transform_matrix.invert(inverse_matrix)
        tight_path_bbox = path.computeTightBounds()
        path_frame = transform_matrix.mapRect(tight_path_bbox).makeOutset(*frame_buff)
        path_bbox = inverse_matrix.mapRect(path_frame)
        resolution = self.calculate_resolution_by_frame(path_frame)
        path.transform(skia.Matrix.MakeRectToRect(
            path_bbox, skia.Rect.MakeWH(*resolution), skia.Matrix.kFill_ScaleToFit
        ))
        super().__init__(frame=path_frame, resolution=resolution)
        self.path: skia.Path = path
        self.fill_paint: skia.Paint | None = None
        self.stroke_paint: skia.Paint | None = None

    def draw(self: Self, canvas: skia.Canvas) -> None:
        if self.fill_paint is not None:
            canvas.drawPath(self.path, self.fill_paint)
        if self.stroke_paint is not None:
            canvas.drawPath(self.path, self.stroke_paint)


class SVGMobject(Group):
    def __init__(
        self: Self,
        file_path: str,
        *,
        width: Real | None = None,
        height: Real | None = None
    ):
        svg = se.SVG.parse(file_path)
        # TODO: bbox() may return None
        svg_bbox = skia.Rect.MakeXYWH(*svg.bbox())
        svg_frame = SkiaMobject.calculate_frame_by_aspect_ratio(
            width, height, svg_bbox.width() / svg_bbox.height()
        )  # TODO: handle this staticmethod
        transform_matrix = skia.Matrix.MakeRectToRect(
            svg_bbox, svg_frame, skia.Matrix.kFill_ScaleToFit
        )

        mobjects = []
        for shape in svg.elements():
            path = self.shape_to_path(shape)
            if path is None:
                continue
            if isinstance(shape, se.Transformable) and shape.apply:
                self.apply_transform(path, shape.transform)
            mobject = SkiaPathMobject(
                path=path,
                transform_matrix=transform_matrix
            )
            self.apply_style_to_mobject(mobject, shape)
            mobjects.append(mobject)
        super().__init__(*mobjects)

    @classmethod
    def shape_to_path(cls, shape: se.Shape) -> skia.Path | None:
        if isinstance(shape, (se.Group, se.Use)):
            return None
        if isinstance(shape, se.Path):
            return cls.path_to_path(shape)
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
    def apply_transform(cls, path: skia.Path, matrix: se.Matrix) -> skia.Path:
        skia_matrix = skia.Matrix.MakeAll(
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
        path.transform(skia_matrix)
        return path
        #transform_matrix = pyrr.Matrix44.identity()
        #transform_matrix[[0, 1, 3]][:, [0, 1, 3]] = np.array((
        #    (matrix.a, matrix.b, 0.0),
        #    (matrix.c, matrix.d, 0.0),
        #    (matrix.e, matrix.f, 1.0)
        #))
        #mobject.apply_matrix(transform_matrix, about_point=ORIGIN)
        #mat = np.array([
        #    [matrix.a, matrix.c],
        #    [matrix.b, matrix.d]
        #])
        #vec = np.array([matrix.e, matrix.f, 0.0])
        #mob.apply_matrix(mat)
        #mob.shift(vec)

    @classmethod
    def apply_style_to_mobject(cls, mobject: SkiaPathMobject, shape: se.GraphicObject) -> SkiaPathMobject:
        if shape.fill is not None and shape.fill.argb is not None:
            mobject.fill_paint = skia.Paint(
                Style=skia.Paint.kFill_Style,
                Color=shape.fill.argb
            )
        if shape.stroke is not None and shape.stroke.argb is not None and shape.stroke_width is not None:
            mobject.stroke_paint = skia.Paint(
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=shape.stroke_width,
                Color=shape.stroke.argb
            )
        return mobject
        #mob.set_style(
        #    stroke_width=shape.stroke_width,
        #    stroke_color=shape.stroke.hexrgb,
        #    stroke_opacity=shape.stroke.opacity,
        #    fill_color=shape.fill.hexrgb,
        #    fill_opacity=shape.fill.opacity
        #)
        #return mob

    @classmethod
    def path_to_path(cls, path_shape: se.Path) -> skia.Path:
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
