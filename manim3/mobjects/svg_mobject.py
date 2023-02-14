__all__ = ["SVGMobject"]


import numpy as np
import svgelements as se

from ..custom_typing import Real
from ..mobjects.shape_mobject import ShapeMobject


class SVGMobject(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        file_path: str | None = None,
        *,
        width: Real | None = None,
        height: Real | None = None,
        frame_scale: Real | None = None
    ):
        super().__init__()
        if file_path is None:
            return

        svg: se.SVG = se.SVG.parse(file_path)
        bbox: tuple[float, float, float, float] | None = svg.bbox()
        if bbox is None:
            return

        # Handle transform before constructing ShapeGeometry
        # so that the center of the geometry falls on the origin.
        x_min, y_min, x_max, y_max = bbox
        x_scale, y_scale = self._get_frame_scale_vector(
            x_max - x_min,
            y_max - y_min,
            width,
            height,
            frame_scale
        )
        transform = se.Matrix(
            x_scale,
            0.0,
            0.0,
            y_scale,
            -(x_max + x_min) * x_scale / 2.0,
            -(y_max + y_min) * y_scale / 2.0
        )

        #if default_style is None:
        #    default_style = {}
        #shape_mobjects: list[ShapeMobject] = []
        shape_mobjects = [
            shape_mobject.set_style(
                color=None if shape.fill is None else shape.fill.hexrgb,
                #opacity=None if shape.fill is None else shape.fill.opacity
                #stroke_color=None if shape.stroke is None else shape.stroke.hexrgb,
                #stroke_opacity=None if shape.stroke is None else shape.stroke.opacity,
                # Don't know why, svgelements may parse stroke_width out of nothing...
                #stroke_width=shape.stroke_width
            )
            for shape in svg.elements()
            if isinstance(shape, se.Shape) and (shape_mobject := ShapeMobject(shape * transform))._has_local_sample_points_
        ]
        #if shape_mobjects:
        #    return

        #for shape in svg.elements():
        #    if not isinstance(shape, se.Shape):
        #        continue
        #    #path = self.shape_to_path(shape)
        #    #if path is None:
        #    #    continue
        #    mobject = ShapeMobject(shape)
        #    #if isinstance(shape, se.Transformable) and shape.apply:
        #    #    mobject.apply_transform(self.convert_transform(shape.transform))
        #    #mobject.apply_transform_locally(transform_matrix)
        #    #if paint_settings is not None and (color := paint_settings.get("fill_color")) is not None:
        #    #    mobject.set_fill(color=color)
        #    #mobject.set_style(**default_style)
        #    mobject.set_style(**self.get_style_from_shape(shape))
        #    #if (color := self.get_paint_settings_from_shape(shape).get("fill_color")) is not None:
        #    #    mobject.set_fill(color=color)
        #    #mobject.set_paint(**self.get_paint_settings_from_shape(shape))
        #    shape_mobjects.append(mobject)

        self.add(*shape_mobjects)
        #self.adjust_frame(
        #    svg.width,
        #    svg.height,
        #    width,
        #    height,
        #    frame_scale
        #)
        self.scale(np.array((1.0, -1.0, 1.0)))  # flip y

    @property
    def _shape_mobjects(self) -> list[ShapeMobject]:
        return [child for child in self._children if isinstance(child, ShapeMobject)]

    #@classmethod
    #def shape_to_path(cls, shape: se.Shape) -> se.Path | None:
    #    if isinstance(shape, (se.Group, se.Use)):
    #        return None
    #    if isinstance(shape, se.Path):
    #        return shape
    #        #mob = self.path_to_mobject(shape)
    #    if isinstance(shape, se.SimpleLine):
    #        return None
    #        #mob = self.line_to_mobject(shape)
    #    if isinstance(shape, se.Rect):
    #        return None
    #        #mob = self.rect_to_mobject(shape)
    #    if isinstance(shape, (se.Circle, se.Ellipse)):
    #        return None
    #        #mob = self.ellipse_to_mobject(shape)
    #    if isinstance(shape, se.Polygon):
    #        return None
    #        #mob = self.polygon_to_mobject(shape)
    #    if isinstance(shape, se.Polyline):
    #        return None
    #        #mob = self.polyline_to_mobject(shape)
    #    if type(shape) == se.SVGElement:
    #        return None
    #    warnings.warn(f"Unsupported element type: {type(shape)}")
    #    return None

    #@classmethod
    #def convert_transform(cls, matrix: se.Matrix) -> Mat4T:
    #    return np.array((
    #        (matrix.a, matrix.c, 0.0, matrix.e),
    #        (matrix.b, matrix.d, 0.0, matrix.f),
    #        (     0.0,      0.0, 1.0,      0.0),
    #        (     0.0,      0.0, 0.0,      1.0)
    #    ))

    #@classmethod
    #def get_style_from_shape(cls, shape: se.Shape) -> dict[str, Any]:
    #    return {
    #        "color": None if shape.fill is None else shape.fill.hexrgb,
    #        "opacity": None if shape.fill is None else shape.fill.opacity,
    #        #"stroke_color": None if shape.stroke is None else shape.stroke.hexrgb,
    #        #"stroke_opacity": None if shape.stroke is None else shape.stroke.opacity,
    #        # Don't know why, svgelements may parse stroke_width out of nothing...
    #        #"stroke_width": shape.stroke_width
    #    }
