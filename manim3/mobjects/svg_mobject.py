__all__ = ["SVGMobject"]


import numpy as np
import svgelements as se

from ..lazy.core import LazyWrapper
from ..mobjects.shape_mobject import ShapeMobject
from ..utils.color import ColorUtils
from ..utils.shape import Shape


class SVGMobject(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        file_path: str | None = None,
        *,
        width: float | None = None,
        height: float | None = None,
        frame_scale: float | None = None
    ) -> None:
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

        shape_mobjects = [
            shape_mobject.set_style(
                color=None if shape.fill is None else shape.fill.hexrgb
                #opacity=None if shape.fill is None else shape.fill.opacity
                #stroke_color=None if shape.stroke is None else shape.stroke.hexrgb,
                #stroke_opacity=None if shape.stroke is None else shape.stroke.opacity,
                # Don't know why, svgelements may parse stroke_width out of nothing...
                #stroke_width=shape.stroke_width
            )
            for shape in svg.elements()
            if isinstance(shape, se.Shape) and (shape_mobject := ShapeMobject(
                Shape.from_se_shape(shape * transform)
            ))._has_local_sample_points_.value
        ]

        # Share the `_color_` values.
        # Useful when calling `concatenate()` method.
        color_hex_to_mobjects: dict[str, list[ShapeMobject]] = {}
        for mobject in shape_mobjects:
            color_hex_to_mobjects.setdefault(ColorUtils.color_to_hex(mobject._color_.value), []).append(mobject)
        for color_hex, mobjects in color_hex_to_mobjects.items():
            rgb, _ = ColorUtils.decompose_color(color_hex)
            color_value = LazyWrapper(rgb)
            for mobject in mobjects:
                mobject._color_ = color_value

        self.add(*shape_mobjects)
        self.scale(np.array((1.0, -1.0, 1.0)))  # flip y
