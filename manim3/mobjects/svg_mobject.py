__all__ = ["SVGMobject"]


import itertools as it
import pathlib
from typing import (
    Generator,
    overload
)

import numpy as np
from scipy.interpolate import BSpline
import svgelements as se

from ..constants import X_AXIS
from ..custom_typing import (
    FloatsT,
    Vec2T,
    Vec2sT,
    Vec3T
)
from ..lazy.core import LazyWrapper
from ..mobjects.shape_mobject import ShapeMobject
from ..utils.color import ColorUtils
from ..utils.shape import Shape
from ..utils.space import SpaceUtils


class BezierCurve(BSpline):
    __slots__ = ()

    def __init__(
        self,
        control_points: Vec2sT
    ) -> None:
        order = len(control_points) - 1
        super().__init__(
            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
            c=control_points,
            k=order
        )

    @overload
    def __call__(
        self,
        sample: float
    ) -> Vec2T: ...

    @overload
    def __call__(
        self,
        sample: FloatsT
    ) -> Vec2sT: ...

    def __call__(
        self,
        sample: float | FloatsT
    ) -> Vec2T | Vec2sT:
        return super().__call__(sample)


class SVGMobject(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        file_path: str | pathlib.Path | None = None,
        *,
        frame_scale: float | None = None,
        width: float | None = None,
        height: float | None = None
    ) -> None:
        super().__init__()
        if file_path is None:
            return

        svg: se.SVG = se.SVG.parse(file_path)
        bbox: tuple[float, float, float, float] | None = svg.bbox()
        if bbox is None:
            return

        # Handle transform before constructing `ShapeGeometry`
        # so that the center of the geometry falls on the origin.
        x_min, y_min, x_max, y_max = bbox
        x_scale, y_scale = self._get_frame_scale_vector(
            original_width=x_max - x_min,
            original_height=y_max - y_min,
            specified_frame_scale=frame_scale,
            specified_width=width,
            specified_height=height
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
            if isinstance(shape, se.Shape) and (
                shape_mobject := self._get_mobject_from_se_shape(shape * transform)
            )._has_local_sample_points_.value
        ]

        # Share the `_color_` values.
        # Useful when calling `concatenate()` method.
        color_hex_to_value_dict: dict[str, LazyWrapper[Vec3T]] = {}
        for mobject in shape_mobjects:
            color = mobject._color_.value
            color_hex = ColorUtils.color_to_hex(color)
            if (color_value := color_hex_to_value_dict.get(color_hex)) is None:
                color_value = LazyWrapper(color)
                color_hex_to_value_dict[color_hex] = color_value
            mobject._color_ = color_value

        self.add(*shape_mobjects)
        self.flip(X_AXIS)

    @classmethod
    def _get_mobject_from_se_shape(
        cls,
        se_shape: se.Shape
    ) -> ShapeMobject:

        def smoothen_samples(
            curve: BezierCurve,
            samples: FloatsT,
            bisect_depth: int
        ) -> FloatsT:
            # Bisect a segment if one of its endpoints has a turning angle above the threshold.
            # Bisect for no more than 4 times, so each curve will be split into no more than 16 segments.
            if bisect_depth == 4:
                return samples
            points = curve(samples)
            directions = SpaceUtils.normalize(points[1:] - points[:-1])
            angles = abs(np.arccos((directions[1:] * directions[:-1]).sum(axis=1)))
            large_angle_indices = np.squeeze(np.argwhere(angles > np.pi / 16.0), axis=1)
            if not len(large_angle_indices):
                return samples
            insertion_index_pairs = np.array(list(dict.fromkeys(it.chain(*(
                ((i, i + 1), (i + 1, i + 2))
                for i in large_angle_indices
            )))))
            new_samples = np.average(samples[insertion_index_pairs], axis=1)
            return smoothen_samples(curve, np.sort(np.concatenate((samples, new_samples))), bisect_depth + 1)

        def get_bezier_sample_points(
            control_points: Vec2sT
        ) -> Vec2sT:
            gamma = BezierCurve(control_points)
            if np.isclose(SpaceUtils.norm(gamma(1.0) - gamma(0.0)), 0.0):
                return np.array((gamma(0.0),))
            samples = smoothen_samples(gamma, np.linspace(0.0, 1.0, 3), 1)
            return gamma(samples)

        def iter_coords_from_se_shape(
            se_shape: se.Shape
        ) -> Generator[Vec2sT, None, None]:
            se_path = se.Path(se_shape.segments(transformed=True))
            se_path.approximate_arcs_with_cubics()
            coords_list: list[Vec2T] = []
            for segment in se_path.segments(transformed=True):
                if isinstance(segment, se.Move):
                    yield np.array(coords_list)
                    coords_list = [np.array(segment.end)]
                elif isinstance(segment, se.Linear):  # Line & Close
                    coords_list.append(np.array(segment.end))
                else:
                    if isinstance(segment, se.QuadraticBezier):
                        control_points = [segment.start, segment.control, segment.end]
                    elif isinstance(segment, se.CubicBezier):
                        control_points = [segment.start, segment.control1, segment.control2, segment.end]
                    else:
                        raise ValueError(f"Cannot handle path segment type: {type(segment)}")
                    coords_list.extend(get_bezier_sample_points(np.array(control_points))[1:])
            yield np.array(coords_list)

        return ShapeMobject(Shape(iter_coords_from_se_shape(se_shape)))
