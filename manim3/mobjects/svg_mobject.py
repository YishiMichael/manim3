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

from ..custom_typing import (
    FloatsT,
    Vec2T,
    Vec2sT
    #Vec3T
)
#from ..lazy.core import LazyWrapper
from ..mobjects.shape_mobject import ShapeMobject
#from ..utils.color import ColorUtils
from ..utils.shape import Shape
from ..utils.space import SpaceUtils


class BezierCurve(BSpline):
    __slots__ = ()

    def __init__(
        self,
        control_points: Vec2sT
    ) -> None:
        degree = len(control_points) - 1
        super().__init__(
            t=np.append(np.zeros(degree + 1), np.ones(degree + 1)),
            c=control_points,
            k=degree
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
        transform = self._get_transform(
            bbox=bbox,
            frame_scale=frame_scale,
            width=width,
            height=height
        )
        self.add(*(
            shape_mobject.set_style(
                # TODO: handle other attributes including opacity, strokes, etc.
                color=None if shape.fill is None else shape.fill.hexrgb
            )
            for shape in svg.elements()
            if isinstance(shape, se.Shape) and (
                shape_mobject := self._get_mobject_from_se_shape(shape * transform)
            )._has_local_sample_points_.value
        ))

        # Share the `_color_` values.
        # Useful when calling `concatenate()` method.
        #color_hex_to_value_dict: dict[str, LazyWrapper[Vec3T]] = {}
        #for mobject in shape_mobjects:
        #    color = mobject._color_.value
        #    color_hex = ColorUtils.color_to_hex(color)
        #    if (color_value := color_hex_to_value_dict.get(color_hex)) is None:
        #        color_value = LazyWrapper(color)
        #        color_hex_to_value_dict[color_hex] = color_value
        #    mobject._color_ = color_value

        #self.add(*shape_mobjects)
        #self.flip(X_AXIS)

    @classmethod
    def _get_transform(
        cls,
        bbox: tuple[float, float, float, float],
        frame_scale: float | None,
        width: float | None,
        height: float | None
    ) -> se.Matrix:

        def perspective(
            origin_x: float,
            origin_y: float,
            radius_x: float,
            radius_y: float
        ) -> se.Matrix:
            # `(origin=(0.0, 0.0), radius=(1.0, 1.0))` ->
            # `(origin=(origin_x, origin_y), radius=(radius_x, radius_y))`
            return se.Matrix(
                radius_x,
                0.0,
                0.0,
                radius_y,
                origin_x,
                origin_y
            )

        x_min, y_min, x_max, y_max = bbox
        origin_x = (x_min + x_max) / 2.0
        origin_y = (y_min + y_max) / 2.0
        radius_x = (x_max - x_min) / 2.0
        radius_y = (y_max - y_min) / 2.0
        transform = ~perspective(
            origin_x=origin_x,
            origin_y=origin_y,
            radius_x=radius_x,
            radius_y=radius_y
        )
        x_scale, y_scale = cls._get_frame_scale_vector(
            original_width=radius_x * 2.0,
            original_height=radius_y * 2.0,
            specified_frame_scale=frame_scale,
            specified_width=width,
            specified_height=height
        )
        transform *= perspective(
            origin_x=0.0,
            origin_y=0.0,
            radius_x=x_scale * radius_x,
            radius_y=-y_scale * radius_y  # Flip y.
        )
        # transform = flip_mat @ shift_origin_mat @ scale_mat
        # flip_mat = (
        #     (1.0,  0.0, 0.0),
        #     (0.0, -1.0, 0.0)
        # )
        # shift_origin_mat = (
        #     (1.0, 0.0, (x_max + x_min) / 2.0),
        #     (0.0, 1.0, (y_max + y_min) / 2.0)
        # )
        # scale_mat = (
        #     (x_scale,     0.0, 0.0),
        #     (    0.0, y_scale, 0.0)
        # )
        #transform = se.Matrix(
        #    x_scale,
        #    0.0,
        #    0.0,
        #    -y_scale,
        #    -(x_max + x_min) * x_scale / 2.0,
        #    (y_max + y_min) * y_scale / 2.0
        #)
        return transform

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
            insertion_index_pairs = np.array(list(dict.fromkeys(it.chain.from_iterable(
                ((i, i + 1), (i + 1, i + 2))
                for i in large_angle_indices
            ))))
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

        def iter_args_from_se_shape(
            se_shape: se.Shape
        ) -> Generator[tuple[Vec2sT, bool], None, None]:
            se_path = se.Path(se_shape.segments(transformed=True))
            se_path.approximate_arcs_with_cubics()
            points_list: list[Vec2T] = []
            is_ring: bool = False
            for segment in se_path.segments(transformed=True):
                if isinstance(segment, se.Move):
                    yield np.array(points_list), is_ring
                    points_list = [np.array(segment.end)]
                    is_ring = False
                elif isinstance(segment, se.Close):
                    is_ring = True
                elif isinstance(segment, se.Line):
                    points_list.append(np.array(segment.end))
                else:
                    if isinstance(segment, se.QuadraticBezier):
                        control_points = [segment.start, segment.control, segment.end]
                    elif isinstance(segment, se.CubicBezier):
                        control_points = [segment.start, segment.control1, segment.control2, segment.end]
                    else:
                        raise ValueError(f"Cannot handle path segment type: {type(segment)}")
                    points_list.extend(get_bezier_sample_points(np.array(control_points))[1:])
            yield np.array(points_list), is_ring

        return ShapeMobject(Shape(iter_args_from_se_shape(se_shape)))
