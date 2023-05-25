import itertools as it
import pathlib
from typing import (
    Callable,
    Iterator,
    overload
)

import numpy as np
from scipy.interpolate import BSpline
import svgelements as se

from ..custom_typing import (
    FloatsT,
    Vec2T,
    Vec2sT
)
from ..mobjects.shape_mobject import ShapeMobject
from ..shape.shape import Shape
from ..utils.iterables import IterUtils
from ..utils.space import SpaceUtils


class BezierCurve(BSpline):
    __slots__ = ("_degree",)

    def __init__(
        self,
        control_points: Vec2sT
    ) -> None:
        degree = len(control_points) - 1
        assert degree >= 0
        super().__init__(
            t=np.append(np.zeros(degree + 1), np.ones(degree + 1)),
            c=control_points,
            k=degree
        )
        self._degree: int = degree

    @overload
    def gamma(
        self,
        sample: float
    ) -> Vec2T: ...

    @overload
    def gamma(
        self,
        sample: FloatsT
    ) -> Vec2sT: ...

    def gamma(
        self,
        sample: float | FloatsT
    ) -> Vec2T | Vec2sT:
        return self.__call__(sample)

    def get_sample_points(self) -> Vec2sT:
        # Approximate the bezier curve with a polyline.

        def smoothen_samples(
            gamma: Callable[[FloatsT], Vec2sT],
            samples: FloatsT,
            bisect_depth: int
        ) -> FloatsT:
            # Bisect a segment if one of its endpoints has a turning angle above the threshold.
            # Bisect for no more than 4 times, so each curve will be split into no more than 16 segments.
            if bisect_depth == 4:
                return samples
            points = gamma(samples)
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
            return smoothen_samples(gamma, np.sort(np.concatenate((samples, new_samples))), bisect_depth + 1)

        if self._degree <= 1:
            start_point = self.gamma(0.0)
            stop_point = self.gamma(1.0)
            if np.isclose(SpaceUtils.norm(stop_point - start_point), 0.0):
                return np.array((start_point,))
            return np.array((start_point, stop_point))
        samples = smoothen_samples(self.gamma, np.linspace(0.0, 1.0, 3), 1)
        return self.gamma(samples)


class SVGMobject(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        file_path: str | pathlib.Path | None = None,
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

        # Handle transform before constructing `ShapeGeometry`
        # so that the center of the geometry falls on the origin.
        transform = self._get_transform(
            bbox=bbox,
            width=width,
            height=height,
            frame_scale=frame_scale
        )

        #mobjects: list[ShapeMobject] = []
        #hexa_to_mobjects: dict[str, list[ShapeMobject]] = {}
        #for shape in svg.elements():
        #    if not isinstance(shape, se.Shape):

        # TODO: handle strokes, etc.
        mobject_hexa_pairs = [
            (
                fill.hexa if (fill := shape.fill) is not None else None,
                shape_mobject
            )
            for shape in svg.elements()
            if isinstance(shape, se.Shape) and (
                shape_mobject := ShapeMobject(self._get_shape_from_se_shape(shape * transform))
            )._has_local_sample_points_
        ]
        for hexa, mobjects in IterUtils.categorize(mobject_hexa_pairs):
            ShapeMobject().add(*mobjects).set_style(color=hexa)

        self.add(*(
            shape_mobject
            for _, shape_mobject in mobject_hexa_pairs
            #for shape in svg.elements()
            #if isinstance(shape, se.Shape) and (
            #    # TODO: handle strokes, etc.
            #    shape_mobject := self._get_mobject_from_se_shape(shape * transform)
            #)._has_local_sample_points_
        ))

    @classmethod
    def _get_transform(
        cls,
        bbox: tuple[float, float, float, float],
        width: float | None,
        height: float | None,
        frame_scale: float | None
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

        min_x, min_y, max_x, max_y = bbox
        origin_x = (min_x + max_x) / 2.0
        origin_y = (min_y + max_y) / 2.0
        radius_x = (max_x - min_x) / 2.0
        radius_y = (max_y - min_y) / 2.0
        transform = ~perspective(
            origin_x=origin_x,
            origin_y=origin_y,
            radius_x=radius_x,
            radius_y=radius_y
        )
        scale_x, scale_y = SpaceUtils._get_frame_scale_vector(
            original_width=radius_x * 2.0,
            original_height=radius_y * 2.0,
            specified_width=width,
            specified_height=height,
            specified_frame_scale=frame_scale
        )
        transform *= perspective(
            origin_x=0.0,
            origin_y=0.0,
            radius_x=scale_x * radius_x,
            radius_y=-scale_y * radius_y  # Flip y.
        )
        return transform

    @classmethod
    def _get_shape_from_se_shape(
        cls,
        se_shape: se.Shape
    ) -> Shape:

        def iter_args_from_se_shape(
            se_shape: se.Shape
        ) -> Iterator[tuple[Vec2sT, bool]]:
            se_path = se.Path(se_shape.segments(transformed=True))
            se_path.approximate_arcs_with_cubics()
            points_list: list[Vec2T] = []
            is_ring: bool = False
            for segment in se_path.segments(transformed=True):
                match segment:
                    case se.Move(end=end):
                        yield np.array(points_list), is_ring
                        points_list = [np.array(end)]
                        is_ring = False
                    case se.Close():
                        is_ring = True
                    case se.Line() | se.QuadraticBezier() | se.CubicBezier():
                        control_points = np.array(segment)
                        points_list.extend(BezierCurve(control_points).get_sample_points()[1:])
                    case _:
                        raise ValueError(f"Cannot handle path segment type: {type(segment)}")
            yield np.array(points_list), is_ring

        return Shape(iter_args_from_se_shape(se_shape))
        #if (fill := se_shape.fill) is not None and (hexa := fill.hexa) is not None:
        #    result.set_style(color=hexa)
        #return result
