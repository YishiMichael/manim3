__all__ = ["PathMobject"]


import numpy as np
from scipy.interpolate import BSpline
import svgelements as se

from ..custom_typing import (
    Vec2T,
    Vec2sT
)
from ..mobjects.shape_mobject import ShapeMobject
from ..utils.shape import (
    LineString2D,
    MultiLineString2D,
    Shape
)


class PathMobject(ShapeMobject):
    def __init__(self, path: se.Path | str | None = None):
        if path is None:
            shape = Shape()
        else:
            shape = self.path_to_shape(path)
        super().__init__(shape)

    @classmethod
    def path_to_shape(cls, se_path: se.Path | str) -> Shape:
        if isinstance(se_path, str):
            se_path = se.Path(se_path)
        se_path.approximate_arcs_with_cubics()
        point_lists: list[list[Vec2T]] = []
        point_list: list[Vec2T] = []
        current_path_start_point: Vec2T = np.zeros(2)
        for segment in se_path.segments():
            if isinstance(segment, se.Move):
                point_lists.append(point_list)
                current_path_start_point = np.array(segment.end)
                point_list = [current_path_start_point]
            elif isinstance(segment, se.Close):
                point_list.append(current_path_start_point)
                point_lists.append(point_list)
                point_list = []
            else:
                if isinstance(segment, se.Line):
                    control_points = [segment.start, segment.end]
                elif isinstance(segment, se.QuadraticBezier):
                    control_points = [segment.start, segment.control, segment.end]
                elif isinstance(segment, se.CubicBezier):
                    control_points = [segment.start, segment.control1, segment.control2, segment.end]
                else:
                    raise ValueError(f"Cannot handle path segment type: {type(segment)}")
                point_list.extend(cls._get_bezier_sample_points(np.array(control_points))[1:])
        point_lists.append(point_list)

        return Shape(MultiLineString2D([
            LineString2D(np.array(coords))
            for coords in point_lists if coords
        ]))

    @classmethod
    def _get_bezier_sample_points(cls, points: Vec2sT) -> Vec2sT:
        order = len(points) - 1
        num_samples = 2 if order == 1 else 17
        gamma = BSpline(
            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
            c=points,
            k=order
        )
        return gamma(np.linspace(0.0, 1.0, num_samples)).astype(float)
