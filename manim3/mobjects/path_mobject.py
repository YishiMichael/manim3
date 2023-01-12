__all__ = ["PathMobject"]


from functools import reduce
from typing import Callable

import numpy as np
from scipy.interpolate import BSpline
import shapely.geometry
import svgelements as se

from ..custom_typing import (
    ColorType,
    Vec2T,
    Vec2sT,
    Vec4T
)
from ..mobjects.shape_mobject import ShapeMobject
from ..utils.shape import Shape


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
        polygon_point_lists: list[list[Vec2T]] = []
        current_list: list[Vec2T] = []
        for segment in se_path.segments():
            if isinstance(segment, se.Move):
                polygon_point_lists.append(current_list)
                current_list = [np.array(segment.end)]
            elif isinstance(segment, se.Close):
                polygon_point_lists.append(current_list)
                current_list = []
            else:
                if isinstance(segment, se.Line):
                    control_points = [segment.start, segment.end]
                elif isinstance(segment, se.QuadraticBezier):
                    control_points = [segment.start, segment.control, segment.end]
                elif isinstance(segment, se.CubicBezier):
                    control_points = [segment.start, segment.control1, segment.control2, segment.end]
                else:
                    raise ValueError(f"Cannot handle path segment type: {type(segment)}")
                current_list.extend(cls._get_bezier_sample_points(np.array(control_points))[1:])
        polygon_point_lists.append(current_list)

        return Shape(reduce(shapely.geometry.base.BaseGeometry.__xor__, [
            shapely.geometry.Polygon(polygon_point_list)
            for polygon_point_list in polygon_point_lists
            if polygon_point_list
        ]))

    @classmethod
    def _get_bezier_sample_points(cls, points: Vec2sT) -> Vec2sT:
        order = len(points) - 1
        num_samples = 2 if order == 1 else 17
        return BSpline(
            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
            c=points,
            k=order
        )(np.linspace(0.0, 1.0, num_samples)).astype(float)

    def set_local_fill(self, color: ColorType | Callable[..., Vec4T]):
        self._color_ = color
        return self

    def set_fill(
        self,
        color: ColorType | Callable[..., Vec4T],
        *,
        broadcast: bool = True
    ):
        for mobject in self.get_descendants(broadcast=broadcast):
            if not isinstance(mobject, PathMobject):
                continue
            mobject.set_local_fill(color=color)
        return self
