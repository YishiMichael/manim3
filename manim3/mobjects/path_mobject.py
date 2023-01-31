#__all__ = ["PathMobject"]


#import itertools as it
#from typing import Callable

#import numpy as np
#from scipy.interpolate import BSpline
#import svgelements as se

#from ..custom_typing import (
#    FloatsT,
#    Vec2T,
#    Vec2sT
#)
#from ..mobjects.shape_mobject import ShapeMobject
#from ..utils.shape import (
#    LineString2D,
#    MultiLineString2D,
#    Shape
#)


#class PathMobject(ShapeMobject):
#    def __init__(self, path: se.Path | str | None = None):
#        super().__init__()
#        if path is not None:
#            self.set_shape(self._path_to_shape(path))

#    @classmethod
#    def _path_to_shape(cls, se_path: se.Path | str) -> Shape:
#        if isinstance(se_path, str):
#            se_path = se.Path(se_path)
#        se_path.approximate_arcs_with_cubics()
#        point_lists: list[list[Vec2T]] = []
#        point_list: list[Vec2T] = []
#        current_path_start_point: Vec2T = np.zeros(2)
#        for segment in se_path.segments():
#            if isinstance(segment, se.Move):
#                point_lists.append(point_list)
#                current_path_start_point = np.array(segment.end)
#                point_list = [current_path_start_point]
#            elif isinstance(segment, se.Close):
#                point_list.append(current_path_start_point)
#                point_lists.append(point_list)
#                point_list = []
#            elif isinstance(segment, se.Line):
#                point_list.append(np.array(segment.end))
#            else:
#                if isinstance(segment, se.QuadraticBezier):
#                    control_points = [segment.start, segment.control, segment.end]
#                elif isinstance(segment, se.CubicBezier):
#                    control_points = [segment.start, segment.control1, segment.control2, segment.end]
#                else:
#                    raise ValueError(f"Cannot handle path segment type: {type(segment)}")
#                point_list.extend(cls._get_bezier_sample_points(np.array(control_points))[1:])
#        point_lists.append(point_list)

#        return Shape(MultiLineString2D([
#            LineString2D(np.array(coords))
#            for coords in point_lists if coords
#        ]))

#    @classmethod
#    def _get_bezier_sample_points(cls, control_points: Vec2sT) -> Vec2sT:
#        def smoothen_samples(curve: Callable[[FloatsT], Vec2sT], samples: FloatsT, bisect_depth: int) -> FloatsT:
#            # Bisect a segment if one of its endpoints has a turning angle above the threshold.
#            # Bisect for no more than 4 times, so each curve will be split into no more than 16 segments.
#            if bisect_depth == 4:
#                return samples
#            points = curve(samples)
#            directions = points[1:] - points[:-1]
#            directions /= np.linalg.norm(directions, axis=1)[:, None]
#            angles = abs(np.arccos((directions[1:] * directions[:-1]).sum(axis=1)))
#            large_angle_indices = np.squeeze(np.argwhere(angles > np.pi / 16.0), axis=1)
#            if not len(large_angle_indices):
#                return samples
#            insertion_index_pairs = np.array(list(dict.fromkeys(it.chain(*(
#                ((i, i + 1), (i + 1, i + 2))
#                for i in large_angle_indices
#            )))))
#            new_samples = np.average(samples[insertion_index_pairs], axis=1)
#            return smoothen_samples(curve, np.sort(np.concatenate((samples, new_samples))), bisect_depth + 1)

#        order = len(control_points) - 1
#        gamma = BSpline(
#            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
#            c=control_points,
#            k=order
#        )
#        if np.isclose(np.linalg.norm(gamma(1.0) - gamma(0.0)), 0.0):
#            return np.array((gamma(0.0),))
#        samples = smoothen_samples(gamma, np.linspace(0.0, 1.0, 3), 1)
#        return gamma(samples).astype(float)
