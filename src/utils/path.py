import copy
from dataclasses import dataclass
from functools import reduce
import itertools as it
from typing import Generator, Self

from mapbox_earcut import triangulate_float32
import numpy as np
import pathops
from scipy.special import comb
import svgelements as se

from utils.arrays import Array, IArray, Mat3, Vec2, Vec2s


#__all__ = ["Path"]


def line_intersection(line0: tuple[Vec2, Vec2], line1: tuple[Vec2, Vec2]) -> Vec2:
    a = np.array([
        [line0[1][1] - line0[0][1], line0[0][0] - line0[1][0]],
        [line1[1][1] - line1[0][1], line1[0][0] - line1[1][0]]
    ])
    b = np.array([np.cross(*line0), np.cross(*line1)])
    return np.linalg.solve(a, b)


class BezierCurve:
    """
    Abstract base class for bezier curves.

    Defined on domain [0, 1].
    """
    def __init__(self: Self, *points: Vec2):
        self.points: Vec2s = np.array(points)

    @staticmethod
    def interpolate_multiple(points: Vec2s, alpha: float) -> Vec2:
        n = len(points) - 1
        k = np.arange(n + 1)
        return np.sum((comb(n, k) * ((1.0 - alpha) ** (n - k) + alpha ** k))[:, None] * points, axis=0)

    def parametrize(self: Self, t: float) -> Vec2:
        return self.interpolate_multiple(self.points, t)

    def get_tangent_item(self: Self, t: float) -> tuple[Vec2, tuple[Vec2, Vec2]]:
        # Returns (P, (P0, P1)) where the curve tangents the line P0P1 at P.
        p0 = self.interpolate_multiple(self.points[:-1], t)
        p1 = self.interpolate_multiple(self.points[1:], t)
        p = (1.0 - t) * p0 + t * p1
        return (p, (p0, p1))

    def is_straight(self: Self) -> bool:
        points = self.points
        diffs = points[1:] - points[0]
        cross_prods = np.cross(points[0], diffs)
        return np.allclose(cross_prods, 0)

    def approximate_curve_by_quads(self: Self, *ts: float) -> list["QuadraticCurve"]:
        # TODO: Check if bisecting cubic curves will lead to zero division
        return [
            QuadraticCurve(p0, line_intersection(line0, line1), p1)
            for (p0, line0), (p1, line1) in it.pairwise(
                self.get_tangent_item(t) for t in [0.0, *ts, 1.0]
            )
        ]


class LinearCurve(BezierCurve):
    def __init__(self: Self, p0: Vec2, p1: Vec2):
        super().__init__(p0, p1)


class QuadraticCurve(BezierCurve):
    def __init__(self: Self, p0: Vec2, p1: Vec2, p2: Vec2):
        super().__init__(p0, p1, p2)


class CubicCurve(BezierCurve):
    def __init__(self: Self, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2):
        super().__init__(p0, p1, p2, p3)


@dataclass(repr=False)
class PathSegment:
    def __repr__(self: Self) -> str:
        return f"""{self.__class__.__name__}({', '.join(
            f"({vec[0]}, {vec[1]})"
            for vec in self.__dict__.values()
        )})"""

    __str__ = __repr__

@dataclass(repr=False)
class MoveTo(PathSegment):
    p1: Vec2

@dataclass(repr=False)
class LineTo(PathSegment):
    p1: Vec2

@dataclass(repr=False)
class QuadTo(PathSegment):
    p1: Vec2
    p2: Vec2

@dataclass(repr=False)
class CurveTo(PathSegment):
    p1: Vec2
    p2: Vec2
    p3: Vec2

@dataclass(repr=False)
class ClosePath(PathSegment):
    pass

@dataclass(repr=False)
class EndPath(PathSegment):
    pass


class Path:
    def __init__(self: Self, path: pathops.Path | str | None = None):
        self._path: pathops.Path = pathops.Path()
        if path is not None:
            self.set_path(path)

    def copy(self: Self) -> Self:
        return copy.copy(self)

    def get_path(self: Self) -> pathops.Path:
        return self._path

    def set_path(self: Self, path: pathops.Path | str) -> Self:
        if isinstance(path, pathops.Path):
            self._path = path
        elif isinstance(path, str):
            self._path = self.from_svg_path_str(path).get_path()
        else:
            raise TypeError(f"Cannot instance Path with argument: '{path}'")
        return self

    def get_pen(self: Self) -> pathops.PathPen:
        return self.get_path().getPen()

    # path commands

    def move_to(self: Self, p1: Vec2) -> Self:
        self.get_pen().moveTo(tuple(p1))
        return self

    def line_to(self: Self, p1: Vec2) -> Self:
        self.get_pen().lineTo(tuple(p1))
        return self

    def quad_to(self: Self, p1: Vec2, p2: Vec2) -> Self:
        self.get_pen().qCurveTo(tuple(p1), tuple(p2))
        return self

    def curve_to(self: Self, p1: Vec2, p2: Vec2, p3: Vec2) -> Self:
        self.get_pen().curveTo(tuple(p1), tuple(p2), tuple(p3))
        return self

    def close_path(self: Self) -> Self:
        self.get_pen().closePath()
        return self

    def end_path(self: Self) -> Self:
        self.get_pen().endPath()
        return self

    # methods inherited from pathops.Path

    def transform(self: Self, mat3: Mat3) -> Self:
        self.get_path().transform(
            scaleX=mat3[0][0],
            skewY=mat3[1][0],
            skewX=mat3[0][1],
            scaleY=mat3[1][1],
            translateX=mat3[0][2],
            translateY=mat3[1][2],
            perspectiveX=mat3[2][0],
            perspectiveY=mat3[2][1],
            perspectiveBias=mat3[2][2]
        )

    def reverse(self: Self) -> Self:
        self.get_path().reverse()
        return self

    def simplify(self: Self, fix_winding: bool = True, keep_starting_points: bool = True, clockwise: bool = False) -> Self:
        self.get_path().simplify(
            fix_winding=fix_winding,
            keep_starting_points=keep_starting_points,
            clockwise=clockwise
        )
        return self

    def clear(self: Self) -> Self:
        self.get_path().reset()
        return self

    def area(self: Self) -> float:
        return self.get_path().area

    def bounding_box(self: Self) -> tuple[float, float, float, float]:
        return self.get_path().bounds

    def contains(self: Self, point: Vec2) -> bool:
        return self.get_path().contains(tuple(point))

    def clockwise(self: Self) -> bool:
        return self.get_path().clockwise

    # structure

    def segments(self: Self) -> Generator[PathSegment, None, None]:
        for path_verb, params in self.get_path().segments:
            if path_verb == "moveTo":
                yield MoveTo(np.array(params[0]))
            elif path_verb == "lineTo":
                yield LineTo(np.array(params[0]))
            elif path_verb == "qCurveTo":
                yield QuadTo(np.array(params[0]), np.array(params[1]))
            elif path_verb == "curveTo":
                yield CurveTo(np.array(params[0]), np.array(params[1]), np.array(params[2]))
            elif path_verb == "closePath":
                yield ClosePath()
            elif path_verb == "endPath":
                yield EndPath()
            else:
                raise ValueError(f"Unsupported path verb: {path_verb}")

    def contours(self: Self) -> Generator[Self, None, None]:
        for contour in self.get_path().contours:
            yield Path(contour)

    def regions(self: Self) -> Generator[list[Self], None, None]:
        clockwise = self.clockwise()
        contours_cache = []
        for contour in self.contours():
            if not clockwise ^ contour.clockwise():
                if contours_cache:
                    yield contours_cache
                    contours_cache.clear()
            contours_cache.append(contour)
        if contours_cache:
            yield contours_cache

    def get_first_point(self: Self) -> Vec2:
        for segment in self.segments():
            if isinstance(segment, MoveTo):
                return segment.p1
        return np.zeros(2)

    def __bool__(self: Self) -> bool:
        return bool(self.get_path())

    def __repr__(self: Self) -> str:
        return f"Path({', '.join(segment.__repr__() for segment in self.segments())})"

    __str__ = __repr__

    # boolean operations

    @classmethod
    def union_multiple(cls, paths: list[Self]) -> Self:
        result = cls()
        pathops.union(
            [path.get_path() for path in paths],
            result.get_pen()
        )
        return result

    @classmethod
    def difference_multiple(cls, paths0: list[Self], paths1: list[Self]) -> Self:
        result = cls()
        pathops.difference(
            [path.get_path() for path in paths0],
            [path.get_path() for path in paths1],
            result.get_pen()
        )
        return result

    @classmethod
    def intersection_multiple(cls, paths0: list[Self], paths1: list[Self]) -> Self:
        result = cls()
        pathops.intersection(
            [path.get_path() for path in paths0],
            [path.get_path() for path in paths1],
            result.get_pen()
        )
        return result

    @classmethod
    def xor_multiple(cls, paths0: list[Self], paths1: list[Self]) -> Self:
        result = cls()
        pathops.xor(
            [path.get_path() for path in paths0],
            [path.get_path() for path in paths1],
            result.get_pen()
        )
        return result

    @classmethod
    def union(cls, *paths: Self) -> Self:
        return cls.union_multiple(list(paths))

    @classmethod
    def difference(cls, path0: Self, path1: Self) -> Self:
        return cls.difference_multiple([path0], [path1])

    @classmethod
    def intersection(cls, *paths: Self) -> Self:
        return reduce(lambda path0, path1: cls.intersection_multiple([path0], [path1]), paths)

    @classmethod
    def xor(cls, *paths: Self) -> Self:
        return reduce(lambda path0, path1: cls.xor_multiple([path0], [path1]), paths)

    def __add__(self: Self, path: Self) -> Self:
        return self.union(self, path)

    def __sub__(self: Self, path: Self) -> Self:
        return self.difference(self, path)

    def __mul__(self: Self, path: Self) -> Self:
        return self.intersection(self, path)

    def __xor__(self: Self, path: Self) -> Self:
        return self.xor(self, path)

    __or__ = __add__

    __and__ = __mul__

    def covers(self: Self, path: Self) -> bool:
        return not (path - self)

    def intersects(self: Self, path: Self) -> bool:
        return bool(path & self)

    # svg path string parser

    @classmethod
    def from_svg_path_str(cls, path_str: str) -> Self:
        path = cls()
        for segment in se.Path(path_str).segments():
            if isinstance(segment, se.Move):
                path.move_to(np.array(segment.end))
            elif isinstance(segment, se.Line):
                path.line_to(np.array(segment.end))
            elif isinstance(segment, se.QuadraticBezier):
                path.quad_to(np.array(segment.control), np.array(segment.end))
            elif isinstance(segment, se.CubicBezier):
                path.curve_to(np.array(segment.control1), np.array(segment.control2), np.array(segment.end))
            elif isinstance(segment, se.Arc):
                for quad_curve in segment.as_quad_curves():
                    path.quad_to(np.array(quad_curve.control), np.array(quad_curve.end))
            elif isinstance(segment, se.Close):
                path.close_path()
        return path

    # triangulate

    @classmethod
    def _triangulate_polygon_path(cls, path: Self) -> tuple[IArray, Vec2s]:
        def get_vertices_from_contour(contour: Path) -> Generator[Vec2, None, None]:
            yield contour.get_first_point()
            for segment in contour.segments():
                if isinstance(segment, (MoveTo, ClosePath, EndPath)):
                    continue
                elif isinstance(segment, LineTo):
                    yield segment.p1
                elif isinstance(segment, (QuadTo, CurveTo)):
                    if isinstance(segment, QuadTo):
                        yield segment.p2
                    else:
                        yield segment.p3
                    #raise ValueError(f"Curve segment unexpectedly occured: {segment}")
                else:
                    raise ValueError(f"Unsupported path segment: {segment}")

        def triangulate_region(region: list[Path]) -> tuple[IArray, Vec2s]:
            verts_lists = [
                list(get_vertices_from_contour(contour))
                for contour in region
            ]
            verts = np.array(list(it.chain(*verts_lists)))
            rings = np.array(list(it.accumulate(len(verts) for verts in verts_lists)))
            indices = triangulate_float32(verts, rings)
            return indices, verts

        if not path:
            return np.array([]), np.array([])
        indices_arrays = []
        verts_arrays = []
        index_offset = 0
        for region in path.regions():
            indices, verts = triangulate_region(region)
            indices_arrays.append(indices + index_offset)
            verts_arrays.append(verts)
            index_offset += len(verts)
        return np.concatenate(indices_arrays), np.concatenate(verts_arrays)

    @classmethod
    def triangulate(cls, path: Self) -> tuple[IArray, Array]:
        # Deal with sides that draw over themselves
        simplified_path = path.copy()
        simplified_path.simplify(clockwise=False)

        clip_frags = []
        mend_frags = []

        def clip_quads_from_quad(
            quad: QuadraticCurve, clockwise: bool, depth_limit: int
        ) -> Generator[tuple[bool, QuadraticCurve], None, None]:
            p0, p1, p2 = quad.points
            triangle = Path()
            triangle.move_to(p0)
            triangle.line_to(p1)
            triangle.line_to(p2)
            triangle.close_path()
            bent_inwards = clockwise ^ triangle.clockwise()
            quad_frag = Path()
            quad_frag.move_to(p0)
            quad_frag.quad_to(p1, p2)
            if bent_inwards:
                quad_frag.line_to(p1)
            quad_frag.close_path()

            if depth_limit == 0:
                # Use segments to approximate to prevent from infinite recursion
                mend_frags.append(triangle - quad_frag)
                return
            if simplified_path.covers(quad_frag) and all(not frag.intersects(quad_frag) for frag in clip_frags):
                clip_frags.append(quad_frag)
                yield (bent_inwards, quad)
                return
            for sub_quad in quad.approximate_curve_by_quads(0.5):
                yield from clip_quads_from_quad(sub_quad, clockwise, depth_limit - 1)

        def clip_quads_from_curve(
            curve: QuadraticCurve | CubicCurve, clockwise: bool
        ) -> Generator[tuple[bool, QuadraticCurve], None, None]:
            if curve.is_straight():
                null_frag = Path()
                null_frag.move_to(curve.parametrize(0.0))
                null_frag.line_to(curve.parametrize(1.0))
                null_frag.close_path()
                mend_frags.append(null_frag)
                return
            if isinstance(curve, QuadraticCurve):
                quads = [curve]
            else:  # cubic
                quads = curve.approximate_curve_by_quads(0.5)
            for quad in quads:
                yield from clip_quads_from_quad(quad, clockwise, 5)

        def clip_quads_from_contour(contour: Path) -> Generator[tuple[bool, QuadraticCurve], None, None]:
            point = contour.get_first_point()
            for segment in contour.segments():
                if isinstance(segment, (MoveTo, LineTo)):
                    point = segment.p1
                    continue
                elif isinstance(segment, QuadTo):
                    curve = QuadraticCurve(point, segment.p1, segment.p2)
                elif isinstance(segment, CurveTo):
                    curve = CubicCurve(point, segment.p1, segment.p2, segment.p3)
                elif isinstance(segment, ClosePath):
                    point = contour.get_first_point()
                    continue
                elif isinstance(segment, EndPath):
                    continue
                else:
                    raise ValueError(f"Unsupported path segment: {segment}")
                yield from clip_quads_from_curve(curve, contour.clockwise())

        def clip_quads_from_path(path: Path) -> Generator[tuple[bool, QuadraticCurve], None, None]:
            for contour in path.contours():
                yield from clip_quads_from_contour(contour)

        flagged_quads = list(clip_quads_from_path(simplified_path))
        polygon_path = cls.difference_multiple([simplified_path, *mend_frags], clip_frags)
        print(simplified_path)
        print(clip_frags)
        print(polygon_path)
        indices, verts = cls._triangulate_polygon_path(polygon_path)

        all_indices = np.append(indices, np.arange(len(verts), len(verts) + len(flagged_quads) * 3))
        all_verts = np.array([
            (vert[0], vert[1], 0)
            for vert in verts
        ] + [
            (vert[0], vert[1], 0 if i != 1 else -1 if bent_inwards else 1)
            for bent_inwards, quad in flagged_quads
            for i, vert in enumerate(quad.points)
        ])
        print(all_indices, all_verts)
        return all_indices, all_verts
