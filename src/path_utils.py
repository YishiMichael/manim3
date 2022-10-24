from __future__ import annotations

from abc import ABC, abstractmethod
import itertools as it

from mapbox_earcut import triangulate_float32
import numpy as np
import pathops

from utils.arrays import Vec2

_PathVerb = pathops.PathVerb


def _interpolate(p0: Vec2, p1: Vec2, t: float) -> Vec2:
    return p0.lerp(p1, t)


class _Curve(ABC):
    @abstractmethod
    def _parametrize(self, t: float) -> Vec2:
        raise NotImplementedError

    def get_points(self) -> list[Vec2]:
        return [
            self._parametrize(t)
            for t in np.linspace(0.0, 1.0, self.segment_divisions + 1)
        ]

    @property
    def segment_divisions(self):
        return 16


class _LineCurve(_Curve):
    def __init__(self, p0: Vec2, p1: Vec2):
        self._p0 = p0
        self._p1 = p1

    def _parametrize(self, t: float) -> Vec2:
        p0 = self._p0
        p1 = self._p1
        return _interpolate(p0, p1, t)

    @property
    def segment_divisions(self):
        return 1  # Simplifies computation


class _QuadraticBezierCurve(_Curve):
    def __init__(self, p0: Vec2, p1: Vec2, p2: Vec2):
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        super().__init__()

    def _parametrize(self, t: float) -> Vec2:
        p0 = self._p0
        p1 = self._p1
        p2 = self._p2
        a0 = _interpolate(p0, p1, t)
        a1 = _interpolate(p1, p2, t)
        return _interpolate(a0, a1, t)


class _CubicBezierCurve(_Curve):
    def __init__(self, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2):
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3
        super().__init__()

    def _parametrize(self, t: float) -> Vec2:
        p0 = self._p0
        p1 = self._p1
        p2 = self._p2
        p3 = self._p3
        a0 = _interpolate(p0, p1, t)
        a1 = _interpolate(p1, p2, t)
        a2 = _interpolate(p2, p3, t)
        b0 = _interpolate(a0, a1, t)
        b1 = _interpolate(a1, a2, t)
        return _interpolate(b0, b1, t)


def get_points_from_contour(contour: pathops.Path) -> list[Vec2]:
    points = []
    p0 = Vec2()
    for path_verb, params in contour:
        if path_verb == _PathVerb.MOVE:
            point = Vec2(*params[0])
        else:
            if path_verb == _PathVerb.LINE:
                p1 = Vec2(*params[0])
                curve = _LineCurve(p0, p1)
            elif path_verb == _PathVerb.QUAD:
                p1 = Vec2(*params[0])
                p2 = Vec2(*params[1])
                curve = _QuadraticBezierCurve(p0, p1, p2)
            elif path_verb == _PathVerb.CUBIC:
                p1 = Vec2(*params[0])
                p2 = Vec2(*params[1])
                p3 = Vec2(*params[2])
                curve = _CubicBezierCurve(p0, p1, p2, p3)
            elif path_verb == _PathVerb.CLOSE:
                p1 = points[0]
                curve = _LineCurve(p0, p1)
            else:
                raise ValueError(f"Unsupported path verb: {path_verb}")
            sample_points = curve.get_points()
            points.extend(sample_points[:-1])
            point = sample_points[-1]
        p0 = point
    return points


def triangulate_path(path: pathops.Path) -> tuple[list[Vec2], list[int]]:
    all_verts = []
    all_indices = []
    contours = []

    def digest_contours():
        verts = list(it.chain(*contours))
        if not verts:
            return
        rings = np.array(list(it.accumulate(len(contour) for contour in contours)))
        indices = triangulate_float32(np.array([list(vec) for vec in verts]), rings)
        index_offset = len(all_verts)
        all_verts.extend(verts)
        all_indices.extend(indices + index_offset)
        contours.clear()

    # simplify twice in case of complicated paths
    path_ = pathops.simplify(pathops.simplify(path), clockwise=False)
    for contour in path_.contours:
        points = get_points_from_contour(contour)
        if not contour.clockwise:
            digest_contours()
        contours.append(points)
    digest_contours()
    return all_verts, all_indices
