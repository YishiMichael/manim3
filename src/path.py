from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import itertools as it
from typing import Any, Generic, Iterator, Literal, TypeVar, Union

import numpy as np
from utils.arrays import Vec2


T = TypeVar("T")
Self = Any  # To be removed in py 3.11


def vec2_cross(vec_0: Vec2, vec1: Vec2) -> float:
    return vec_0.x * vec1.y - vec_0.y * vec1.x


@dataclass
class AbstractKeyFrame(Generic[T]):
    timestamp: float
    frame: T


class AbstractKeyFramesContainer(ABC, Generic[T]):
    def __init__(self, keyframes: list[AbstractKeyFrame[T]]):
        # Should contain at least one frame
        # Should ensure ts increasing
        self._keyframes: list[AbstractKeyFrame[T]] = keyframes
        self._period: float = keyframes[-1].timestamp - keyframes[0].timestamp

    @abstractmethod
    @staticmethod
    def _interpolate(frame_0: T, frame_1: T, t: float) -> T:
        pass

    def interpolate(self, t: float) -> T:
        return self._get_partial_frames(t)[-1]

    def _get_all_frames(self) -> list[T]:
        return [keyframe.frame for keyframe in self._keyframes]

    def _get_partial_frames(self, t: float) -> list[T]:
        target_timestamp = t * self._period
        keyframes = self._keyframes
        timestamps = [keyframe.timestamp for keyframe in keyframes]
        index = self._find_insertion_index(target_timestamp, timestamps)
        if index == 0:
            return [keyframes[0].frame]
        prev_kf = keyframes[index - 1]
        next_kf = keyframes[index]
        try:
            sub_t = (target_timestamp - prev_kf.timestamp) \
                / (next_kf.timestamp - prev_kf.timestamp)
        except ZeroDivisionError:
            sub_t = 0.0
        prev_frames = [kf.frame for kf in keyframes[:index]]
        curr_frame = self._interpolate(prev_kf.frame, next_kf.frame, sub_t)
        return [*prev_frames, curr_frame]

    @staticmethod
    def _find_insertion_index(target: float, vals: list[float]) -> int:
        """
        Returns i such that 0 <= i <= len(vals) and
        vals[i - 1] < target <= vals[i]
        """
        if not vals:
            return 0
        # Binary search
        left = 0
        right = len(vals) - 1
        while left <= right:
            mid = (right - left) // 2 + left
            if vals[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

@dataclass
class Box2:
    x_span: tuple[float, float]
    y_span: tuple[float, float]


class Polyline(AbstractKeyFramesContainer[Vec2]):
    """
    2D polyline, immutable
    """
    def __init__(self, points: list[Vec2]):
        super().__init__([
            AbstractKeyFrame(timestamp=accumulated_length, frame=point)
            for point, accumulated_length in zip(points, [
                0.0, *it.accumulate(
                    abs(p1 - p0)
                    for p0, p1 in it.pairwise(points)
                )
            ])
        ])

    @staticmethod
    def _interpolate(p0: Vec2, p1: Vec2, t: float) -> Vec2:
        return p0.lerp(p1, t)

    def get_all_points(self) -> list[Vec2]:
        return self._get_all_frames()

    def get_partial_points(self, t: float) -> list[Vec2]:
        return self._get_partial_frames(t)

    def get_bounding_box(self) -> Box2:
        points = self.get_all_points()
        xs = [point.x for point in points]
        ys = [point.y for point in points]
        return Box2(
            x_span=(min(xs), max(xs)),
            y_span=(min(ys), max(xs))
        )

    def get_area(self) -> float:
        """
        Calculate area of the contour polygon

        Port from ShapeUtils.area(), three/src/extras/ShapeUtils.js
        """
        points = self.get_all_points()
        return sum([vec2_cross(points[-1], points[0]), *(
            vec2_cross(p0, p1) for p0, p1 in it.pairwise(points)
        )]) / 2

    def is_clockwise(self) -> bool:
        return self.get_area() < 0


class Curve(Polyline):
    """
    Abstract base class for 2D curves, immutable

    Approximate curves with line segments
    """
    SEGMENT_DIVISIONS = 16

    def __init__(self):
        points = [
            self._parametrize(t)
            for t in np.linspace(0.0, 1.0, self.SEGMENT_DIVISIONS + 1)
        ]
        super().__init__(points)

    @abstractmethod
    def _parametrize(self, t: float) -> Vec2:
        pass


class LineCurve(Curve):
    SEGMENT_DIVISIONS = 1  # Simplifies computation

    def __init__(self, p0: Vec2, p1: Vec2):
        self._p0 = p0
        self._p1 = p1
        super().__init__()

    def _parametrize(self, t: float) -> Vec2:
        interpolate = self._interpolate
        p0 = self._p0
        p1 = self._p1
        return interpolate(p0, p1, t)


class QuadraticBezierCurve(Curve):
    def __init__(self, p0: Vec2, p1: Vec2, p2: Vec2):
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        super().__init__()

    def _parametrize(self, t: float) -> Vec2:
        interpolate = self._interpolate
        p0 = self._p0
        p1 = self._p1
        p2 = self._p2
        a0 = interpolate(p0, p1, t)
        a1 = interpolate(p1, p2, t)
        return interpolate(a0, a1, t)


class CubicBezierCurve(Curve):
    def __init__(self, p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2):
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        self._p3 = p3
        super().__init__()

    def _parametrize(self, t: float) -> Vec2:
        interpolate = self._interpolate
        p0 = self._p0
        p1 = self._p1
        p2 = self._p2
        p3 = self._p3
        a0 = interpolate(p0, p1, t)
        a1 = interpolate(p1, p2, t)
        a2 = interpolate(p2, p3, t)
        b0 = interpolate(a0, a1, t)
        b1 = interpolate(a1, a2, t)
        return interpolate(b0, b1, t)


class Polylines:
    def __init__(self, polylines: list[Polyline]):
        self._polylines = polylines

    def __iter__(self) -> Iterator[Polyline]:
        return iter(self._polylines)

    def get_bounding_box(self) -> Box2:
        sub_boxes = [polyline.get_bounding_box() for polyline in self._polylines]
        xmins, xmaxs = zip(*(box.x_span for box in sub_boxes))
        ymins, ymaxs = zip(*(box.y_span for box in sub_boxes))
        return Box2(
            x_span=(min(xmins), max(xmaxs)),
            y_span=(min(ymins), max(ymaxs))
        )


class PathCommandSymbols(Enum):
    M = 0
    L = 1
    Q = 2
    C = 3
    Z = 4


PathCommand = Union[
    tuple[Literal[PathCommandSymbols.M], Vec2],
    tuple[Literal[PathCommandSymbols.L], Vec2],
    tuple[Literal[PathCommandSymbols.Q], Vec2, Vec2],
    tuple[Literal[PathCommandSymbols.C], Vec2, Vec2, Vec2],
    tuple[Literal[PathCommandSymbols.Z]]
]


class Path:
    def __init__(self):
        self._path_commands: list[PathCommand] = []
        #self._connected_curve_lists: list[list[Curve]] = []
        #self._curr_curve_list: list[Curve] = []
        #self._curr_point: Vec2 = init_point

    def m(self, point: Vec2) -> Self:
        self._path_commands.append((PathCommandSymbols.M, point))
        return self

    def l(self, point: Vec2) -> Self:
        self._path_commands.append((PathCommandSymbols.L, point))
        return self

    def q(self, control: Vec2, point: Vec2) -> Self:
        self._path_commands.append((PathCommandSymbols.Q, control, point))
        return self

    def c(self, control_0: Vec2, control_1: Vec2, point: Vec2) -> Self:
        self._path_commands.append((PathCommandSymbols.C, control_0, control_1, point))
        return self

    def z(self) -> Self:
        self._path_commands.append((PathCommandSymbols.Z,))
        return self

    def _to_polylines(self) -> Polylines:
        result = []
        polyline_points = []
        curr_point = Vec2()
        for symbol, *args in self._path_commands:
            if symbol == PathCommandSymbols.M:
                point, = args
                result.append(Polyline(polyline_points))
                polyline_points.clear()
                polyline_points.append(point)
                curr_point = point
                continue
            elif symbol == PathCommandSymbols.L:
                point, = args
                curve = LineCurve(curr_point, point)
            elif symbol == PathCommandSymbols.Q:
                control, point = args
                curve = QuadraticBezierCurve(curr_point, control, point)
            elif symbol == PathCommandSymbols.C:
                control_0, control_1, point = args
                curve = CubicBezierCurve(curr_point, control_0, control_1, point)
            elif symbol == PathCommandSymbols.Z:
                point = polyline_points[0]
                curve = LineCurve(curr_point, point)
            else:
                raise ValueError("Cannot handle path command: %s", symbol)
            polyline_points.extend(curve.get_all_points()[1:])
            curr_point = point
        return Polylines(result)
