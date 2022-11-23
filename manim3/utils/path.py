from abc import abstractmethod
from typing import Callable, Generic, TypeVar

import numpy as np
import scipy.integrate
import scipy.interpolate
import skia

from ..utils.lazy import expire_properties, LazyMeta, lazy_property
from ..custom_typing import *


__all__ = ["Path"]


T = TypeVar("T", bound="CurveInterpolantBase")


def interp1d(x: FloatArrayType, y: FloatArrayType, tol: Real = 1e-6, **kwargs) -> scipy.interpolate.interp1d:
    # Append one more sample point at each side in order to prevent from floating error.
    # Also solves the issue where we have only one sample, while the original function requires at least two.
    # Assumed that `x` is already sorted.
    new_x = np.array([x[0] - tol, *x, x[-1] + tol])
    new_y = np.array([y[0], *y, y[-1]])
    return scipy.interpolate.interp1d(new_x, new_y, **kwargs)


class CurveInterpolantBase(metaclass=LazyMeta):
    @lazy_property
    @abstractmethod
    def a_final(self: Self) -> float:
        raise NotImplementedError

    @lazy_property
    @abstractmethod
    def l_final(self: Self) -> float:
        raise NotImplementedError

    @abstractmethod
    def a_to_p(self: Self, a: Real) -> Vector2Type:
        raise NotImplementedError

    @abstractmethod
    def a_to_l(self: Self, a: Real) -> float:
        raise NotImplementedError

    @abstractmethod
    def l_to_a(self: Self, l: Real) -> float:
        raise NotImplementedError

    def a_ratio_to_p(self: Self, a_ratio: Real) -> Vector2Type:
        return self.a_to_p(a_ratio * self.a_final)

    def a_ratio_to_l_ratio(self: Self, a_ratio: Real) -> float:
        try:
            return self.a_to_l(a_ratio * self.a_final) / self.l_final
        except ZeroDivisionError:
            return 0.0

    def l_ratio_to_a_ratio(self: Self, l_ratio: Real) -> float:
        try:
            return self.l_to_a(l_ratio * self.l_final) / self.a_final
        except ZeroDivisionError:
            return 0.0

    @abstractmethod
    def partial_by_a(self: Self, a: Real) -> Self:
        raise NotImplementedError

    def partial_by_l(self: Self, l: Real) -> Self:
        return self.partial_by_a(self.l_to_a(l))

    def partial_by_a_ratio(self: Self, a_ratio: Real) -> Self:
        return self.partial_by_a(a_ratio * self.a_final)

    def partial_by_l_ratio(self: Self, l_ratio: Real) -> Self:
        return self.partial_by_l(l_ratio * self.l_final)


class CurveInterpolant(CurveInterpolantBase, Generic[T]):
    """
    A general tree-structured curve interpolant.

    Typically, a curve has an alpha parametrization (`a`, defined on [0, `a_final`])
    and a length parametrization (`l`, defoned on [0, `l_final`]).
    A bunch of translation methods are defined.
    """
    def __init__(self: Self, children: list[T] | None = None):
        if children is None:
            children = []
        self._children: list[T] = children
        #a_knots = np.zeros(1)
        #l_knots = np.zeros(1)
        #self._a_knots: FloatArrayType = a_knots
        #self._l_knots: FloatArrayType = l_knots
        #self._a_interpolator: Callable[[Real], tuple[int, float]] = self.integer_interpolator(a_knots)
        #self._l_interpolator: Callable[[Real], tuple[int, float]] = self.integer_interpolator(l_knots)
        #self._a_final: float = a_knots[-1]
        #self._l_final: float = l_knots[-1]
        #self.data_require_updating: bool = True
        #self.update_data()

    #def update_data(self: Self) -> Self:
    #    if self.data_require_updating:
    #        #children = self.get_updated_children()
    #        #self._children = children
    #        children = self._children
    #        a_knots = np.insert(np.cumsum([child.a_final for child in children]), 0, 0.0)
    #        l_knots = np.insert(np.cumsum([child.l_final for child in children]), 0, 0.0)
    #        self._a_knots = a_knots
    #        self._l_knots = l_knots
    #        self._a_interpolator = self.integer_interpolator(a_knots)
    #        self._l_interpolator = self.integer_interpolator(l_knots)
    #        self._a_final = a_knots[-1]
    #        self._l_final = l_knots[-1]
    #        self.data_require_updating = False
    #    return self

    #def get_updated_children(self: Self) -> list[T]:
    #    return self._children

    @lazy_property
    def children(self: Self) -> list[T]:
        #self.update_data()
        return self._children

    @lazy_property
    def a_knots(self: Self) -> FloatArrayType:
        #self.update_data()
        return np.insert(np.cumsum([child.a_final for child in self.children]), 0, 0.0)

    @lazy_property
    def l_knots(self: Self) -> FloatArrayType:
        #self.update_data()
        return np.insert(np.cumsum([child.l_final for child in self.children]), 0, 0.0)

    @lazy_property
    def a_final(self: Self) -> float:
        #self.update_data()
        return self.a_knots[-1]

    @lazy_property
    def l_final(self: Self) -> float:
        #self.update_data()
        return self.l_knots[-1]

    @lazy_property
    def a_interpolator(self: Self) -> Callable[[Real], tuple[int, float]]:
        return self.integer_interpolator(self.a_knots)

    @lazy_property
    def l_interpolator(self: Self) -> Callable[[Real], tuple[int, float]]:
        return self.integer_interpolator(self.l_knots)

    def a_interpolate(self: Self, a: Real) -> tuple[int, float]:
        #self.update_data()
        return self.a_interpolator(a)

    def l_interpolate(self: Self, l: Real) -> tuple[int, float]:
        #self.update_data()
        return self.l_interpolator(l)

    def a_to_p(self: Self, a: Real) -> Vector2Type:
        i, a_remainder = self.a_interpolate(a)
        assert a_remainder
        return self.children[i].a_to_p(a_remainder)

    def a_to_l(self: Self, a: Real) -> float:
        i, a_remainder = self.a_interpolate(a)
        l = self.l_knots[i]
        if a_remainder:
            l += self.children[i].a_to_l(a_remainder)
        return l

    def l_to_a(self: Self, l: Real) -> float:
        i, l_remainder = self.l_interpolate(l)
        a = self.a_knots[i]
        if l_remainder:
            a += self.children[i].l_to_a(l_remainder)
        return a

    def partial_by_a(self: Self, a: Real) -> Self:
        i, a_remainder = self.a_interpolate(a)
        children = self.children[:i]
        if a_remainder:
            children.append(self.children[i].partial_by_a(a_remainder))
        return self.__class__(children=children)

    @staticmethod
    def integer_interpolator(array: FloatArrayType) -> Callable[[Real], tuple[int, float]]:
        def wrapped(target: Real) -> tuple[int, float]:
            """
            Assumed that `array` is already sorted, and that `array[0] <= target <= array[-1]`
            If `target == array[0]`, returns `(0, 0.0)`.
            Otherwise, returns `(i, target - array[i])` such that
            `0 <= i < len(array) - 1` and `array[i] < target <= array[i + 1]`.
            """
            index = int(interp1d(array, np.array(range(len(array))) - 1.0, kind="next")(target))
            if index == -1:
                return 0, 0.0
            return index, target - array[index]
        return wrapped


class BezierCurve(CurveInterpolantBase):
    """
    Bezier curves defined on domain [0, 1].
    """
    def __init__(self: Self, points: Vector2ArrayType):
        self._points: Vector2ArrayType = points
        super().__init__()

    @lazy_property
    def points(self: Self) -> Vector2ArrayType:
        return self._points

    @lazy_property
    def order(self: Self) -> int:
        return len(self.points) - 1

    @lazy_property
    def gamma(self: Self) -> scipy.interpolate.BSpline:
        order = self.order
        return scipy.interpolate.BSpline(
            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
            c=self.points,
            k=order
        )

    @lazy_property
    def a_samples(self: Self) -> FloatArrayType:
        segments = 16 if self.order > 1 else 1
        return np.linspace(0.0, 1.0, segments + 1)

    @lazy_property
    def l_samples(self: Self) -> FloatArrayType:
        p_samples = self.gamma(self.a_samples)
        segment_lengths = np.linalg.norm(p_samples[1:] - p_samples[:-1], axis=1)
        return np.insert(np.cumsum(segment_lengths), 0, 0.0)

    @lazy_property
    def a_l_interp(self: Self) -> scipy.interpolate.interp1d:
        return interp1d(self.a_samples, self.l_samples)

    @lazy_property
    def l_a_interp(self: Self) -> Callable[[Real], Real]:
        return interp1d(self.l_samples, self.a_samples)

    @lazy_property
    def a_final(self: Self) -> float:
        return 1.0

    @lazy_property
    def l_final(self: Self) -> float:
        return self.a_l_interp(self.a_final)

    def a_to_p(self: Self, a: Real) -> Vector2Type:
        return self.gamma(a)

    def a_to_l(self: Self, a: Real) -> float:
        return self.a_l_interp(a)

    def l_to_a(self: Self, l: Real) -> float:
        return self.l_a_interp(l)

    def partial_by_a(self: Self, a: Real) -> Self:
        return BezierCurve(np.array([
            BezierCurve(self.points[:n]).a_to_p(a)
            for n in range(1, self.order + 2)
        ]))

    def rise_order_to(self: Self, new_order: int) -> Self:
        new_points = self.points
        for n in range(self.order + 1, new_order + 1):
            mat = np.zeros((n + 1, n))
            mat[(np.arange(n), np.arange(n))] = np.arange(n, 0, -1) / n
            mat[(np.arange(n) + 1, np.arange(n))] = np.arange(1, n + 1) / n
            new_points = mat @ new_points
        return BezierCurve(new_points)


class Contour(CurveInterpolant[BezierCurve]):
    """
    A list of chained Bezier curves
    """
    pass


class Contours(CurveInterpolant[Contour]):
    """
    A list of contours, either open or closed
    """
    pass


class Path(metaclass=LazyMeta):
    """
    A list of contours, either open or closed
    """
    def __init__(
        self: Self,
        path: skia.Path | Contours | None = None
        #children: list[Contour] | None = None
    ):
        if isinstance(path, skia.Path):
            skia_path = path
        elif isinstance(path, Contours):
            skia_path = self.get_skia_path_by_contours(path)
        elif path is None:
            skia_path = skia.Path()
        else:
            raise ValueError(f"Unsupported path type: {type(path)}")

        self.skia_path: skia.Path = skia_path
        #self._contours: Contours = Contours()
        #self.contours_require_updating: bool = True

    @lazy_property
    def _skia_path(self: Self) -> skia.Path:
        return self.skia_path

    @_skia_path.setter
    def _skia_path(self: Self, arg: skia.Path) -> None:
        pass

    @lazy_property
    def contours(self: Self) -> Contours:
        return self.get_contours_by_skia_path(self._skia_path)
        #if self.contours_require_updating:
        #    self._contours = self.get_contours_by_skia_path(self.skia_path)
        #    self.contours_require_updating = False
        #return self._contours

    #def get_updated_children(self: Self) -> list[Contour]:
    #    return self.get_contours_by_skia_path(self.skia_path)

    @staticmethod
    def get_contours_by_skia_path(path: skia.Path) -> Contours:
        contours = []
        contour = []
        iterator = iter(path)
        verb, points = iterator.next()
        while verb != skia.Path.kDone_Verb:
            if verb == skia.Path.Verb.kMove_Verb:
                pass
            elif verb in (
                skia.Path.Verb.kLine_Verb,
                skia.Path.Verb.kQuad_Verb,
                skia.Path.Verb.kCubic_Verb
            ):
                contour.append(BezierCurve(np.array([
                    np.array(list(point)) for point in points
                ])))
            elif verb == skia.Path.Verb.kConic_Verb:
                # Approximate per conic curve with 8 quads
                quad_points = skia.Path.ConvertConicToQuads(*points, iterator.conicWeight(), 3)
                for i in range(0, len(quad_points), 2):
                    contour.append(BezierCurve(np.array([
                        np.array(list(point)) for point in quad_points[i:i + 3]
                    ])))
            elif verb == skia.Path.Verb.kClose_Verb:
                if contour:
                    contours.append(Contour(contour))
                    contour = []
            else:
                raise ValueError
            verb, points = iterator.next()
        if contour:
            contours.append(Contour(contour))
        return Contours(contours)

    @staticmethod
    def get_skia_path_by_contours(contours: Contours) -> skia.Path:
        path = skia.Path()
        for contour in contours.children:
            path.moveTo(*contour.children[0].points[0])
            for curve in contour.children:
                points = curve.points
                len_points = len(points)
                if len_points == 2:
                    path.lineTo(*points[1])
                elif len_points == 3:
                    path.quadTo(*points[1], *points[2])
                elif len_points == 4:
                    path.cubicTo(*points[1], *points[2], *points[3])
                else:
                    raise ValueError
            path.close()
        return path

    @expire_properties("_skia_path")
    def move_to(self: Self, point: Vector2Type) -> Self:
        #self.contours_require_updating = True
        self.skia_path.moveTo(skia.Point(*point))
        #self._skia_path = self.skia_path
        return self

    @expire_properties("_skia_path")
    def line_to(self: Self, point: Vector2Type) -> Self:
        #self.contours_require_updating = True
        self.skia_path.lineTo(skia.Point(*point))
        #self._skia_path = self.skia_path
        return self

    @expire_properties("_skia_path")
    def quad_to(self: Self, control_point: Vector2Type, point: Vector2Type) -> Self:
        #self.contours_require_updating = True
        self.skia_path.quadTo(skia.Point(*control_point), skia.Point(*point))
        #self._skia_path = self.skia_path
        return self

    @expire_properties("_skia_path")
    def cubic_to(self: Self, control_point_0: Vector2Type, control_point_1: Vector2Type, point: Vector2Type) -> Self:
        #self.contours_require_updating = True
        self.skia_path.cubicTo(skia.Point(*control_point_0), skia.Point(*control_point_1), skia.Point(*point))
        #self._skia_path = self.skia_path
        return self

    @expire_properties("_skia_path")
    def conic_to(self: Self, control_point: Vector2Type, point: Vector2Type, weight: Real) -> Self:
        #self.contours_require_updating = True
        self.skia_path.conicTo(skia.Point(*control_point), skia.Point(*point), weight)
        #self._skia_path = self.skia_path
        return self

    @expire_properties("_skia_path")
    def close_path(self: Self) -> Self:
        #self.contours_require_updating = True
        self.skia_path.close()
        #self._skia_path = self.skia_path
        return self

    @lazy_property
    def a_final(self: Self) -> float:
        return self.contours.a_final

    @lazy_property
    def l_final(self: Self) -> float:
        return self.contours.l_final

    def a_to_p(self: Self, a: Real) -> Vector2Type:
        return self.contours.a_to_p(a)

    def a_to_l(self: Self, a: Real) -> float:
        return self.contours.a_to_l(a)

    def l_to_a(self: Self, l: Real) -> float:
        return self.contours.l_to_a(l)

    def a_ratio_to_p(self: Self, a_ratio: Real) -> Vector2Type:
        return self.contours.a_ratio_to_p(a_ratio)

    def a_ratio_to_l_ratio(self: Self, a_ratio: Real) -> float:
        return self.contours.a_ratio_to_l_ratio(a_ratio)

    def l_ratio_to_a_ratio(self: Self, l_ratio: Real) -> float:
        return self.contours.l_ratio_to_a_ratio(l_ratio)

    def partial_by_a(self: Self, a: Real) -> Self:
        return Path(self.contours.partial_by_a(a))

    def partial_by_l(self: Self, l: Real) -> Self:
        return Path(self.contours.partial_by_l(l))

    def partial_by_a_ratio(self: Self, a_ratio: Real) -> Self:
        return Path(self.contours.partial_by_a_ratio(a_ratio))

    def partial_by_l_ratio(self: Self, l_ratio: Real) -> Self:
        return Path(self.contours.partial_by_l_ratio(l_ratio))
