__all__ = ["Path"]


from abc import abstractmethod
from typing import Callable, Generic, TypeVar

import numpy as np
import scipy.interpolate
import skia

from ..utils.lazy import LazyBase, lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..custom_typing import *


_T = TypeVar("_T", bound="CurveInterpolantBase")


def interp1d(x: FloatArrayType, y: FloatArrayType, tol: Real = 1e-6, **kwargs) -> scipy.interpolate.interp1d:
    # Append one more sample point at each side in order to prevent from floating error.
    # Also solves the issue where we have only one sample, while the original function requires at least two.
    # Assumed that `x` is already sorted.
    new_x = np.array([x[0] - tol, *x, x[-1] + tol])
    new_y = np.array([y[0], *y, y[-1]])
    return scipy.interpolate.interp1d(new_x, new_y, **kwargs)


class CurveInterpolantBase(LazyBase):
    @property
    @abstractmethod
    def _a_final_(self) -> float:
        pass

    @property
    @abstractmethod
    def _l_final_(self) -> float:
        pass

    @abstractmethod
    def a_to_p(self, a: Real) -> Vector2Type:
        pass

    @abstractmethod
    def a_to_l(self, a: Real) -> float:
        pass

    @abstractmethod
    def l_to_a(self, l: Real) -> float:
        pass

    def a_ratio_to_p(self, a_ratio: Real) -> Vector2Type:
        return self.a_to_p(a_ratio * self._a_final_)

    def a_ratio_to_l_ratio(self, a_ratio: Real) -> float:
        try:
            return self.a_to_l(a_ratio * self._a_final_) / self._l_final_
        except ZeroDivisionError:
            return 0.0

    def l_ratio_to_a_ratio(self, l_ratio: Real) -> float:
        try:
            return self.l_to_a(l_ratio * self._l_final_) / self._a_final_
        except ZeroDivisionError:
            return 0.0

    @abstractmethod
    def partial_by_a(self, a: Real):
        pass

    def partial_by_l(self, l: Real):
        return self.partial_by_a(self.l_to_a(l))

    def partial_by_a_ratio(self, a_ratio: Real):
        return self.partial_by_a(a_ratio * self._a_final_)

    def partial_by_l_ratio(self, l_ratio: Real):
        return self.partial_by_l(l_ratio * self._l_final_)


class CurveInterpolant(Generic[_T], CurveInterpolantBase):
    """
    A general tree-structured curve interpolant.

    Typically, a curve has an alpha parametrization (`a`, defined on [0, `a_final`])
    and a length parametrization (`l`, defoned on [0, `l_final`]).
    A bunch of translation methods are defined.
    """
    def __init__(self, children: list[_T] | None = None):
        super().__init__()
        if children is not None:
            self._children_.extend(children)

    @lazy_property_initializer
    @staticmethod
    def _children_() -> list[_T]:
        return []

    @lazy_property
    @staticmethod
    def _a_knots_(children: list[_T]) -> FloatArrayType:
        return np.insert(np.cumsum([child._a_final_ for child in children]), 0, 0.0)

    @lazy_property
    @staticmethod
    def _l_knots_(children: list[_T]) -> FloatArrayType:
        return np.insert(np.cumsum([child._l_final_ for child in children]), 0, 0.0)

    @lazy_property
    @staticmethod
    def _a_final_(a_knots: FloatArrayType) -> float:
        return a_knots[-1]

    @lazy_property
    @staticmethod
    def _l_final_(l_knots: FloatArrayType) -> float:
        return l_knots[-1]

    @lazy_property
    @staticmethod
    def _a_interpolator_(a_knots: FloatArrayType) -> Callable[[Real], tuple[int, float]]:
        return CurveInterpolant.integer_interpolator(a_knots)

    @lazy_property
    @staticmethod
    def _l_interpolator_(l_knots: FloatArrayType) -> Callable[[Real], tuple[int, float]]:
        return CurveInterpolant.integer_interpolator(l_knots)

    def a_interpolate(self, a: Real) -> tuple[int, float]:
        return self._a_interpolator_(a)

    def l_interpolate(self, l: Real) -> tuple[int, float]:
        return self._l_interpolator_(l)

    def a_to_p(self, a: Real) -> Vector2Type:
        i, a_remainder = self.a_interpolate(a)
        assert a_remainder
        return self._children_[i].a_to_p(a_remainder)

    def a_to_l(self, a: Real) -> float:
        i, a_remainder = self.a_interpolate(a)
        l = self._l_knots_[i]
        if a_remainder:
            l += self._children_[i].a_to_l(a_remainder)
        return l

    def l_to_a(self, l: Real) -> float:
        i, l_remainder = self.l_interpolate(l)
        a = self._a_knots_[i]
        if l_remainder:
            a += self._children_[i].l_to_a(l_remainder)
        return a

    def partial_by_a(self, a: Real):
        i, a_remainder = self.a_interpolate(a)
        children = self._children_[:i]
        if a_remainder:
            children.append(self._children_[i].partial_by_a(a_remainder))
        return self.__class__(children=children)

    @classmethod
    def integer_interpolator(cls, array: FloatArrayType) -> Callable[[Real], tuple[int, float]]:
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
    def __init__(self, points: Vector2ArrayType):
        super().__init__()
        self._points_ = points

    @lazy_property_initializer_writable
    @staticmethod
    def _points_() -> Vector2ArrayType:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _order_(points: Vector2ArrayType) -> int:
        return len(points) - 1

    @lazy_property
    @staticmethod
    def _gamma_(order: int, points: Vector2ArrayType) -> scipy.interpolate.BSpline:
        return scipy.interpolate.BSpline(
            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
            c=points,
            k=order
        )

    @lazy_property
    @staticmethod
    def _a_samples_(order: int) -> FloatArrayType:
        num_samples = 9 if order > 1 else 2
        return np.linspace(0.0, 1.0, num_samples)

    @lazy_property
    @staticmethod
    def _l_samples_(gamma: scipy.interpolate.BSpline, a_samples: FloatArrayType) -> FloatArrayType:
        p_samples = gamma(a_samples)
        segment_lengths = np.linalg.norm(p_samples[1:] - p_samples[:-1], axis=1)
        return np.insert(np.cumsum(segment_lengths), 0, 0.0)

    @lazy_property
    @staticmethod
    def _a_l_interp_(a_samples: FloatArrayType, l_samples: FloatArrayType) -> scipy.interpolate.interp1d:
        return interp1d(a_samples, l_samples)

    @lazy_property
    @staticmethod
    def _l_a_interp_(l_samples: FloatArrayType, a_samples: FloatArrayType) -> Callable[[Real], Real]:
        return interp1d(l_samples, a_samples)

    @lazy_property
    @staticmethod
    def _a_final_() -> float:
        return 1.0

    @lazy_property
    @staticmethod
    def _l_final_(a_l_interp: scipy.interpolate.interp1d, a_final: float) -> float:
        return a_l_interp(a_final)

    def a_to_p(self, a: Real) -> Vector2Type:
        return self._gamma_(a)

    def a_to_l(self, a: Real) -> float:
        return self._a_l_interp_(a)

    def l_to_a(self, l: Real) -> float:
        return self._l_a_interp_(l)

    def partial_by_a(self, a: Real):
        return BezierCurve(np.array([
            BezierCurve(self._points_[:n]).a_to_p(a)
            for n in range(1, self._order_ + 2)
        ]))

    #def rise_order_to(self, new_order: int):
    #    new_points = self._points_
    #    for n in range(self._order_ + 1, new_order + 1):
    #        mat = np.zeros((n + 1, n))
    #        mat[(np.arange(n), np.arange(n))] = np.arange(n, 0, -1) / n
    #        mat[(np.arange(n) + 1, np.arange(n))] = np.arange(1, n + 1) / n
    #        new_points = mat @ new_points
    #    return BezierCurve(new_points)


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


class Path(LazyBase):
    """
    A list of contours, either open or closed
    """
    def __init__(
        self,
        path: skia.Path | Contours | None = None
    ):
        super().__init__()
        if isinstance(path, skia.Path):
            self._skia_path_ = path
        elif isinstance(path, Contours):
            self._skia_path_ = Path._get_skia_path_by_contours(path)
        elif path is None:
            pass
        else:
            raise ValueError(f"Unsupported path type: {type(path)}")

    def __deepcopy__(self, memo=None):  # TODO
        return Path(skia.Path(self._skia_path_))

    @classmethod
    def _get_contours_by_skia_path(cls, path: skia.Path) -> Contours:
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

    @classmethod
    def _get_skia_path_by_contours(cls, contours: Contours) -> skia.Path:
        path = skia.Path()
        for contour in contours._children_:
            path.moveTo(*contour._children_[0]._points_[0])
            for curve in contour._children_:
                points = curve._points_
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

    @lazy_property_initializer_writable
    @staticmethod
    def _skia_path_() -> skia.Path:
        return skia.Path()

    @lazy_property
    @staticmethod
    def _contours_(skia_path: skia.Path) -> Contours:
        return Path._get_contours_by_skia_path(skia_path)

    @_skia_path_.updater
    def move_to(self, point: Vector2Type):
        self._skia_path_.moveTo(skia.Point(*point))
        return self

    @_skia_path_.updater
    def line_to(self, point: Vector2Type):
        self._skia_path_.lineTo(skia.Point(*point))
        return self

    @_skia_path_.updater
    def quad_to(self, control_point: Vector2Type, point: Vector2Type):
        self._skia_path_.quadTo(skia.Point(*control_point), skia.Point(*point))
        return self

    @_skia_path_.updater
    def cubic_to(self, control_point_0: Vector2Type, control_point_1: Vector2Type, point: Vector2Type):
        self._skia_path_.cubicTo(skia.Point(*control_point_0), skia.Point(*control_point_1), skia.Point(*point))
        return self

    @_skia_path_.updater
    def conic_to(self, control_point: Vector2Type, point: Vector2Type, weight: Real):
        self._skia_path_.conicTo(skia.Point(*control_point), skia.Point(*point), weight)
        return self

    @_skia_path_.updater
    def close_path(self):
        self._skia_path_.close()
        return self

    @lazy_property
    @staticmethod
    def _a_final_(contours: Contours) -> float:
        return contours._a_final_

    @lazy_property
    @staticmethod
    def _l_final_(contours: Contours) -> float:
        return contours._l_final_

    def a_to_p(self, a: Real) -> Vector2Type:
        return self._contours_.a_to_p(a)

    def a_to_l(self, a: Real) -> float:
        return self._contours_.a_to_l(a)

    def l_to_a(self, l: Real) -> float:
        return self._contours_.l_to_a(l)

    def a_ratio_to_p(self, a_ratio: Real) -> Vector2Type:
        return self._contours_.a_ratio_to_p(a_ratio)

    def a_ratio_to_l_ratio(self, a_ratio: Real) -> float:
        return self._contours_.a_ratio_to_l_ratio(a_ratio)

    def l_ratio_to_a_ratio(self, l_ratio: Real) -> float:
        return self._contours_.l_ratio_to_a_ratio(l_ratio)

    def partial_by_a(self, a: Real):
        return Path(self._contours_.partial_by_a(a))

    def partial_by_l(self, l: Real):
        return Path(self._contours_.partial_by_l(l))

    def partial_by_a_ratio(self, a_ratio: Real):
        return Path(self._contours_.partial_by_a_ratio(a_ratio))

    def partial_by_l_ratio(self, l_ratio: Real):
        return Path(self._contours_.partial_by_l_ratio(l_ratio))
