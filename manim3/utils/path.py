from typing import Callable, Generic, TypeVar

import numpy as np
import scipy.interpolate
import skia

from ..custom_typing import *


__all__ = ["Path"]


T = TypeVar("T")


class CurveInterpolant(Generic[T]):
    def __init__(self: Self):
        self._children: list[T] = []
        #self._a_interpolator: Callable[[Real], tuple[int, float]] = lambda a: (0, 0.0)
        #self._l_interpolator: Callable[[Real], tuple[int, float]] = lambda a: (0, 0.0)
        #self._a_final: float = 0.0
        #self._l_final: float = 0.0
        self._a_knots: FloatArrayType = np.zeros(1)
        self._l_knots: FloatArrayType = np.zeros(1)
        #self._a_samples: FloatArrayType = np.array((0.0, 1.0))
        #self._l_samples: FloatArrayType = np.array((0.0, 1.0))
        #self._a_to_l: scipy.interpolate.interp1d = scipy.interpolate.interp1d(self._a_samples, self._l_samples)
        #self._l_to_a: scipy.interpolate.interp1d = scipy.interpolate.interp1d(self._l_samples, self._a_samples)
        self.requires_updating: bool = True

    def update_data(self: Self) -> Self:
        if self.requires_updating:
            children = self.get_updated_children()
            self._children = children
            self._a_knots = np.insert(np.cumsum([child.a_final for child in children]), 0, 0.0)
            self._l_knots = np.insert(np.cumsum([child.l_final for child in children]), 0, 0.0)
            #self._a_interpolator = self.integer_interpolator(a_knots)
            #self._l_interpolator = self.integer_interpolator(l_knots)
            #self._a_final = a_knots[-1]
            #self._l_final = l_knots[-1]
        return self
        #    #self._a_samples, self._l_samples = self.get_updated_samples()
        #    #self._a_to_l = scipy.interpolate.interp1d(self._a_samples, self._l_samples)
        #    #self._l_to_a = scipy.interpolate.interp1d(self._l_samples, self._a_samples)
        #    self.requires_updating = False

    @property
    def children(self: Self) -> list[T]:
        self.update_data()
        return self._children

    @property
    def a_knots(self: Self) -> FloatArrayType:
        self.update_data()
        return self._a_knots

    @property
    def l_knots(self: Self) -> FloatArrayType:
        self.update_data()
        return self._l_knots

    def a_interpolate(self: Self, a: Real) -> tuple[int, float]:
        return self.integer_interpolator(self.a_knots)(a)

    def l_interpolate(self: Self, l: Real) -> tuple[int, float]:
        return self.integer_interpolator(self.l_knots)(l)

    @property
    def a_final(self: Self) -> float:
        return self.a_knots[-1]

    @property
    def l_final(self: Self) -> float:
        return self.l_knots[-1]

    @staticmethod
    def integer_interpolator(array: FloatArrayType) -> Callable[[Real], tuple[int, float]]:
        def wrapped(target: Real) -> tuple[int, float]:
            """
            Assumed that `array` is already sorted, and that `array[0] <= target <= array[-1]`
            If `target == array[0]`, returns `(0, 0.0)`.
            Otherwise, returns `(i, target - array[i])` such that
            `0 <= i < len(array)` and `array[i] < target <= array[i + 1]`.
            """
            index = int(scipy.interpolate.interp1d(array, range(len(array)), kind="next")(target)) - 1
            if index == -1:
                return 0, 0.0
            return index, target - array[index]
        return wrapped

    def add_child(self: Self, child: T) -> Self:
        self.requires_updating = True
        self._children.append(child)
        return self

    def get_updated_children(self: Self) -> list[T]:
        return self._children

    #@property
    #def a_samples(self: Self) -> FloatArrayType:
    #    self.update_data()
    #    return self._a_samples

    #@property
    #def l_samples(self: Self) -> FloatArrayType:
    #    self.update_data()
    #    return self._l_samples

    #@property
    #def a_to_l(self: Self) -> scipy.interpolate.interp1d:
    #    self.update_data()
    #    return self._a_to_l

    #@property
    #def l_to_a(self: Self) -> scipy.interpolate.interp1d:
    #    self.update_data()
    #    return self._l_to_a

    #def get_updated_samples(self: Self) -> tuple[FloatArrayType, FloatArrayType]:
    #    # Both should start from zero
    #    a_samples = [0.0]
    #    l_samples = [0.0]
    #    prev_a = 0.0
    #    prev_l = 0.0
    #    for child in self.children:
    #        a_samples.extend((child.a_samples + prev_a)[1:])
    #        l_samples.extend((child.l_samples + prev_l)[1:])
    #        prev_a = child.last_a
    #        prev_l = child.last_l
    #    return np.array(a_samples), np.array(l_samples)

    def a_to_p(self: Self, a: Real) -> Vector2Type:
        i, a_remainder = self.a_interpolate(a)
        return self.children[i].a_to_p(a_remainder)
        #for child in self.children:
        #    if a < child.a_final():
        #        return child.a_to_p(a)
        #    a -= child.a_final()
        #return self.children[-1].a_to_p(a)

    def a_to_l(self: Self, a: Real) -> float:
        i, a_remainder = self.a_interpolate(a)
        return self.l_knots[i] + self.children[i].a_to_l(a_remainder)
        #l = 0.0
        #for child in self.children:
        #    if a < child.a_final():
        #        l += child.a_to_l(a)
        #        break
        #    a -= child.a_final()
        #    l += child.l_final()
        #return l

    def l_to_a(self: Self, l: Real) -> float:
        i, l_remainder = self.l_interpolate(l)
        return self.a_knots[i] + self.children[i].l_to_a(l_remainder)
        #a = 0.0
        #for child in self.children:
        #    if l < child.l_final():
        #        a += child.l_to_a(l)
        #        break
        #    l -= child.l_final()
        #    a += child.a_final()
        #return a

    def partial_by_a(self: Self, a: Real) -> Self:
        i, a_remainder = self.a_interpolate(a)
        result = self.__class__()
        for child in self.children[:i]:
            #if a < child.a_final():
            #    result.add_child(child.partial_by_a(a))
            #    break
            #a -= child.a_final()
            result.add_child(child)
        result.add_child(self.children[i].partial_by_a(a_remainder))
        return result

    def partial_by_l(self: Self, l: Real) -> Self:
        return self.partial_by_a(self.l_to_a(l))

    def partial_by_a_ratio(self: Self, a: Real) -> Self:
        return self.partial_by_a(a * self.a_final())

    def partial_by_l_ratio(self: Self, l: Real) -> Self:
        return self.partial_by_l(l * self.l_final())


class BezierCurve(CurveInterpolant[None]):
    """
    Bezier curves defined on domain [0, 1].
    """
    def __init__(self: Self, *points: Vector2Type):
        order = len(points) - 1
        assert order >= 0
        points_array = np.array(points)
        self.order: int = order
        self.points: Vector2ArrayType = points_array
        self.func: scipy.interpolate.BSpline = scipy.interpolate.BSpline(
            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
            c=points_array,
            k=order
        )

        segments = 16 if self.order > 1 else 1
        a_samples = np.linspace(0.0, 1.0, segments + 1)
        p_samples = self.func(a_samples)
        segment_lengths = np.linalg.norm(p_samples[1:] - p_samples[:-1], axis=1)
        l_samples = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        self.curve_length: float = l_samples[-1]
        self.a_l_interp: scipy.interpolate.interp1d = scipy.interpolate.interp1d(a_samples, l_samples)
        self.l_a_interp: scipy.interpolate.interp1d = scipy.interpolate.interp1d(l_samples, a_samples)
        super().__init__()

    @property
    def children(self: Self) -> list[None]:
        raise NotImplementedError

    def add_child(self: Self, child: None) -> Self:
        raise NotImplementedError

    def a_final(self: Self) -> float:
        return 1.0

    def l_final(self: Self) -> float:
        return self.curve_length

    def a_to_p(self: Self, a: Real) -> Vector2Type:
        return self.func(a)

    def a_to_l(self: Self, a: Real) -> float:
        return self.a_l_interp(a)

    def l_to_a(self: Self, l: Real) -> float:
        return self.l_a_interp(l)

    def partial_by_a(self: Self, a: Real) -> Self:
        return BezierCurve(*(
            BezierCurve(*self.points[:n]).a_to_p(a)
            for n in range(1, self.order + 2)
        ))

    def rise_order_to(self: Self, new_order: int) -> Self:
        new_points = self.points
        for n in range(self.order + 1, new_order + 1):
            mat = np.zeros((n + 1, n))
            mat[(np.arange(n), np.arange(n))] = np.arange(n, 0, -1) / n
            mat[(np.arange(n) + 1, np.arange(n))] = np.arange(1, n + 1) / n
            new_points = mat @ new_points
        return BezierCurve(*new_points)


class Contour(CurveInterpolant[BezierCurve]):
    """
    A list of chained Bezier curves
    """
    pass


class Path(CurveInterpolant[Contour]):
    """
    A list of contours, either open or closed
    """
    def __init__(self: Self, path: skia.Path | None = None):
        if path is None:
            path = skia.Path()
        self.skia_path: skia.Path = path
        self.requires_updating = True
        super().__init__()

    #@property
    #def children(self: Self) -> list[Contour]:
    #    if self.requires_updating:
    #        self._children = self.get_contours_by_skia_path(self.skia_path)
    #        self.requires_updating = False
    #    return self._children

    def add_child(self: Self, contour: Contour) -> Self:
        if contour.children:
            self.move_to(contour.children[0].points[0])
        for curve in contour.children:
            points = curve.points
            if len(points) == 1:
                self.line_to(points[0])
            elif len(points) == 2:
                self.quad_to(points[0], points[1])
            elif len(points) == 3:
                self.cubic_to(points[0], points[1], points[2])
            else:
                raise ValueError
        return self

    @staticmethod
    def get_contours_by_skia_path(path: skia.Path) -> list[Contour]:
        contours = []
        contour = Contour()
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
                contour.add_child(BezierCurve(*(
                    np.array(list(point)) for point in points
                )))
            elif verb == skia.Path.Verb.kConic_Verb:
                # Approximate per conic curve with 8 quads
                quad_points = skia.Path.ConvertConicToQuads(*points, iterator.conicWeight(), 3)
                for i in range(0, len(quad_points), 2):
                    contour.add_child(BezierCurve(*(
                        np.array(list(point)) for point in quad_points[i:i + 3]
                    )))
            elif verb == skia.Path.Verb.kClose_Verb:
                if contour.children:
                    contours.append(contour)
                    contour = Contour()
            else:
                raise ValueError
            verb, points = iterator.next()
        if contour.children:
            contours.append(contour)
        return contours

    def move_to(self: Self, point: Vector2Type) -> Self:
        self.requires_updating = True
        self.skia_path.moveTo(skia.Point(*point))
        return self

    def line_to(self: Self, point: Vector2Type) -> Self:
        self.requires_updating = True
        self.skia_path.lineTo(skia.Point(*point))
        return self

    def quad_to(self: Self, control_point: Vector2Type, point: Vector2Type) -> Self:
        self.requires_updating = True
        self.skia_path.quadTo(skia.Point(*control_point), skia.Point(*point))
        return self

    def cubic_to(self: Self, control_point_0: Vector2Type, control_point_1: Vector2Type, point: Vector2Type) -> Self:
        self.requires_updating = True
        self.skia_path.cubicTo(skia.Point(*control_point_0), skia.Point(*control_point_1), skia.Point(*point))
        return self

    def conic_to(self: Self, control_point: Vector2Type, point: Vector2Type, weight: Real) -> Self:
        self.requires_updating = True
        self.skia_path.conicTo(skia.Point(*control_point), skia.Point(*point), weight)
        return self

    def close_path(self: Self) -> Self:
        self.requires_updating = True
        self.skia_path.close()
        return self

    #def get_updated_samples(self: Self) -> tuple[FloatArrayType, FloatArrayType]:
    #    prev_alpha = 0.0
    #    prev_length = 0.0
    #    sample_alpha_lists = []
    #    sample_length_lists = []
    #    for contour in self.contours:
    #        contour_a_samples, contour_l_samples = contour.get_updated_samples()
    #        sample_alpha_lists.append(contour_a_samples + prev_alpha)
    #        sample_length_lists.append(contour_l_samples + prev_length)
    #        prev_alpha = contour_a_samples[-1]
    #        prev_length = contour_l_samples[-1]
    #    return np.concatenate(sample_alpha_lists), np.concatenate(sample_length_lists)

    #def partial_by_alpha(self: Self, alpha: Real) -> Self:
    #    result = Path()
    #    for contour in self.contours:
    #        if alpha < 1.0:
    #            result.add_contour(contour.partial_by_alpha(alpha))
    #            break
    #        result.add_contour(contour)
    #    return result

    #def length_ratio_to_alpha(self: Self, length_ratio: Real) -> float:
    #    a_samples, l_samples = self.get_updated_samples()
    #    length_to_alpha = scipy.interpolate.interp1d(l_samples, a_samples)
    #    return length_to_alpha(length_ratio * l_samples[-1])

    #def partial_by_length_ratio(self: Self, length_ratio: Real) -> Self:
    #    return self.partial_by_alpha(self.length_ratio_to_alpha(length_ratio))

    #@classmethod
    #def from_skia_path(cls, path: skia.Path) -> Self:
    #    result = cls()
    #    iterator = iter(path)
    #    while True:
    #        verb, points = iterator.next()
    #        if verb == skia.Path.kMove_Verb:
    #            result.move_to(as_np(points[0]))
    #        elif verb == skia.Path.kLine_Verb:
    #            result.line_to(as_np(points[1]))
    #        elif verb == skia.Path.kQuad_Verb:
    #            result.quad_to(as_np(points[1]), as_np(points[2]))
    #        elif verb == skia.Path.kCubic_Verb:
    #            result.cubic_to(as_np(points[1]), as_np(points[2]), as_np(points[3]))
    #        elif verb == skia.Path.kConic_Verb:
    #            result.conic_to(as_np(points[1]), as_np(points[2]), iterator.conicWeight())
    #        elif verb == skia.Path.kClose_Verb:
    #            result.close_path()
    #        elif verb == skia.Path.kDone_Verb:
    #            break
    #        else:
    #            raise ValueError
    #    return result

    #def to_skia_path(self: Self) -> skia.Path:
    #    path = skia.path()
    #    for contour in self.contours:
    #        path.moveTo(*contour[0].points[0])
    #        for curve in contour:
    #            points = curve.points
    #            len_points = len(points)
    #            if len_points == 1:
    #                path.lineTo(*points[0])
    #            elif len_points == 2:
    #                path.quadTo(*points[0], *points[1])
    #            elif len_points == 3:
    #                path.cubicTo(*points[0], *points[1], *points[2])
    #            else:
    #                raise ValueError
    #        path.close()
    #    return path


    #@staticmethod
    #def _find_insertion_index(target: float, vals: list[float]) -> int:
    #    """
    #    Assumed that `vals` are already sorted.
    #    Returns `i` such that `0 <= i <= len(vals)` and `vals[i - 1] < target <= vals[i]`.
    #    """
    #    # Binary search
    #    left = 0
    #    right = len(vals) - 1
    #    while left <= right:
    #        mid = (right - left) // 2 + left
    #        if vals[mid] < target:
    #            left = mid + 1
    #        else:
    #            right = mid - 1
    #    return left

    #def length_ratio_to_alpha(self: Self, length_ratio: Real) -> float:
    #    alphas = self.a_samples
    #    lengths = self.sample_length_ratios
    #    i = self._find_insertion_index(length_ratio, lengths)
    #    if i == 0:
    #        return 0.0
    #    if i == len(alphas):
    #        return 1.0
    #    return alphas[i - 1] + (length_ratio - lengths[i - 1]) / (lengths[i] - lengths[i - 1]) * (alphas[i] - alphas[i - 1])


#def partial_path_by_length(path: skia.path, length_ratio: Real) -> skia.Path:
#    contours = get_contours_from_path(path)
#    all_a_samples = []
#    all_segment_lengths = []
#    curve_index = 0
#    for contour in contours:
#        for curve in contour:
#            segments = 16 if curve.order > 1 else 1
#            a_samples = np.linspace(0.0, 1.0, segments + 1)
#            sample_points = curve(a_samples)
#            segment_lengths = np.linalg.norm(sample_points[1:] - sample_points[:-1], axis=0)
#            all_a_samples.extend((a_samples + curve_index)[1:])
#            all_segment_lengths.extend(segment_lengths)
#            #l_samples = np.insert(np.cumsum(partial_lengths), 0, 0.0)
#            #length_to_alpha = scipy.interpolate.interp1d(l_samples, a_samples)
#            curve_index += 1
#    length_to_alpha = scipy.interpolate.interp1d(np.cumsum(all_segment_lengths), all_a_samples)
#    target_length = length_ratio * sum(all_segment_lengths)
#    alpha = length_to_alpha(target_length)
#    result_contours = []
#    for contour in contours:
#        result_contour = []
#        for curve in contour:
#            if alpha < 1.0:
#                result_contour.append(curve.from_partial(alpha))
#                break
#            result_contour.append(curve)
#            alpha -= 1.0
#        result_contours.append(result_contour)
#    return get_path_from_contours(result_contours)




#def get_path_from_contours(contours: list[list[BezierCurve]]) -> skia.Path:
#    path = skia.path()
#    for contour in contours:
#        path.moveTo(*contour[0].points[0])
#        for curve in contour:
#            points = curve.points
#            len_points = len(points)
#            if len_points == 1:
#                path.lineTo(*points[0])
#            elif len_points == 2:
#                path.quadTo(*points[0], *points[1])
#            elif len_points == 3:
#                path.cubicTo(*points[0], *points[1], *points[2])
#            else:
#                raise ValueError
#        path.close()
#    return path
