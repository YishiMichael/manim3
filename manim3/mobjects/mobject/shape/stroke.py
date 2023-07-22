#import itertools as it
#from typing import (
#    Callable,
#    Literal,
#    overload
#)

#import numpy as np

#from ....constants.custom_typing import (
#    NP_f8,
#    NP_i4,
#    NP_x3f8,
#    NP_xf8,
#    NP_xi4
#)
#from ....lazy.lazy import (
#    Lazy,
#    LazyObject
#)
#from ....utils.space import SpaceUtils


#class Stroke(LazyObject):
#    __slots__ = ()

#    def __init__(
#        self,
#        points: NP_x3f8 | None = None,
#        disjoints: NP_xi4 | None = None
#    ) -> None:
#        super().__init__()
#        # Elements in `disjoints` should be strictly increasing
#        # in the open interval `(0, len(points))`.
#        # E.g. 8 points with disjoints `[3, 6]` give
#        # `0-1-2, 3-4-5, 6-7`.
#        if points is not None:
#            self._points_ = points
#        if disjoints is not None:
#            self._disjoints_ = disjoints

#    @Lazy.variable_array
#    @classmethod
#    def _points_(cls) -> NP_x3f8:
#        return np.zeros((0, 3))

#    @Lazy.variable_array
#    @classmethod
#    def _disjoints_(cls) -> NP_xi4:
#        return np.zeros((0,), dtype=np.int32)

#    @Lazy.property_array
#    @classmethod
#    def _cumlengths_(
#        cls,
#        points: NP_x3f8,
#        disjoints: NP_xi4
#    ) -> NP_xf8:
#        if not len(points):
#            return np.zeros((0,))
#        vectors: NP_x3f8 = np.diff(points, axis=0)
#        lengths = np.insert(SpaceUtils.norm(vectors), 0, 0.0)
#        lengths[disjoints] = 0.0
#        return lengths.cumsum()

#    @Lazy.property_array
#    @classmethod
#    def _length_(
#        cls,
#        cumlengths: NP_xf8
#    ) -> NP_f8:
#        if not len(cumlengths):
#            return np.zeros(())
#        return cumlengths[-1] * np.ones(())

#    @overload
#    @classmethod
#    def _interpolate_knots(
#        cls,
#        knots: NP_xf8,
#        values: NP_f8,
#        *,
#        side: Literal["left", "right"]
#    ) -> tuple[NP_i4, NP_f8]: ...

#    @overload
#    @classmethod
#    def _interpolate_knots(
#        cls,
#        knots: NP_xf8,
#        values: NP_xf8,
#        *,
#        side: Literal["left", "right"]
#    ) -> tuple[NP_xi4, NP_xf8]: ...

#    @classmethod
#    def _interpolate_knots(
#        cls,
#        knots: NP_xf8,
#        values: NP_f8 | NP_xf8,
#        *,
#        side: Literal["left", "right"]
#    ) -> tuple[NP_i4, NP_f8] | tuple[NP_xi4, NP_xf8]:
#        index = np.clip(np.searchsorted(knots, values, side=side), 1, len(knots) - 1)
#        residue = (values - knots[index - 1]) / np.maximum(knots[index] - knots[index - 1], 1e-6)
#        return index, residue

#    #@classmethod
#    #def _partial_residues(
#    #    cls,
#    #    knots: NP_xf8,
#    #    start: float,
#    #    stop: float
#    #) -> tuple[int, float, int, float]:
#    #    start_index, start_residue = cls._interpolate_knots(knots, start * np.ones((1,)), side="right")
#    #    stop_index, stop_residue = cls._interpolate_knots(knots, stop * np.ones((1,)), side="left")
#    #    return int(start_index), float(start_residue), int(stop_index), float(stop_residue)

#    @classmethod
#    def partial(
#        cls,
#        stroke: "Stroke"
#    ) -> "Callable[[float, float], Stroke]":

#        #def interpolate_knots(
#        #    knots: NP_xf8,
#        #    value: float,
#        #    *,
#        #    side: Literal["left", "right"]
#        #) -> tuple[int, float]:
#        #    index = np.searchsorted(knots, value, side=side)
#        #    residue = (value - knots[index - 1]) / np.maximum(knots[index] - knots[index - 1], 1e-6)
#        #    return int(index), float(residue)

#        points = stroke._points_
#        disjoints = stroke._disjoints_
#        cumlengths = stroke._cumlengths_
#        length = stroke._length_

#        def callback(
#            start: float,
#            stop: float
#        ) -> Stroke:
#            if not len(cumlengths) or start > stop:
#                return Stroke()
#            start_index, start_residue = cls._interpolate_knots(cumlengths, start * length, side="right")
#            stop_index, stop_residue = cls._interpolate_knots(cumlengths, stop * length, side="left")
#            #start_index = np.searchsorted(cumlengths, start * length, side="right") - 1
#            #stop_index = np.searchsorted(cumlengths, stop * length, side="left") - 1
#            #start_residue = (start * length - cumlengths[start_index]) / (cumlengths[start_index + 1] - cumlengths[start_index])
#            #stop_residue = (stop * length - cumlengths[stop_index]) / (cumlengths[stop_index + 1] - cumlengths[stop_index])
#            disjoint_start_index = np.searchsorted(disjoints, start_index, side="right")
#            disjoint_stop_index = np.searchsorted(disjoints, stop_index, side="left")
#            return Stroke(
#                points=np.array([
#                    SpaceUtils.lerp(points[start_index - 1], points[start_index])(start_residue),
#                    *points[start_index:stop_index],
#                    SpaceUtils.lerp(points[stop_index - 1], points[stop_index])(stop_residue)
#                ]),
#                disjoints=disjoints[disjoint_start_index:disjoint_stop_index]
#            )

#        return callback

#    @classmethod
#    def interpolate(
#        cls,
#        stroke_0: "Stroke",
#        stroke_1: "Stroke"
#    ) -> "Callable[[float], Stroke]":
#        return cls._interpolate(stroke_0, stroke_1, has_inlay=False)

#    @classmethod
#    def _interpolate(
#        cls,
#        stroke_0: "Stroke",
#        stroke_1: "Stroke",
#        *,
#        has_inlay: bool
#    ) -> "Callable[[float], Stroke]":
#        assert len(stroke_0._points_)
#        assert len(stroke_1._points_)
#        points_0 = stroke_0._points_
#        points_1 = stroke_1._points_
#        disjoints_0 = stroke_0._disjoints_
#        disjoints_1 = stroke_1._disjoints_
#        knots_0 = stroke_0._cumlengths_ * stroke_1._length_
#        knots_1 = stroke_1._cumlengths_ * stroke_0._length_

#        #indices_0 = np.minimum(np.searchsorted(knots_0, knots_1[:-1], side="right"), len(knots_0) - 1)
#        #indices_1 = np.maximum(np.searchsorted(knots_1, knots_0[1:], side="left"), 1)
#        #residues_0 = (knots_1[:-1] - knots_0[indices_0 - 1]) / np.maximum(knots_0[indices_0] - knots_0[indices_0 - 1], 1e-6)
#        #residues_1 = (knots_0[1:] - knots_1[indices_1 - 1]) / np.maximum(knots_1[indices_1] - knots_1[indices_1 - 1], 1e-6)
#        indices_0, residues_0 = cls._interpolate_knots(knots_0, knots_1[:-1], side="right")
#        indices_1, residues_1 = cls._interpolate_knots(knots_1, knots_0[1:], side="left")

#        total_indices_0 = indices_0 + np.arange(len(indices_0)) - 1
#        total_indices_1 = indices_1 + np.arange(len(indices_1))
#        points_order = np.argsort(np.concatenate((total_indices_0, total_indices_1)))
#        total_points_0 = np.concatenate((
#            SpaceUtils.lerp(points_0[indices_0 - 1], points_0[indices_0])(residues_0[:, None]),
#            points_0[1:]
#        ))[points_order]
#        total_points_1 = np.concatenate((
#            points_1[:-1],
#            SpaceUtils.lerp(points_1[indices_1 - 1], points_1[indices_1])(residues_1[:, None])
#        ))[points_order]

#        total_disjoints_0 = total_indices_1[disjoints_0 - 1]
#        total_disjoints_1 = total_indices_0[disjoints_1]
#        total_disjoints = np.sort(np.concatenate((total_disjoints_0, total_disjoints_1)))

#        if has_inlay:
#            n_points = len(indices_0) + len(indices_1)
#            disjoint_indices_0 = np.searchsorted(total_disjoints_0, total_disjoints_1, side="right")
#            disjoint_indices_1 = np.searchsorted(total_disjoints_1, total_disjoints_0, side="left")
#            inlay_points_list_0 = [
#                total_points_0[[start_index, *total_disjoints_1[disjoint_start:disjoint_stop], stop_index - 1]]
#                for (start_index, stop_index), (disjoint_start, disjoint_stop) in zip(
#                    it.pairwise((0, *total_disjoints_0, n_points)),
#                    it.pairwise((0, *disjoint_indices_1, len(total_disjoints_1))),
#                    strict=True
#                )
#            ]
#            inlay_points_list_1 = [
#                total_points_1[[start_index, *total_disjoints_0[disjoint_start:disjoint_stop], stop_index - 1]]
#                for (start_index, stop_index), (disjoint_start, disjoint_stop) in zip(
#                    it.pairwise((0, *total_disjoints_1, n_points)),
#                    it.pairwise((0, *disjoint_indices_0, len(total_disjoints_0))),
#                    strict=True
#                )
#            ]
#            inlay_disjoints_0 = np.cumsum([len(inlay_points) for inlay_points in inlay_points_list_0])
#            inlay_disjoints_1 = np.cumsum([len(inlay_points) for inlay_points in inlay_points_list_1])

#            total_points_0 = np.concatenate((
#                total_points_0,
#                *inlay_points_list_0,
#                *(
#                    inlay_points.mean(axis=0, keepdims=True).repeat(len(inlay_points), axis=0)
#                    for inlay_points in inlay_points_list_1
#                )
#            ))
#            total_points_1 = np.concatenate((
#                total_points_1,
#                *(
#                    inlay_points.mean(axis=0, keepdims=True).repeat(len(inlay_points), axis=0)
#                    for inlay_points in inlay_points_list_0
#                ),
#                *inlay_points_list_1
#            ))
#            total_disjoints = np.concatenate((
#                total_disjoints,
#                np.insert(inlay_disjoints_0[:-1], 0, 0) + n_points,
#                np.insert(inlay_disjoints_1[:-1], 0, 0) + n_points + inlay_disjoints_0[-1]
#            ))

#        def callback(
#            alpha: float
#        ) -> Stroke:
#            return Stroke(
#                points=SpaceUtils.lerp(total_points_0, total_points_1)(alpha),
#                disjoints=total_disjoints
#            )

#        return callback

#    @classmethod
#    def concatenate(
#        cls,
#        *strokes: "Stroke"
#    ) -> "Callable[[], Stroke]":
#        result = cls._concatenate(*strokes)

#        def callback() -> Stroke:
#            return result

#        return callback

#    @classmethod
#    def _concatenate(
#        cls,
#        *strokes: "Stroke"
#    ) -> "Stroke":
#        if not strokes:
#            return Stroke()

#        points = np.concatenate([
#            stroke._points_
#            for stroke in strokes
#        ])
#        offsets = np.insert(np.cumsum([
#            len(stroke._points_)
#            for stroke in strokes[:-1]
#        ]), 0, 0)
#        disjoints = np.concatenate([
#            offset + np.insert(stroke._disjoints_, 0, 0)
#            for stroke, offset in zip(strokes, offsets, strict=True)
#        ])[1:]
#        return Stroke(
#            points=points,
#            disjoints=disjoints
#        )
