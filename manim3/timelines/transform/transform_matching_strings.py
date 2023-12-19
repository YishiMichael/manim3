#from __future__ import annotations


#import itertools
#from typing import (
#    Iterator,
#    Self
#)


#from ...constants.custom_typing import SelectorType
#from ...mobjects.string_mobjects.string_mobject import StringMobject
#from ..composition.parallel import Parallel
#from ..fade.fade_in import FadeIn
#from ..fade.fade_out import FadeOut
#from ..timeline import Timeline
#from .transform import Transform


#class TransformMatchingStrings(Timeline):
#    __slots__ = (
#        "_start_mobject",
#        "_stop_mobject",
#        "_matched_indices_tuples"
#    )

#    def __init__(
#        self: Self,
#        start_mobject: StringMobject,
#        stop_mobject: StringMobject,
#        key_map: dict[SelectorType, SelectorType] | None = None
#    ) -> None:

#        def match_elements_evenly[T](
#            elements_0: tuple[T, ...],
#            elements_1: tuple[T, ...]
#        ) -> Iterator[tuple[tuple[T, ...], tuple[T, ...]]]:
#            len_0 = len(elements_0)
#            len_1 = len(elements_1)
#            if len_0 > len_1:
#                q, r = divmod(len_0, len_1)
#                iter_0 = itertools.chain(range(0, r * (q + 1), q + 1), range(r * (q + 1), len_0 + 1, q))
#                iter_1 = iter(range(len_1 + 1))
#            else:
#                q, r = divmod(len_1, len_0)
#                iter_0 = iter(range(len_0 + 1))
#                iter_1 = itertools.chain(range(0, r * (q + 1), q + 1), range(r * (q + 1), len_1 + 1, q))
#            for (start_0, start_1), (stop_0, stop_1) in itertools.pairwise(zip(iter_0, iter_1, strict=True)):
#                yield elements_0[start_0:stop_0], elements_1[start_1:stop_1]

#        if key_map is None:
#            key_map = {}

#        matched_indices_tuples: list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = []
#        indices_0: set[int] = set()
#        indices_1: set[int] = set()
#        for selector_0, selector_1 in itertools.chain(
#            key_map.items(),
#            (
#                (matched_key, matched_key)
#                for matched_key in sorted(
#                    set.intersection(
#                        set(start_mobject._string[span.as_slice()] for span in start_mobject._isolated_spans),
#                        set(stop_mobject._string[span.as_slice()] for span in stop_mobject._isolated_spans)
#                    ),
#                    key=len
#                )
#            )
#        ):
#            indices_tuple_0 = tuple(
#                indices
#                for indices in start_mobject._get_indices_tuple_by_selector(selector_0)
#                if indices_0.isdisjoint(indices)
#            )
#            indices_tuple_1 = tuple(
#                indices
#                for indices in stop_mobject._get_indices_tuple_by_selector(selector_1)
#                if indices_1.isdisjoint(indices)
#            )
#            if not indices_tuple_0 or not indices_tuple_1:
#                continue
#            indices_0.update(itertools.chain.from_iterable(indices_tuple_0))
#            indices_1.update(itertools.chain.from_iterable(indices_tuple_1))
#            matched_indices_tuples.extend(match_elements_evenly(indices_tuple_0, indices_tuple_1))

#        super().__init__(run_alpha=1.0)
#        self._start_mobject: StringMobject = start_mobject
#        self._stop_mobject: StringMobject = stop_mobject
#        self._matched_indices_tuples: tuple[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...] = tuple(
#            matched_indices_tuples
#        )

#    async def construct(
#        self: Self
#    ) -> None:
#        start_mobject = self._start_mobject
#        stop_mobject = self._stop_mobject
#        matched_indices_tuples = self._matched_indices_tuples
#        matched_mobjects_pairs = tuple(
#            (start_mobject._shape_mobjects[index_0].copy(), stop_mobject._shape_mobjects[index_1].copy())
#            for indices_tuples_0, indices_tuples_1 in matched_indices_tuples
#            for indices_0, indices_1 in itertools.product(indices_tuples_0, indices_tuples_1)
#            for index_0, index_1 in zip(indices_0, indices_1, strict=True)
#        )
#        mismatched_mobjects_0 = tuple(
#            start_mobject._shape_mobjects[index_0].copy()
#            for index_0 in set(range(len(start_mobject._shape_mobjects))).difference(*itertools.chain.from_iterable(
#                indices_tuples_0
#                for indices_tuples_0, _ in matched_indices_tuples
#            ))
#        )
#        mismatched_mobjects_1 = tuple(
#            stop_mobject._shape_mobjects[index_1].copy()
#            for index_1 in set(range(len(stop_mobject._shape_mobjects))).difference(*itertools.chain.from_iterable(
#                indices_tuples_1
#                for _, indices_tuples_1 in matched_indices_tuples
#            ))
#        )

#        self.scene.discard(start_mobject)
#        self.scene.add(
#            *(matched_mobject_0 for matched_mobject_0, _ in matched_mobjects_pairs),
#            *mismatched_mobjects_0
#        )
#        await self.play(Parallel(
#            *(
#                Transform(matched_mobject_0, matched_mobject_1)
#                for matched_mobject_0, matched_mobject_1 in matched_mobjects_pairs
#            ),
#            *(
#                FadeOut(mismatched_mobject_0)
#                for mismatched_mobject_0 in mismatched_mobjects_0
#            ),
#            *(
#                FadeIn(mismatched_mobject_1)
#                for mismatched_mobject_1 in mismatched_mobjects_1
#            )
#        ))
#        self.scene.discard(
#            *(matched_mobject_1 for _, matched_mobject_1 in matched_mobjects_pairs),
#            *mismatched_mobjects_1
#        )
#        self.scene.add(stop_mobject)
