from __future__ import annotations


import itertools
from typing import (
    Iterator,
    Self
)

from ..constants.custom_typing import SelectorT
from ..mobjects.string_mobjects.string_mobject import StringMobject
from .composition.parallel import Parallel
from .fade.fade_in import FadeIn
from .fade.fade_out import FadeOut
from .fade.fade_transform import FadeTransform
from .timeline.timeline import Timeline
#from .fade.fade_transform import FadeTransform
#from .transform import Transform


#_K = TypeVar("_K", bound=Hashable)
#_K0 = TypeVar("_K0", bound=Hashable)
#_K1 = TypeVar("_K1", bound=Hashable)
#_T = TypeVar("_T")


class TransformMatchingStrings(Timeline):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject",
        "_matched_indices_tuples"
        #"_to_discard"
    )

    def __init__(
        self: Self,
        start_mobject: StringMobject,
        stop_mobject: StringMobject,
        key_map: dict[SelectorT, SelectorT] | None = None
    ) -> None:

        def match_elements_evenly[T](
            elements_0: tuple[T, ...],
            elements_1: tuple[T, ...]
        ) -> Iterator[tuple[tuple[T, ...], tuple[T, ...]]]:
            len_0 = len(elements_0)
            len_1 = len(elements_1)
            if len_0 > len_1:
                q, r = divmod(len_0, len_1)
                iter_0 = itertools.chain(range(0, r * (q + 1), q + 1), range(r * (q + 1), len_0 + 1, q))
                iter_1 = iter(range(len_1 + 1))
            else:
                q, r = divmod(len_1, len_0)
                iter_0 = iter(range(len_0 + 1))
                iter_1 = itertools.chain(range(0, r * (q + 1), q + 1), range(r * (q + 1), len_1 + 1, q))
            for (start_0, start_1), (stop_0, stop_1) in itertools.pairwise(zip(iter_0, iter_1, strict=True)):
                yield elements_0[start_0:stop_0], elements_1[start_1:stop_1]

            #if len_0 > len_1:
            #    for tuple_1, tuple_0 in match_elements_evenly(elements_1, elements_0):
            #        yield tuple_0, tuple_1
            #    return
            #assert len_0 and len_1
            ##if len(elements_0) > len(elements_1):
            ##    elements_0, elements_1 = elements_1, elements_0
            ##len_0 = len(elements_0)
            ##len_1 = len(elements_1)
            #q, r = divmod(len_1, len_0)
            #for index_0, (start_1, stop_1) in itertools.chain(
            #    zip(
            #        range(r),
            #        itertools.pairwise(itertools.count(0, q + 1)),
            #        strict=False
            #    ),
            #    zip(
            #        range(r, len_0),
            #        itertools.pairwise(itertools.count(r * (q + 1), q)),
            #        strict=False
            #    )
            #):
            #    yield (elements_0[index_0],), elements_1[start_1:stop_1]

        if key_map is None:
            key_map = {}

        matched_indices_tuples: list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]] = []
        indices_0: set[int] = set()
        indices_1: set[int] = set()
        for selector_0, selector_1 in itertools.chain(
            key_map.items(),
            (
                (matched_key, matched_key)
                for matched_key in sorted(
                    set.intersection(
                        set(start_mobject._string[span.as_slice()] for span in start_mobject._isolated_spans),
                        set(stop_mobject._string[span.as_slice()] for span in stop_mobject._isolated_spans)
                    ),
                    key=len
                )
            )
        ):
            indices_tuple_0 = tuple(
                indices
                for indices in start_mobject._get_indices_tuple_by_selector(selector_0)
                if indices_0.isdisjoint(indices)
            )
            indices_tuple_1 = tuple(
                indices
                for indices in stop_mobject._get_indices_tuple_by_selector(selector_1)
                if indices_1.isdisjoint(indices)
            )
            #print(indices_tuple_0, indices_tuple_1)
            if not indices_tuple_0 or not indices_tuple_1:
                continue
            indices_0.update(itertools.chain.from_iterable(indices_tuple_0))
            indices_1.update(itertools.chain.from_iterable(indices_tuple_1))
            matched_indices_tuples.extend(match_elements_evenly(indices_tuple_0, indices_tuple_1))


        #def zip_matched_part_items(
        #    *part_item_iters: Iterable[tuple[str, Iterable[ShapeMobject]]]
        #) -> Iterator[tuple[Iterable[Iterable[ShapeMobject]], ...]]:

        #    def categorize(
        #        iterable: Iterable[tuple[_K, _T]]
        #    ) -> Iterator[tuple[_K, list[_T]]]:
        #        categories: dict[_K, list[_T]] = {}
        #        for key, value in iterable:
        #            categories.setdefault(key, []).append(value)
        #        yield from categories.items()

        #    def recategorize(
        #        iterable: Iterable[tuple[_K0, Iterable[tuple[_K1, _T]]]]
        #    ) -> Iterator[tuple[_K1, Iterable[tuple[_K0, _T]]]]:
        #        return categorize(
        #            (key_1, (key_0, item))
        #            for key_0, grouper_0 in iterable
        #            for key_1, item in grouper_0
        #        )

        #    def get_key_from_substr(
        #        substr: str
        #    ) -> tuple[int, str]:
        #        # A longer substring has higher priority.
        #        return (-len(substr), substr)

        #    n = len(part_item_iters)
        #    for substr, indexed_mobject_iter_iter in sorted(
        #        recategorize(enumerate(
        #            categorize(part_item_iter)
        #            for part_item_iter in part_item_iters
        #        )),
        #        key=lambda t: get_key_from_substr(t[0])
        #    ):
        #        if not substr:
        #            continue
        #        #_, mobject_iter_iter = IterUtils.unzip_pairs(indexed_mobject_iter_iter)
        #        result = tuple(mobject_iter for _, mobject_iter in indexed_mobject_iter_iter)
        #        if len(result) != n:
        #            continue
        #        yield result

        #def get_timeline_items(
        #    timeline_items: Iterable[tuple[bool, tuple[Iterable[Iterable[ShapeMobject]], ...]]],
        #    children_tuple: tuple[Iterable[ShapeMobject], ...]
        #) -> Iterator[tuple[bool, tuple[list[list[ShapeMobject]], ...]]]:
        #    used_mobject_set_tuple: tuple[set[ShapeMobject], ...] = tuple(set() for _ in children_tuple)
        #    timeline_mobject_list_list_tuple: tuple[list[list[ShapeMobject]], ...] = tuple([] for _ in children_tuple)
        #    timeline_used_mobject_set_tuple: tuple[set[ShapeMobject], ...] = tuple(set() for _ in children_tuple)
        #    for shape_match, mobject_iter_iter_tuple in timeline_items:
        #        for mobject_iter_iter, used_mobject_set, timeline_mobject_list_list, timeline_used_mobject_set in zip(
        #            mobject_iter_iter_tuple,
        #            used_mobject_set_tuple,
        #            timeline_mobject_list_list_tuple,
        #            timeline_used_mobject_set_tuple,
        #            strict=True
        #        ):
        #            timeline_mobject_list_list.clear()
        #            timeline_used_mobject_set.clear()
        #            for mobject_iter in mobject_iter_iter:
        #                mobject_list = list(mobject_iter)
        #                if not mobject_list:
        #                    continue
        #                if used_mobject_set.intersection(mobject_list) or timeline_used_mobject_set.intersection(mobject_list):
        #                    continue
        #                timeline_mobject_list_list.append(mobject_list)
        #                timeline_used_mobject_set.update(mobject_list)
        #        if not all(timeline_mobject_list_list_tuple):
        #            continue
        #        yield shape_match, timeline_mobject_list_list_tuple
        #        for used_mobject_set, timeline_used_mobject_set in zip(
        #            used_mobject_set_tuple,
        #            timeline_used_mobject_set_tuple,
        #            strict=True
        #        ):
        #            used_mobject_set.update(timeline_used_mobject_set)

        #    rest_mobject_list_tuple = tuple(list(children_iter) for children_iter in children_tuple)
        #    for rest_mobject_list, used_mobject_set in zip(
        #        rest_mobject_list_tuple,
        #        used_mobject_set_tuple,
        #        strict=True
        #    ):
        #        for used_mobject in used_mobject_set:
        #            rest_mobject_list.remove(used_mobject)
        #    if not all(rest_mobject_list_tuple):
        #        return
        #    yield False, tuple(
        #        [rest_mobject_list]
        #        for rest_mobject_list in rest_mobject_list_tuple
        #    )

        #def get_timelines(
        #    shape_match: bool,
        #    mobject_list_list_tuple: tuple[list[list[ShapeMobject]], ...]
        #) -> Iterator[FadeTransform | Transform]:

        #    def match_elements_evenly(
        #        elements_0: list[_T],
        #        elements_1: list[_T]
        #    ) -> Iterator[tuple[list[_T], list[_T]]]:
        #        len_0 = len(elements_0)
        #        len_1 = len(elements_1)
        #        if len_0 > len_1:
        #            for list_1, list_0 in match_elements_evenly(elements_1, elements_0):
        #                yield list_0, list_1
        #            return
        #        assert len_0 and len_1
        #        q, r = divmod(len_1, len_0)
        #        for i_0, (start_1, stop_1) in itertools.chain(
        #            zip(
        #                range(r),
        #                itertools.pairwise(itertools.count(0, q + 1)),
        #                strict=False
        #            ),
        #            zip(
        #                range(r, len_0),
        #                itertools.pairwise(itertools.count(r * (q + 1), q)),
        #                strict=False
        #            )
        #        ):
        #            yield [elements_0[i_0]], elements_1[start_1:stop_1]

        #    start_mobject_list_list, stop_mobject_list_list = mobject_list_list_tuple
        #    for start_mobject_list, stop_mobject_list in match_elements_evenly(
        #        [
        #            ShapeMobject().add(*mobject_list)
        #            for mobject_list in start_mobject_list_list
        #        ],
        #        [
        #            ShapeMobject().add(*mobject_list)
        #            for mobject_list in stop_mobject_list_list
        #        ]
        #    ):
        #        for start_mobject, stop_mobject in itertools.product(start_mobject_list, stop_mobject_list):
        #            start_mobject_copy = start_mobject.copy()
        #            stop_mobject_copy = stop_mobject.copy()
        #            if shape_match:
        #                yield FadeTransform(
        #                    start_mobject_copy,
        #                    stop_mobject_copy
        #                )
        #            else:
        #                yield Transform(
        #                    start_mobject_copy.concatenate(),
        #                    stop_mobject_copy.concatenate()
        #                )

        ##if key_map is None:
        ##    key_map = {}

        #parser_0 = start_mobject._parser
        #parser_1 = stop_mobject._parser
        #timeline_item_groups: tuple[tuple[bool, Iterable[tuple[Iterable[Iterable[ShapeMobject]], ...]]], ...] = (
        #    #(False, (
        #    #    (
        #    #        parser_0.iter_iter_shape_mobjects_by_selector(selector_0),
        #    #        parser_1.iter_iter_shape_mobjects_by_selector(selector_1)
        #    #    )
        #    #    for selector_0, selector_1 in key_map.items()
        #    #)),
        #    (True, zip_matched_part_items(
        #        parser_0.iter_specified_part_items(),
        #        parser_1.iter_specified_part_items()
        #    )),
        #    (True, zip_matched_part_items(
        #        parser_0.iter_group_part_items(),
        #        parser_1.iter_group_part_items()
        #    ))
        #)
        #timelines = list(itertools.chain.from_iterable(
        #    get_timelines(shape_match, mobject_list_list_tuple)
        #    for shape_match, mobject_list_list_tuple in get_timeline_items(
        #        timeline_items=(
        #            (shape_match, mobject_iter_iter_tuple)
        #            for shape_match, mobject_iter_iter_tuple_iter in timeline_item_groups
        #            for mobject_iter_iter_tuple in mobject_iter_iter_tuple_iter
        #        ),
        #        children_tuple=(
        #            parser_0.iter_shape_mobjects(),
        #            parser_1.iter_shape_mobjects()
        #        )
        #    )
        #))

        super().__init__(run_alpha=1.0)
        self._start_mobject: StringMobject = start_mobject
        self._stop_mobject: StringMobject = stop_mobject
        self._matched_indices_tuples: tuple[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...] = tuple(
            matched_indices_tuples
        )
        #self._to_discard: list[Mobject] = [
        #    timeline._stop_mobject
        #    for timeline in timelines
        #]

    async def construct(
        self: Self
    ) -> None:
        start_mobject = self._start_mobject
        stop_mobject = self._stop_mobject
        matched_indices_tuples = self._matched_indices_tuples
        matched_mobjects_pairs = tuple(
            (start_mobject._shape_mobjects[index_0].copy(), stop_mobject._shape_mobjects[index_1].copy())
            for indices_tuples_0, indices_tuples_1 in matched_indices_tuples
            for indices_0, indices_1 in itertools.product(indices_tuples_0, indices_tuples_1)
            for index_0, index_1 in zip(indices_0, indices_1, strict=True)
        )
        #matched_mobjects_0 = tuple(
        #    start_mobject._build_from_indices_tuple(indices_tuples_0).copy()
        #    for indices_tuples_0, _ in matched_indices_tuples
        #)
        #matched_mobjects_1 = tuple(
        #    stop_mobject._build_from_indices_tuple(indices_tuples_1).copy()
        #    for _, indices_tuples_1 in matched_indices_tuples
        #)
        #fade_transform_mobject_pairs = tuple(
        #    (
        #        start_mobject._build_from_indices_tuple(indices_tuples_0).copy(),
        #        stop_mobject._build_from_indices_tuple(indices_tuples_1).copy()
        #    )
        #    for indices_tuples_0, indices_tuples_1 in matched_indices_tuples
        #)
        mismatched_mobjects_0 = tuple(
            start_mobject._shape_mobjects[index_0].copy()
            for index_0 in set(range(len(start_mobject._shape_mobjects))).difference(*(
                indices_tuples_0
                for indices_tuples_0, _ in matched_indices_tuples
            ))
        )
        mismatched_mobjects_1 = tuple(
            stop_mobject._shape_mobjects[index_1].copy()
            for index_1 in set(range(len(stop_mobject._shape_mobjects))).difference(*(
                indices_tuples_1
                for _, indices_tuples_1 in matched_indices_tuples
            ))
        )

        #start_mobject._build_from_indices(
        #    tuple()
        #).copy()
        #mismatched_mobjects_1 = stop_mobject._build_from_indices(
        #    tuple(set(range(len(stop_mobject._shape_mobjects))).difference(*(
        #        indices_tuples_1
        #        for _, indices_tuples_1 in matched_indices_tuples
        #    )))
        #).copy()

        self.scene.discard(start_mobject)
        self.scene.add(
            *(matched_mobject_0 for matched_mobject_0, _ in matched_mobjects_pairs),
            *mismatched_mobjects_0
        )
        await self.play(Parallel(
            *(
                FadeTransform(matched_mobject_0, matched_mobject_1)
                for matched_mobject_0, matched_mobject_1 in matched_mobjects_pairs
            ),
            *(
                FadeOut(mismatched_mobject_0)
                for mismatched_mobject_0 in mismatched_mobjects_0
            ),
            *(
                FadeIn(mismatched_mobject_1)
                for mismatched_mobject_1 in mismatched_mobjects_1
            )
        ))
        self.scene.discard(
            *(matched_mobject_1 for _, matched_mobject_1 in matched_mobjects_pairs),
            *mismatched_mobjects_1
        )
        self.scene.add(stop_mobject)
