import itertools as it
from typing import (
    Callable,
    Iterator,
    TypeVar
)

from ..animations.animation import Animation
from ..animations.composition import Parallel
from ..animations.fade import FadeTransform
from ..animations.transform import Transform
from ..custom_typing import SelectorT
from ..mobjects.shape_mobject import ShapeMobject
from ..strings.string_mobject import StringMobject
from ..utils.iterables import IterUtils
from ..utils.rate import RateUtils


_K0 = TypeVar("_K0", str, int)
_K1 = TypeVar("_K1", str, int)
_T0 = TypeVar("_T0")
_T1 = TypeVar("_T1")
_T = TypeVar("_T")


class TransformMatchingStrings(Parallel):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject"
    )

    def __init__(
        self,
        start_mobject: StringMobject,
        stop_mobject: StringMobject,
        key_map: dict[SelectorT, SelectorT] | None = None,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:

        def group_reorder(
            iterator: Iterator[tuple[_K0, Iterator[tuple[_K1, _T]]]]
        ) -> Iterator[tuple[_K1, Iterator[tuple[_K0, _T]]]]:
            for key_1, triplet_iterator in it.groupby(sorted((
                (key_0, key_1, item)
                for key_0, grouper_0 in iterator
                for key_1, item in grouper_0
            ), key=lambda t: t[1]), key=lambda t: t[1]):
                yield key_1, (
                    (key_0, item)
                    for key_0, _, item in list(triplet_iterator)
                    # `list()` is needed due to the behavior of `groupby`.
                )

        def get_key_from_substr(
            substr: str
        ) -> tuple[int, str]:
            # A longer substring has higher priority.
            return (-len(substr), substr)



        # Iterator[Iterator[tuple[str, Iterator[ShapeMobject]]]]
        # Iterator[tuple[int, Iterator[tuple[str, A]]]]
        # => Iterator[tuple[str, Iterator[tuple[int, Iterator[A]]]]]


        def zip_matched_part_items(
            *part_item_iterators: Iterator[tuple[str, Iterator[ShapeMobject]]]
        ) -> Iterator[tuple[Iterator[Iterator[ShapeMobject]], ...]]:

            #def extract_mobject_iterators_from_index_grouper(
            #    index_grouper: Iterator[tuple[str, int, Iterator[ShapeMobject]]]
            #) -> Iterator[Iterator[ShapeMobject]]:
            #    _, _, mobject_iter_iterators = IterUtils.unzip_triplets(index_grouper)
            #    return mobject_iter_iterators

            #def group_part_items(
            #    part_item_iterator: Iterator[tuple[str, Iterator[ShapeMobject]]]
            #) -> Iterator[tuple[str, Iterator[Iterator[ShapeMobject]]]]:
            #    #substr_iterator, grouper_iterator = IterUtils.unzip_pairs(
            #    #    it.groupby(sorted(part_item_iterator, key=lambda t: t[0]), key=lambda t: t[0])
            #    #)
            #    for substr, grouper in it.groupby(sorted(part_item_iterator, key=lambda t: t[0]), key=lambda t: t[0]):
            #        yield substr, IterUtils.unzip_pairs(grouper)[1]


            n = len(part_item_iterators)
            for substr, indexed_mobject_iterator_iterator in sorted(
                group_reorder(enumerate(
                    (
                        (substr, IterUtils.unzip_pairs(list(grouper))[1])
                        # `list()` is needed due to the behavior of `groupby`.
                        for substr, grouper in it.groupby(sorted(
                            part_item_iterator,
                            key=lambda t: t[0]
                        ), key=lambda t: t[0])
                    )
                    for part_item_iterator in part_item_iterators
                )),
                key=lambda t: get_key_from_substr(t[0])
            ):
                #print(list(indexed_mobject_iterator_iterator))  # sorted??
                if not substr:
                    continue
                _, mobject_iterator_iterator = IterUtils.unzip_pairs(indexed_mobject_iterator_iterator)
                if len(result := tuple(mobject_iterator_iterator)) != n:
                    continue
                yield result

            #indexed_part_items = sorted(it.chain.from_iterable(
            #    (
            #        (substr, index, mobject_iterator)
            #        for substr, mobject_iterator in part_item_iterator
            #    )
            #    for index, part_item_iterator in enumerate(part_item_iterators)
            #), key=lambda t: get_key_from_substr(t[0]))
            #for substr, substr_grouper in it.groupby(indexed_part_items, key=lambda t: t[0]):
            #    if not substr:
            #        continue
            #    _, index_grouper_iterator = IterUtils.unzip_pairs(it.groupby(
            #        substr_grouper,
            #        key=lambda t: t[1]
            #    ))
            #    index_groupers = tuple(index_grouper_iterator)
            #    if len(index_groupers) != n:
            #        continue
            #    yield tuple(
            #        extract_mobject_iterators_from_index_grouper(index_grouper)
            #        for index_grouper in index_groupers
            #    )

                #yield tuple(
                #    (mobject_iterator for _, _, mobject_iterator in zipped_indexed_part_items)
                #    for zipped_indexed_part_items in zip(*(
                #        index_grouper for _, index_grouper in index_groupers
                #    ), strict=False)
                #)

                #for zipped_indexed_part_items in zip(*(
                #    index_grouper for _, index_grouper in index_groupers
                #), strict=False):
                #    yield tuple(
                #        mobject_iterator
                #        for _, _, mobject_iterator in zipped_indexed_part_items
                #    )

                #for _, index_grouper in index_groupers:
                #    yield tuple(iter(zip(*(
                #        mobject_iterator for _, _, mobject_iterator in index_grouper
                #    ), strict=False)))
                #yield tuple(zip(*(
                #    for _, index_grouper in index_groupers
                #), strict=False))

                #list(iter(zip(*(mobject_iterator for _, _, mobject_iterator in index_grouper) for _, index_grouper in index_groupers), strict=False))

        #def get_filtered_mobject_lists(
        #    mobject_iter_iter: Iterator[Iterator[ShapeMobject]],
        #    used_mobjects: list[ShapeMobject]
        #) -> tuple[list[list[ShapeMobject]], list[ShapeMobject]]:
        #    result: list[list[ShapeMobject]] = []
        #    result_chained: list[ShapeMobject] = []
        #    for mobject_iter in mobject_iter_iter:
        #        mobjects = list(mobject_iter)
        #        if not all(
        #            mobject not in used_mobjects and mobject not in result_chained
        #            for mobject in mobjects
        #        ):
        #            continue
        #        result.append(mobjects)
        #        result_chained.extend(mobjects)
        #    return result, result_chained


        def get_animation_items(
            animation_items: Iterator[tuple[bool, tuple[Iterator[Iterator[ShapeMobject]], ...]]],
            children_tuple: tuple[Iterator[ShapeMobject], ...]
        ) -> Iterator[tuple[bool, tuple[list[list[ShapeMobject]], ...]]]:
            #items_iter = (
            #    (shape_match, mobject_iter_iter_tuple)
            #    for shape_match, mobject_iter_iter_tuple_iter in animation_items
            #    for mobject_iter_iter_tuple in mobject_iter_iter_tuple_iter
            #)
            used_mobject_set_tuple: tuple[set[ShapeMobject], ...] = tuple(set() for _ in children_tuple)
            animation_mobject_list_list_tuple: tuple[list[list[ShapeMobject]], ...] = tuple([] for _ in children_tuple)
            animation_used_mobject_set_tuple: tuple[set[ShapeMobject], ...] = tuple(set() for _ in children_tuple)
            for shape_match, mobject_iter_iter_tuple in animation_items:
                #print([
                #    list(mobjects)
                #    for mobjects in mobject_iter_iter_tuple
                #])
                #print(shape_match, mobject_iter_iter_tuple)
                for mobject_iter_iter, used_mobject_set, animation_mobject_list_list, animation_used_mobject_set in zip(
                    mobject_iter_iter_tuple,
                    used_mobject_set_tuple,
                    animation_mobject_list_list_tuple,
                    animation_used_mobject_set_tuple,
                    strict=True
                ):
                    animation_mobject_list_list.clear()
                    animation_used_mobject_set.clear()
                    for mobject_iter in mobject_iter_iter:
                        mobject_list = list(mobject_iter)
                        if not mobject_list:
                            continue
                        if used_mobject_set.intersection(mobject_list) or animation_used_mobject_set.intersection(mobject_list):
                            continue
                        animation_mobject_list_list.append(mobject_list)
                        animation_used_mobject_set.update(mobject_list)
                #print(animation_mobject_list_list_tuple)
                if not all(animation_mobject_list_list_tuple):
                    continue
                yield shape_match, animation_mobject_list_list_tuple
                for used_mobject_set, animation_used_mobject_set in zip(
                    used_mobject_set_tuple,
                    animation_used_mobject_set_tuple,
                    strict=True
                ):
                    #assert used_mobject_set.issuperset(animation_used_mobject_set)
                    used_mobject_set.update(animation_used_mobject_set)

                #available_mobject_list_list_tuple = tuple(
                #    [
                #        mobject 
                #    ]
                #    for mobject_iter_iter, rest_mobject_list in zip(
                #        mobject_iter_iter_tuple, rest_mobject_list_tuple, strict=True
                #    )
                #)

            rest_mobject_list_tuple = tuple(list(children_iter) for children_iter in children_tuple)
            for rest_mobject_list, used_mobject_set in zip(
                    rest_mobject_list_tuple,
                    used_mobject_set_tuple,
                    strict=True
                ):
                    for used_mobject in used_mobject_set:
                        rest_mobject_list.remove(used_mobject)
            if not all(rest_mobject_list_tuple):
                return
            yield False, tuple(
                [rest_mobject_list]
                for rest_mobject_list in rest_mobject_list_tuple
            )

        def get_animations(
            shape_match: bool,
            mobject_list_list_tuple: tuple[list[list[ShapeMobject]], ...]
        ) -> Iterator[Animation]:

            def match_elements_evenly(
                elements_0: list[_T0],
                elements_1: list[_T1]
            ) -> Iterator[tuple[list[_T0], list[_T1]]]:
                len_0 = len(elements_0)
                len_1 = len(elements_1)
                if len_0 > len_1:
                    for list_1, list_0 in match_elements_evenly(elements_1, elements_0):
                        yield list_0, list_1
                    return
                assert len_0 and len_1
                r = len_1 % len_0
                for i_0, (start_1, stop_1) in zip(
                    range(r),
                    it.pairwise(it.count(0, len_0 + 1)),
                    strict=False
                ):
                    yield [elements_0[i_0]], elements_1[start_1:stop_1]
                for i_0, (start_1, stop_1) in zip(
                    range(r, len_0),
                    it.pairwise(it.count(r * (len_0 + 1), len_0)),
                    strict=False
                ):
                    yield [elements_0[i_0]], elements_1[start_1:stop_1]

            start_mobject_list_list, stop_mobject_list_list = mobject_list_list_tuple
            for start_mobject_list, stop_mobject_list in match_elements_evenly(
                [
                    ShapeMobject().add(*mobject_list)
                    for mobject_list in start_mobject_list_list
                ],
                [
                    ShapeMobject().add(*mobject_list)
                    for mobject_list in stop_mobject_list_list
                ]
            ):
                for start_mobject, stop_mobject in it.product(start_mobject_list, stop_mobject_list):
                    #print(len(start_mobject._children_), len(stop_mobject._children_))
                    start_mobject_copy = start_mobject.copy()
                    stop_mobject_copy = stop_mobject.copy()
                    if shape_match:
                        yield FadeTransform(
                            start_mobject_copy,
                            stop_mobject_copy
                        )
                    else:
                        yield Transform(
                            start_mobject_copy.concatenate(),
                            stop_mobject_copy.concatenate()
                        )


        if key_map is None:
            key_map = {}


        #submobjects_0 = list(parser_0.iter_mobjects())
        #submobjects_1 = list(parser_1.iter_mobjects())
        #part_items_0 = sorted(list(parser_0.iter_group_part_items()), key=get_key)
        #part_items_1 = sorted(list(parser_1.iter_group_part_items()), key=get_key)

        #used_mobjects_0: list[ShapeMobject] = []
        #used_mobjects_1: list[ShapeMobject] = []
        #used_mobject_list_tuple: tuple[list[ShapeMobject], ...] = []
        parser_0 = start_mobject._parser
        parser_1 = stop_mobject._parser
        animation_item_groups: tuple[tuple[bool, Iterator[tuple[Iterator[Iterator[ShapeMobject]], ...]]], ...] = (
            (False, (
                (
                    parser_0.iter_iter_shape_mobjects_by_selector(selector_0),
                    parser_1.iter_iter_shape_mobjects_by_selector(selector_1)
                )
                for selector_0, selector_1 in key_map.items()
            )),
            (True, zip_matched_part_items(
                parser_0.iter_specified_part_items(),
                parser_1.iter_specified_part_items()
            )),
            (True, zip_matched_part_items(
                parser_0.iter_group_part_items(),
                parser_1.iter_group_part_items()
            ))
        )
        animations = list(it.chain.from_iterable(
            get_animations(shape_match, mobject_list_list_tuple)
            for shape_match, mobject_list_list_tuple in get_animation_items(
                animation_items=(
                    (shape_match, mobject_iter_iter_tuple)
                    for shape_match, mobject_iter_iter_tuple_iter in animation_item_groups
                    for mobject_iter_iter_tuple in mobject_iter_iter_tuple_iter
                ),
                children_tuple=(
                    parser_0.iter_shape_mobjects(),
                    parser_1.iter_shape_mobjects()
                )
            )
        ))
        #print(len(animations))
        #for shape_match, 
        #for shape_match, mobject_iter_iter_tuple_iter in animation_items:
        #    for mobject_iter_iter_0, mobject_iter_iter_1 in mobject_iter_iter_tuple_iter:
        #        mobject_lists_0, mobject_lists_chained_0 = get_filtered_mobject_lists(
        #            mobject_iter_iter_0, used_mobjects_0
        #        )
        #        mobject_lists_1, mobject_lists_chained_1 = get_filtered_mobject_lists(
        #            mobject_iter_iter_1, used_mobjects_1
        #        )
        #        if not mobject_lists_0 or not mobject_lists_1:
        #            continue
        #        animations.append(get_animation(
        #            shape_match=shape_match,
        #            mobject_lists_0=mobject_lists_0,
        #            mobject_lists_1=mobject_lists_1
        #        ))
        #        used_mobjects_0.extend(mobject_lists_chained_0)
        #        used_mobjects_1.extend(mobject_lists_chained_1)

        #animations.append(get_animation(
        #    shape_match=False,
        #    mobject_lists_0=[[
        #        mobject for mobject in parser_0.iter_shape_mobjects()
        #        if mobject not in 
        #    ]],
        #    mobject_lists_1=mobject_lists_1
        #))

        super().__init__(
            *animations,
            run_time=run_time,
            rate_func=rate_func
        )
        self._start_mobject: StringMobject = start_mobject
        self._stop_mobject: StringMobject = stop_mobject

    async def timeline(self) -> None:
        self.discard_from_scene(self._start_mobject)
        await super().timeline()
        self.add_to_scene(self._stop_mobject)
