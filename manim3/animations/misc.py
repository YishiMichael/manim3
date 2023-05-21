import itertools as it
from typing import (
    Callable,
    Iterator
)

from ..animations.animation import Animation
from ..animations.composition import Parallel
from ..animations.fade import FadeTransform
from ..animations.transform import Transform
from ..custom_typing import SelectorT
from ..mobjects.shape_mobject import ShapeMobject
from ..strings.string_mobject import StringMobject
from ..utils.rate import RateUtils


class TransformMatchingStrings(Parallel):
    __slots__ = ()

    def __init__(
        self,
        start_mobject: StringMobject,
        stop_mobject: StringMobject,
        key_map: dict[SelectorT, SelectorT] | None = None,
        *,
        run_time: float = 1.0,
        rate_func: Callable[[float], float] = RateUtils.linear
    ) -> None:

        def get_key_from_substr(
            substr: str
        ) -> tuple[int, str]:
            return (-len(substr), substr)

        def zip_matched_part_items(
            *part_item_iterators: Iterator[tuple[str, Iterator[ShapeMobject]]]
        ) -> Iterator[tuple[Iterator[Iterator[ShapeMobject]], ...]]:
            n = len(part_item_iterators)
            indexed_part_items = sorted(it.chain.from_iterable(
                (
                    (substr, index, mobject_iterator)
                    for substr, mobject_iterator in part_item_iterator
                )
                for index, part_item_iterator in enumerate(part_item_iterators)
            ), key=lambda t: get_key_from_substr(t[0]))
            for substr, substr_grouper in it.groupby(indexed_part_items, key=lambda t: t[0]):
                if not substr:
                    continue
                index_groupers = list(it.groupby(substr_grouper, key=lambda t: t[1]))
                if len(index_groupers) != n:
                    continue
                yield tuple(
                    (mobject_iterator for _, _, mobject_iterator in index_grouper)
                    for _, index_grouper in index_groupers
                )

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

        def get_filtered_mobject_lists(
            mobject_iter_iter: Iterator[Iterator[ShapeMobject]],
            used_mobjects: list[ShapeMobject]
        ) -> tuple[list[list[ShapeMobject]], list[ShapeMobject]]:
            result: list[list[ShapeMobject]] = []
            chained: list[ShapeMobject] = []
            for mobject_iter in mobject_iter_iter:
                mobjects = list(mobject_iter)
                if not all(
                    mobject not in used_mobjects and mobject not in chained
                    for mobject in mobjects
                ):
                    continue
                result.append(mobjects)
                chained.extend(mobjects)
            return result, chained

        if key_map is None:
            key_map = {}

        parser_0 = start_mobject._parser
        parser_1 = stop_mobject._parser

        #submobjects_0 = list(parser_0.iter_mobjects())
        #submobjects_1 = list(parser_1.iter_mobjects())
        #part_items_0 = sorted(list(parser_0.iter_group_part_items()), key=get_key)
        #part_items_1 = sorted(list(parser_1.iter_group_part_items()), key=get_key)

        animations: list[Animation] = []
        used_mobjects_0: list[ShapeMobject] = []
        used_mobjects_1: list[ShapeMobject] = []
        animation_items: tuple[tuple[bool, Iterator[tuple[Iterator[Iterator[ShapeMobject]], ...]]], ...] = (
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
        for shape_match, mobject_iter_iter_tuple_iter in animation_items:
            for mobject_iter_iter_0, mobject_iter_iter_1 in mobject_iter_iter_tuple_iter:
                mobject_lists_0, mobject_lists_chained_0 = get_filtered_mobject_lists(
                    mobject_iter_iter_0, used_mobjects_0
                )
                mobject_lists_1, mobject_lists_chained_1 = get_filtered_mobject_lists(
                    mobject_iter_iter_1, used_mobjects_1
                )
                if not mobject_lists_0 or not mobject_lists_1:
                    continue
                animations.append(get_animation(
                    shape_match,
                    mobject_lists_0,
                    mobject_lists_1
                ))
                used_mobjects_0.extend(mobject_lists_chained_0)
                used_mobjects_1.extend(mobject_lists_chained_1)

        super().__init__(
            *animations,
            run_time=run_time,
            rate_func=rate_func
        )
