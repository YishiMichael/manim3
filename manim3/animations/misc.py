import itertools
from typing import (
    Hashable,
    Iterable,
    Iterator,
    TypeVar
)

from ..mobjects.mobject import Mobject
from ..mobjects.shape_mobjects.shape_mobject import ShapeMobject
from ..mobjects.string_mobjects.string_mobject import StringMobject
from .composition.parallel import Parallel
from .fade.fade_transform import FadeTransform
from .transform.transform import Transform


_K = TypeVar("_K", bound=Hashable)
_K0 = TypeVar("_K0", bound=Hashable)
_K1 = TypeVar("_K1", bound=Hashable)
_T = TypeVar("_T")


# TODO: under construction
class TransformMatchingStrings(Parallel):
    __slots__ = (
        "_start_mobject",
        "_stop_mobject",
        "_to_discard"
    )

    def __init__(
        self,
        start_mobject: StringMobject,
        stop_mobject: StringMobject
    ) -> None:

        def zip_matched_part_items(
            *part_item_iters: Iterable[tuple[str, Iterable[ShapeMobject]]]
        ) -> Iterator[tuple[Iterable[Iterable[ShapeMobject]], ...]]:

            def categorize(
                iterable: Iterable[tuple[_K, _T]]
            ) -> Iterator[tuple[_K, list[_T]]]:
                categories: dict[_K, list[_T]] = {}
                for key, value in iterable:
                    categories.setdefault(key, []).append(value)
                yield from categories.items()

            def recategorize(
                iterable: Iterable[tuple[_K0, Iterable[tuple[_K1, _T]]]]
            ) -> Iterator[tuple[_K1, Iterable[tuple[_K0, _T]]]]:
                return categorize(
                    (key_1, (key_0, item))
                    for key_0, grouper_0 in iterable
                    for key_1, item in grouper_0
                )

            def get_key_from_substr(
                substr: str
            ) -> tuple[int, str]:
                # A longer substring has higher priority.
                return (-len(substr), substr)

            n = len(part_item_iters)
            for substr, indexed_mobject_iter_iter in sorted(
                recategorize(enumerate(
                    categorize(part_item_iter)
                    for part_item_iter in part_item_iters
                )),
                key=lambda t: get_key_from_substr(t[0])
            ):
                if not substr:
                    continue
                #_, mobject_iter_iter = IterUtils.unzip_pairs(indexed_mobject_iter_iter)
                result = tuple(mobject_iter for _, mobject_iter in indexed_mobject_iter_iter)
                if len(result) != n:
                    continue
                yield result

        def get_animation_items(
            animation_items: Iterable[tuple[bool, tuple[Iterable[Iterable[ShapeMobject]], ...]]],
            children_tuple: tuple[Iterable[ShapeMobject], ...]
        ) -> Iterator[tuple[bool, tuple[list[list[ShapeMobject]], ...]]]:
            used_mobject_set_tuple: tuple[set[ShapeMobject], ...] = tuple(set() for _ in children_tuple)
            animation_mobject_list_list_tuple: tuple[list[list[ShapeMobject]], ...] = tuple([] for _ in children_tuple)
            animation_used_mobject_set_tuple: tuple[set[ShapeMobject], ...] = tuple(set() for _ in children_tuple)
            for shape_match, mobject_iter_iter_tuple in animation_items:
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
                if not all(animation_mobject_list_list_tuple):
                    continue
                yield shape_match, animation_mobject_list_list_tuple
                for used_mobject_set, animation_used_mobject_set in zip(
                    used_mobject_set_tuple,
                    animation_used_mobject_set_tuple,
                    strict=True
                ):
                    used_mobject_set.update(animation_used_mobject_set)

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
        ) -> Iterator[FadeTransform | Transform]:

            def match_elements_evenly(
                elements_0: list[_T],
                elements_1: list[_T]
            ) -> Iterator[tuple[list[_T], list[_T]]]:
                len_0 = len(elements_0)
                len_1 = len(elements_1)
                if len_0 > len_1:
                    for list_1, list_0 in match_elements_evenly(elements_1, elements_0):
                        yield list_0, list_1
                    return
                assert len_0 and len_1
                q, r = divmod(len_1, len_0)
                for i_0, (start_1, stop_1) in itertools.chain(
                    zip(
                        range(r),
                        itertools.pairwise(itertools.count(0, q + 1)),
                        strict=False
                    ),
                    zip(
                        range(r, len_0),
                        itertools.pairwise(itertools.count(r * (q + 1), q)),
                        strict=False
                    )
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
                for start_mobject, stop_mobject in itertools.product(start_mobject_list, stop_mobject_list):
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

        #if key_map is None:
        #    key_map = {}

        parser_0 = start_mobject._parser
        parser_1 = stop_mobject._parser
        animation_item_groups: tuple[tuple[bool, Iterable[tuple[Iterable[Iterable[ShapeMobject]], ...]]], ...] = (
            #(False, (
            #    (
            #        parser_0.iter_iter_shape_mobjects_by_selector(selector_0),
            #        parser_1.iter_iter_shape_mobjects_by_selector(selector_1)
            #    )
            #    for selector_0, selector_1 in key_map.items()
            #)),
            (True, zip_matched_part_items(
                parser_0.iter_specified_part_items(),
                parser_1.iter_specified_part_items()
            )),
            (True, zip_matched_part_items(
                parser_0.iter_group_part_items(),
                parser_1.iter_group_part_items()
            ))
        )
        animations = list(itertools.chain.from_iterable(
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

        super().__init__(*animations)
        self._start_mobject: StringMobject = start_mobject
        self._stop_mobject: StringMobject = stop_mobject
        self._to_discard: list[Mobject] = [
            animation._stop_mobject
            for animation in animations
        ]

    async def timeline(self) -> None:
        self.scene.discard(self._start_mobject)
        await super().timeline()
        self.scene.add(self._stop_mobject)
        self.scene.discard(*self._to_discard)
