__all__ = [
    "CommandFlag",
    "EdgeFlag",
    "StringMobject"
]


from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import itertools as it
import re
from typing import (
    Callable,
    Generator
)
import warnings

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..custom_typing import (
    Real,
    Selector
)
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.svg_mobject import SVGMobject
from ..utils.color import ColorUtils
from ..utils.lazy import lazy_slot


class CommandFlag(Enum):
    OPEN = 1
    CLOSE = -1
    OTHER = 0


class EdgeFlag(Enum):
    START = 1
    STOP = -1

    def __neg__(self) -> "EdgeFlag":
        return EdgeFlag(-self.get_value())

    def get_value(self) -> int:
        return self.value


class Span:
    def __init__(self, start: int, stop: int):
        assert start <= stop, f"Invalid span: ({start}, {stop})"
        self.start: int = start
        self.stop: int = stop

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)

    def get_edge_index(self, edge_flag: EdgeFlag) -> int:
        return self.start if edge_flag == EdgeFlag.START else self.stop


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ConfiguredItem:
    span: Span
    attr_dict: dict[str, str]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class IsolatedItem:
    span: Span


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProtectedItem:
    span: Span


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class CommandItem:
    match_obj: re.Match[str]

    @property
    def span(self) -> Span:
        return Span(*self.match_obj.span())


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class LabelledItem:
    label: int
    span: Span
    attr_dict: dict[str, str]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class LabelledInsertionItem:
    labelled_item: LabelledItem
    edge_flag: EdgeFlag

    @property
    def span(self) -> Span:
        index = self.labelled_item.span.get_edge_index(self.edge_flag)
        return Span(index, index)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class LabelledShapeItem:
    label: int
    shape_mobject: ShapeMobject


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ShapeItem:
    span: Span
    shape_mobject: ShapeMobject


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ParsingResult:
    shape_items: list[ShapeItem]
    specified_part_items: list[tuple[str, list[ShapeMobject]]]
    group_part_items: list[tuple[str, list[ShapeMobject]]]


class StringMobject(SVGMobject):
    """
    An abstract base class for `Tex` and `MarkupText`

    This class aims to optimize the logic of "slicing children
    via substrings". This could be much clearer and more user-friendly
    than slicing through numerical indices explicitly.

    Users are expected to specify substrings in `isolate` parameter
    if they want to do anything with their corresponding children.
    `isolate` parameter can be either a string, a `re.Pattern` object,
    or a 2-tuple containing integers or None, or a collection of the above.
    Note, substrings specified cannot *partly* overlap with each other.

    Each instance of `StringMobject` generates 2 svg files.
    The additional one is generated with some color commands inserted,
    so that each child of the original `SVGMobject` will be labelled
    by the color of its paired child from the additional `SVGMobject`.
    """
    __slots__ = ()

    def __new__(
        cls,
        *,
        string: str = "",
        isolate: Selector = (),
        protect: Selector = (),
        configured_items_generator: Generator[tuple[Span, dict[str, str]], None, None],
        get_content_prefix_and_suffix: Callable[[bool], tuple[str, str]],
        get_svg_path: Callable[[str], str],
        width: Real | None,
        height: Real | None,
        frame_scale: Real | None
    ):
        parsing_result = cls._parse(
            string=string,
            isolate=isolate,
            protect=protect,
            configured_items_generator=configured_items_generator,
            get_content_prefix_and_suffix=get_content_prefix_and_suffix,
            get_svg_path=get_svg_path,
            width=width,
            height=height,
            frame_scale=frame_scale
        )
        shape_mobjects = [
            shape_item.shape_mobject
            for shape_item in parsing_result.shape_items
        ]
        instance = super().__new__(cls)
        instance._string = string
        instance._parsing_result = parsing_result
        #instance._shape_mobjects.extend(shape_mobjects)
        instance.add(*shape_mobjects)
        return instance

    @lazy_slot
    @staticmethod
    def _string() -> str:
        return NotImplemented

    @lazy_slot
    @staticmethod
    def _parsing_result() -> ParsingResult:
        return NotImplemented

    @classmethod
    def _parse(
        cls,
        string: str,
        isolate: Selector,
        protect: Selector,
        configured_items_generator: Generator[tuple[Span, dict[str, str]], None, None],
        get_content_prefix_and_suffix: Callable[[bool], tuple[str, str]],
        get_svg_path: Callable[[str], str],
        width: Real | None = None,
        height: Real | None = None,
        frame_scale: Real | None = None
    ) -> ParsingResult:
        labelled_items, replaced_items = cls._get_labelled_items_and_replaced_items(
            string=string,
            isolate=isolate,
            protect=protect,
            configured_items_generator=configured_items_generator
        )
        replaced_spans = [replaced_item.span for replaced_item in replaced_items]
        original_pieces = [
            string[start:stop]
            for start, stop in zip(
                [interval_span.stop for interval_span in replaced_spans[:-1]],
                [interval_span.start for interval_span in replaced_spans[1:]]
            )
        ]

        labelled_shape_items = cls._get_labelled_shape_items(
            original_pieces=original_pieces,
            replaced_items=replaced_items,
            labels_count=len(labelled_items),
            get_svg_path=get_svg_path,
            get_content_prefix_and_suffix=get_content_prefix_and_suffix,
            width=width,
            height=height,
            frame_scale=frame_scale
        )

        label_to_span_dict = {
            labelled_item.label: labelled_item.span
            for labelled_item in labelled_items
        }
        shape_items = cls._get_shape_items(
            labelled_shape_items=labelled_shape_items,
            label_to_span_dict=label_to_span_dict
        )
        specified_part_items = cls._get_specified_part_items(
            shape_items=shape_items,
            string=string,
            labelled_items=labelled_items
        )
        group_part_items = cls._get_group_part_items(
            original_pieces=original_pieces,
            replaced_items=replaced_items,
            labelled_shape_items=labelled_shape_items,
            label_to_span_dict=label_to_span_dict
        )
        return ParsingResult(
            shape_items=shape_items,
            specified_part_items=specified_part_items,
            group_part_items=group_part_items
        )

    @classmethod
    def _get_labelled_items_and_replaced_items(
        cls,
        string: str,
        isolate: Selector,
        protect: Selector,
        configured_items_generator: Generator[tuple[Span, dict[str, str]], None, None]
    ) -> tuple[list[LabelledItem], list[CommandItem | LabelledInsertionItem]]:
        def get_key(
            index_item: tuple[ConfiguredItem | IsolatedItem | ProtectedItem | CommandItem, EdgeFlag, int, int]
        ) -> tuple[int, int, int, int, int]:
            span_item, edge_flag, priority, i = index_item
            flag_value = edge_flag.get_value()
            index = span_item.span.get_edge_index(edge_flag)
            paired_index = span_item.span.get_edge_index(-edge_flag)
            return (
                index,
                flag_value * (2 if index != paired_index else -1),
                -paired_index,
                flag_value * priority,
                flag_value * i
            )

        index_items: list[tuple[ConfiguredItem | IsolatedItem | ProtectedItem | CommandItem, EdgeFlag, int, int]] = sorted((
            (span_item, edge_flag, priority, i)
            for priority, span_item_iter in enumerate((
                (ConfiguredItem(span=span, attr_dict=attr_dict) for span, attr_dict in configured_items_generator),
                (IsolatedItem(span=span) for span in cls._iter_spans_by_selector(isolate, string)),
                (ProtectedItem(span=span) for span in cls._iter_spans_by_selector(protect, string)),
                (CommandItem(match_obj=match_obj) for match_obj in cls._iter_command_matches(string))
            ), start=1)
            for i, span_item in enumerate(span_item_iter)
            for edge_flag in EdgeFlag
        ), key=get_key)

        labelled_items: list[LabelledItem] = []
        replaced_items: list[CommandItem | LabelledInsertionItem] = []
        overlapping_spans: list[Span] = []
        level_mismatched_spans: list[Span] = []
        label_counter: it.count[int] = it.count(start=1)
        protect_level: int = 0
        bracket_count: int = 0
        bracket_stack: list[int] = [0]
        open_command_stack: list[tuple[int, CommandItem]] = []
        open_stack: list[tuple[int, ConfiguredItem | IsolatedItem, int, list[int]]] = []

        def add_labelled_item(labelled_item: LabelledItem, pos: int) -> None:
            labelled_items.append(labelled_item)
            replaced_items.insert(pos, LabelledInsertionItem(
                labelled_item=labelled_item,
                edge_flag=EdgeFlag.START
            ))
            replaced_items.append(LabelledInsertionItem(
                labelled_item=labelled_item,
                edge_flag=EdgeFlag.STOP
            ))

        for span_item, edge_flag, _, _ in index_items:
            if isinstance(span_item, ProtectedItem | CommandItem):
                protect_level += edge_flag.get_value()
                if isinstance(span_item, ProtectedItem):
                    continue
                if edge_flag == EdgeFlag.START:
                    continue
                command_item = span_item
                command_flag = cls._get_command_flag(command_item.match_obj)
                if command_flag == CommandFlag.OPEN:
                    bracket_count += 1
                    bracket_stack.append(bracket_count)
                    replaced_items.append(command_item)
                    open_command_stack.append((len(replaced_items), command_item))
                elif command_flag == CommandFlag.OTHER:
                    replaced_items.append(command_item)
                else:
                    pos, open_command_item = open_command_stack.pop()
                    bracket_stack.pop()
                    attr_dict = cls._get_attr_dict_from_command_pair(
                        open_command_item.match_obj, command_item.match_obj
                    )
                    if attr_dict is not None:
                        add_labelled_item(LabelledItem(
                            label=next(label_counter),
                            span=Span(open_command_item.span.stop, command_item.span.start),
                            attr_dict=attr_dict
                        ), pos)
                    replaced_items.append(command_item)
                continue
            if edge_flag == EdgeFlag.START:
                open_stack.append((
                    len(replaced_items), span_item, protect_level, bracket_stack.copy()
                ))
                continue
            span = span_item.span
            pos, open_span_item, open_protect_level, open_bracket_stack = open_stack.pop()
            if open_span_item is not span_item:
                overlapping_spans.append(span)
                continue
            if open_protect_level or protect_level:
                continue
            if open_bracket_stack != bracket_stack:
                level_mismatched_spans.append(span)
                continue
            add_labelled_item(LabelledItem(
                label=next(label_counter),
                span=span,
                attr_dict=span_item.attr_dict if isinstance(span_item, ConfiguredItem) else {}
            ), pos)
        add_labelled_item(LabelledItem(
            label=0,
            span=Span(0, len(string)),
            attr_dict={}
        ), 0)

        if overlapping_spans:
            warnings.warn(
                "Partly overlapping substrings detected: {0}".format(
                    ", ".join(
                        f"'{string[span.as_slice()]}'"
                        for span in overlapping_spans
                    )
                )
            )
        if level_mismatched_spans:
            warnings.warn(
                "Cannot handle substrings: {0}".format(
                    ", ".join(
                        f"'{string[span.as_slice()]}'"
                        for span in level_mismatched_spans
                    )
                )
            )
        return labelled_items, replaced_items

    @classmethod
    def _iter_spans_by_selector(cls, selector: Selector, string: str) -> Generator[Span, None, None]:
        def iter_spans_by_single_selector(sel: str | re.Pattern[str] | slice, string: str) -> Generator[Span, None, None]:
            if isinstance(sel, str):
                for match_obj in re.finditer(re.escape(sel), string, flags=re.MULTILINE):
                    yield Span(*match_obj.span())
            elif isinstance(sel, re.Pattern):
                for match_obj in sel.finditer(string):
                    yield Span(*match_obj.span())
            elif isinstance(sel, slice):
                start = sel.start
                stop = sel.stop
                assert isinstance(start, int | None)
                assert isinstance(stop, int | None)
                if start is None or start < 0:
                    start = 0
                if stop is None or stop > len(string):
                    stop = len(string)
                yield Span(start, stop)
            else:
                raise TypeError(f"Invalid selector: '{sel}'")

        if isinstance(selector, str | re.Pattern | slice):
            yield from iter_spans_by_single_selector(selector, string)
        else:
            for sel in selector:
                yield from iter_spans_by_single_selector(sel, string)

    @classmethod
    def _get_replaced_pieces(
        cls,
        replaced_items: list[CommandItem | LabelledInsertionItem],
        command_replace_func: Callable[[re.Match[str]], str],
        command_insert_func: Callable[[int, EdgeFlag, dict[str, str]], str]
    ) -> list[str]:
        return [
            command_replace_func(replaced_item.match_obj)
            if isinstance(replaced_item, CommandItem)
            else command_insert_func(
                replaced_item.labelled_item.label,
                replaced_item.edge_flag,
                replaced_item.labelled_item.attr_dict
            )
            for replaced_item in replaced_items
        ]

    @classmethod
    def _replace_string(
        cls,
        original_pieces: list[str],
        replaced_pieces: list[str],
        start_index: int,
        stop_index: int
    ) -> str:
        return "".join(it.chain(*zip(
            original_pieces[start_index:stop_index],
            (*replaced_pieces[start_index + 1:stop_index], ""),
            strict=True
        )))

    @classmethod
    def _get_labelled_shape_items(
        cls,
        original_pieces: list[str],
        replaced_items: list[CommandItem | LabelledInsertionItem],
        labels_count: int,
        get_svg_path: Callable[[str], str],
        get_content_prefix_and_suffix: Callable[[bool], tuple[str, str]],
        width: Real | None,
        height: Real | None,
        frame_scale: Real | None
    ) -> list[LabelledShapeItem]:

        def get_svg_path_by_content(is_labelled: bool) -> str:
            content_replaced_pieces = cls._get_replaced_pieces(
                replaced_items=replaced_items,
                command_replace_func=cls._replace_for_content,
                command_insert_func=lambda label, edge_flag, attr_dict: cls._get_command_string(
                    attr_dict,
                    edge_flag=edge_flag,
                    label=label if is_labelled else None
                )
            )
            body = cls._replace_string(
                original_pieces=original_pieces,
                replaced_pieces=content_replaced_pieces,
                start_index=0,
                stop_index=len(original_pieces)
            )
            prefix, suffix = get_content_prefix_and_suffix(is_labelled)
            content = "".join((prefix, body, suffix))
            return get_svg_path(content)

        plain_shapes = SVGMobject(
            file_path=get_svg_path_by_content(is_labelled=False),
            width=width,
            height=height,
            frame_scale=frame_scale
        )._shape_mobjects

        if labels_count == 1:
            return [
                LabelledShapeItem(
                    label=0,
                    shape_mobject=plain_shape
                )
                for plain_shape in plain_shapes
            ]

        labelled_shapes = SVGMobject(
            file_path=get_svg_path_by_content(is_labelled=True)
        )._shape_mobjects
        if len(plain_shapes) != len(labelled_shapes):
            warnings.warn(
                "Cannot align children of the labelled svg to the original svg. Skip the labelling process."
            )
            return [
                LabelledShapeItem(
                    label=0,
                    shape_mobject=plain_shape
                )
                for plain_shape in plain_shapes
            ]

        rearranged_labelled_shapes = cls._rearrange_labelled_shapes_by_positions(plain_shapes, labelled_shapes)
        unrecognizable_colors: list[str] = []
        labelled_shape_items: list[LabelledShapeItem] = []
        for plain_shape, labelled_shape in zip(plain_shapes, rearranged_labelled_shapes, strict=True):
            color_hex = ColorUtils.color_to_hex(labelled_shape._color_)
            label = int(color_hex[1:], 16)
            if label >= labels_count:
                unrecognizable_colors.append(color_hex)
                label = 0
            labelled_shape_items.append(LabelledShapeItem(
                label=label,
                shape_mobject=plain_shape
            ))

        if unrecognizable_colors:
            warnings.warn(
                "Unrecognizable color labels detected ({0}). The result could be unexpected.".format(
                    ", ".join(unrecognizable_colors)
                )
            )
        return labelled_shape_items

    @classmethod
    def _rearrange_labelled_shapes_by_positions(
        cls,
        plain_shapes: list[ShapeMobject],
        labelled_shapes: list[ShapeMobject]
    ) -> list[ShapeMobject]:
        # Rearrange children of `labelled_svg` so that
        # each child is labelled by the nearest one of `labelled_svg`.
        # The correctness cannot be ensured, since the svg may
        # change significantly after inserting color commands.
        if not labelled_shapes:
            return []

        plain_svg = SVGMobject().add(*plain_shapes)
        labelled_svg = SVGMobject().add(*labelled_shapes)
        labelled_svg.move_to(plain_svg).stretch_to_fit_size(
            2.0 * plain_svg.get_bounding_box().radius
        )

        distance_matrix = cdist(
            [shape.get_center() for shape in plain_shapes],
            [shape.get_center() for shape in labelled_shapes]
        )
        _, indices = linear_sum_assignment(distance_matrix)
        return [
            labelled_shapes[index]
            for index in indices
        ]

    @classmethod
    def _get_shape_items(
        cls,
        labelled_shape_items: list[LabelledShapeItem],
        label_to_span_dict: dict[int, Span]
    ) -> list[ShapeItem]:
        return [
            ShapeItem(
                span=label_to_span_dict[labelled_shape_item.label],
                shape_mobject=labelled_shape_item.shape_mobject
            )
            for labelled_shape_item in labelled_shape_items
        ]

    @classmethod
    def _get_specified_part_items(
        cls,
        shape_items: list[ShapeItem],
        string: str,
        labelled_items: list[LabelledItem]
    ) -> list[tuple[str, list[ShapeMobject]]]:
        return [
            (
                string[labelled_item.span.as_slice()],
                cls._get_shape_mobject_list_by_span(labelled_item.span, shape_items)
            )
            for labelled_item in labelled_items
        ]

    @classmethod
    def _get_shape_mobject_list_by_span(cls, arbitrary_span: Span, shape_items: list[ShapeItem]) -> list[ShapeMobject]:
        return [
            shape_item.shape_mobject
            for shape_item in shape_items
            if cls._span_contains(arbitrary_span, shape_item.span)
        ]

    @classmethod
    def _span_contains(cls, span_0: Span, span_1: Span) -> bool:
        return span_0.start <= span_1.start and span_0.stop >= span_1.stop

    @classmethod
    def _get_group_part_items(
        cls,
        original_pieces: list[str],
        replaced_items: list[CommandItem | LabelledInsertionItem],
        labelled_shape_items: list[LabelledShapeItem],
        label_to_span_dict: dict[int, Span]
    ) -> list[tuple[str, list[ShapeMobject]]]:
        if not labelled_shape_items:
            return []

        range_lens, group_labels = zip(*(
            (len(list(grouper)), val)
            for val, grouper in it.groupby(labelled_shape_item.label for labelled_shape_item in labelled_shape_items)
        ))
        labelled_insertion_item_to_index_dict = {
            (replaced_item.labelled_item.label, replaced_item.edge_flag): index
            for index, replaced_item in enumerate(replaced_items)
            if isinstance(replaced_item, LabelledInsertionItem)
        }
        start_items = [
            (group_labels[0], EdgeFlag.START),
            *(
                (curr_label, EdgeFlag.START)
                if cls._span_contains(
                    label_to_span_dict[prev_label], label_to_span_dict[curr_label]
                )
                else (prev_label, EdgeFlag.STOP)
                for prev_label, curr_label in it.pairwise(group_labels)
            )
        ]
        stop_items = [
            *(
                (curr_label, EdgeFlag.STOP)
                if cls._span_contains(
                    label_to_span_dict[next_label], label_to_span_dict[curr_label]
                )
                else (next_label, EdgeFlag.START)
                for curr_label, next_label in it.pairwise(group_labels)
            ),
            (group_labels[-1], EdgeFlag.STOP)
        ]
        matching_replaced_pieces = cls._get_replaced_pieces(
            replaced_items=replaced_items,
            command_replace_func=cls._replace_for_matching,
            command_insert_func=lambda label, flag, attr_dict: ""
        )
        group_substrs = [
            re.sub(r"\s+", "", cls._replace_string(
                original_pieces=original_pieces,
                replaced_pieces=matching_replaced_pieces,
                start_index=labelled_insertion_item_to_index_dict[start_item],
                stop_index=labelled_insertion_item_to_index_dict[stop_item]
            ))
            for start_item, stop_item in zip(start_items, stop_items)
        ]
        return list(zip(group_substrs, [
            [
                labelled_shape_item.shape_mobject
                for labelled_shape_item in labelled_shape_items[slice(*part_range)]
            ]
            for part_range in it.pairwise((0, *it.accumulate(range_lens)))
        ]))

    # Implemented in subclasses

    @classmethod
    @abstractmethod
    def _iter_command_matches(cls, string: str) -> Generator[re.Match[str], None, None]:
        pass

    @classmethod
    @abstractmethod
    def _get_command_flag(cls, match_obj: re.Match[str]) -> CommandFlag:
        pass

    @classmethod
    @abstractmethod
    def _replace_for_content(cls, match_obj: re.Match[str]) -> str:
        pass

    @classmethod
    @abstractmethod
    def _replace_for_matching(cls, match_obj: re.Match[str]) -> str:
        pass

    @classmethod
    @abstractmethod
    def _get_attr_dict_from_command_pair(
        cls, open_command: re.Match[str], close_command: re.Match[str],
    ) -> dict[str, str] | None:
        pass

    @classmethod
    @abstractmethod
    def _get_command_string(
        cls, attr_dict: dict[str, str], edge_flag: EdgeFlag, label: int | None
    ) -> str:
        pass

    # Selector

    def _iter_shape_mobject_lists_by_selector(self, selector: Selector) -> Generator[list[ShapeMobject], None, None]:
        return (
            shape_mobject_list
            for span in self._iter_spans_by_selector(selector, self._string)
            if (shape_mobject_list := self._get_shape_mobject_list_by_span(span, self._parsing_result.shape_items))
        )

    def select_parts(self, selector: Selector) -> ShapeMobject:
        return ShapeMobject().add(*(
            ShapeMobject().add(*shape_mobject_list)
            for shape_mobject_list in self._iter_shape_mobject_lists_by_selector(selector)
        ))

    def select_part(self, selector: Selector, index: int = 0) -> ShapeMobject:
        return ShapeMobject().add(*(
            list(self._iter_shape_mobject_lists_by_selector(selector))[index]
        ))
