__all__ = [
    "CommandFlag",
    "SpanEdgeFlag",
    "StringMobject"
]


from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from colour import Color
import itertools as it
import re
from typing import (
    Callable,
    #ClassVar,
    Generator
    #Iterable
)
import warnings

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..custom_typing import (
    ColorType,
    Real,
    Selector,
    Span
)
from ..mobjects.shape_mobject import ShapeMobject
from ..mobjects.svg_mobject import SVGMobject
from ..utils.color import ColorUtils


#class Span:
#    def __init__(self, start: int, stop: int):
#        assert start <= stop
#        self.start: int = start
#        self.stop: int = stop


class CommandFlag(Enum):
    OPEN = 1
    CLOSE = -1
    OTHER = 0


class SpanEdgeFlag(Enum):
    START = 1
    STOP = -1


#class ParsingSpanItem:
#    def __init__(self, *, span: Span):
#        self.span: Span = span


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class Label:
#    value: int


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
        return self.match_obj.span()


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
    edge_flag: SpanEdgeFlag
    #def __init__(self, *, label: int, edge_flag: SpanEdgeFlag):
    #    self.label: int = label
    #    self.edge_flag: SpanEdgeFlag = edge_flag
    #    #index = labelled_item.span[edge_flag.value < 0]
    #    #super().__init__(span=(index, index))

    @property
    def span(self) -> Span:
        index = self.labelled_item.span[self.edge_flag.value < 0]
        return (index, index)


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
    An abstract base class for `MTex` and `MarkupText`

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
    #CONFIG = {
    #    "height": None,
    #    "stroke_width": 0,
    #    "stroke_color": WHITE,
    #    "path_string_config": {
    #        "should_subdivide_sharp_curves": True,
    #        "should_remove_null_curves": True,
    #    },
    #    "base_color": WHITE,
    #    "isolate": (),
    #    "protect": (),
    #}

    def __init__(
        self,
        *,
        string: str,
        isolate: Selector = (),
        protect: Selector = (),
        configured_items_generator: Generator[tuple[Span, dict[str, str]], None, None],
        get_content_prefix_and_suffix: Callable[[bool], tuple[str, str]],
        get_svg_path: Callable[[str], str],
        width: Real | None = None,
        height: Real | None = None,
        frame_scale: Real | None = None
    ):
        #self.string: str = string
        #self.base_color: Color = Color("white")  # TODO: as a parameter

        parsing_result = self._parse(
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
        self._string: str = string
        self._parsing_result: ParsingResult = parsing_result
        shape_mobjects = [
            shape_item.shape_mobject
            for shape_item in parsing_result.shape_items
        ]
        super().__init__()
        self._shape_mobjects.extend(shape_mobjects)
        self.add(*shape_mobjects)

        #configured_items = self.get_configured_items()
        #labelled_spans, reconstruct_string = self.parse(
        #    string, isolate, protect, configured_items
        #)
        #self.labelled_spans: list[Span] = labelled_spans
        #self.reconstruct_string: Callable[[
        #    tuple[int, int],
        #    tuple[int, int],
        #    Callable[[re.Match[str]], str],
        #    Callable[[int, int, dict[str, str]], str]
        #], str] = reconstruct_string
        #original_content = self.get_content(is_labelled=False)
        #file_path = self.get_file_path_by_content(original_content)
        #super().__init__(
        #    file_path=file_path,
        #    width=width,
        #    height=height,
        #    frame_scale=frame_scale
        #)
        #self.labels = self.get_labels()

    #def get_file_path(self) -> str:
    #    original_content = self.get_content(is_labelled=False)
    #    return self.get_file_path_by_content(original_content)

    #@abstractmethod
    #def get_file_path_by_content(self, content: str) -> str:
    #    pass

    #def get_labels(self) -> list[int]:
    #    labels_count = len(self.labelled_spans)
    #    if labels_count == 1:
    #        return [0] * len(self._shape_mobjects)

    #    labelled_content = self.get_content(is_labelled=True)
    #    file_path = self.get_file_path_by_content(labelled_content)
    #    labelled_svg = SVGMobject(
    #        file_path=file_path,
    #        #paint_settings={
    #        #    "fill_color": Color("white"),
    #        #    "fill_opacity": 1.0,
    #        #    "disable_stroke": True
    #        #    #"stroke_width": 0.0,
    #        #    #"stroke_opacity": 0.0
    #        #}
    #    )
    #    if len(self._shape_mobjects) != len(labelled_svg._shape_mobjects):
    #        warnings.warn(
    #            "Cannot align children of the labelled svg to the original svg. Skip the labelling process."
    #        )
    #        return [0] * len(self._shape_mobjects)

    #    self._rearrange_labelled_shapes_by_positions(labelled_svg)
    #    unrecognizable_colors = []
    #    labels = []
    #    for child in labelled_svg._shape_mobjects:
    #        child_color = child._color_
    #        #assert not isinstance(child_color, Callable)
    #        label = self.color_to_int(child_color)
    #        if label >= labels_count:
    #            unrecognizable_colors.append(label)
    #            label = 0
    #        labels.append(label)

    #    if unrecognizable_colors:
    #        warnings.warn(
    #            "Unrecognizable color labels detected ({0}). The result could be unexpected.".format(
    #                ", ".join(
    #                    self.int_to_hex(color)  # TODO
    #                    for color in unrecognizable_colors
    #                )
    #            )
    #        )
    #    return labels

    @classmethod
    def _rearrange_labelled_shapes_by_positions(
        cls, plain_svg: SVGMobject, labelled_svg: SVGMobject
    ) -> None:
        # Rearrange children of `labelled_svg` so that
        # each child is labelled by the nearest one of `labelled_svg`.
        # The correctness cannot be ensured, since the svg may
        # change significantly after inserting color commands.
        if not labelled_svg._shape_mobjects:
            return

        bb_0 = plain_svg.get_bounding_box()
        bb_1 = labelled_svg.get_bounding_box()
        labelled_svg.move_to(plain_svg).scale(bb_1.radius / bb_0.radius)

        distance_matrix = cdist(
            [child.get_center() for child in plain_svg._shape_mobjects],
            [child.get_center() for child in labelled_svg._shape_mobjects]
        )
        _, indices = linear_sum_assignment(distance_matrix)
        new_children = [
            labelled_svg._shape_mobjects[index]
            for index in indices
        ]
        labelled_svg.set_children(new_children)

    # Toolkits

    @classmethod
    def _iter_spans_by_selector(cls, selector: Selector, string: str) -> Generator[Span, None, None]:
        def iter_spans_by_single_selector(sel: str | re.Pattern[str] | slice, string: str) -> Generator[Span, None, None]:
            if isinstance(sel, str):
                for match_obj in re.finditer(re.escape(sel), string, flags=re.MULTILINE):
                    yield match_obj.span()
            elif isinstance(sel, re.Pattern):
                for match_obj in sel.finditer(string):
                    yield match_obj.span()
            elif isinstance(sel, slice):
                if sel.start > sel.stop:
                    warnings.warn(f"Caught a span ({sel.start}, {sel.stop}). Perhaps a typo?")
                else:
                    yield (sel.start, sel.stop)
            else:
                raise TypeError(f"Invalid selector: '{sel}'")

        if isinstance(selector, str | re.Pattern | slice):
            yield from iter_spans_by_single_selector(selector, string)
        else:
            for sel in selector:
                yield from iter_spans_by_single_selector(sel, string)


        #def clamp_index(index: int, l: int) -> int:
        #    return min(index, l) if index >= 0 else max(index + l, 0)

        #def find_spans_by_single_selector(sel: Any) -> list[Span] | None:
        #    if isinstance(sel, str):
        #        return [
        #            match_obj.span()
        #            for match_obj in re.finditer(re.escape(sel), string)
        #        ]
        #    if isinstance(sel, re.Pattern):
        #        return [
        #            match_obj.span()
        #            for match_obj in sel.finditer(string)
        #        ]
        #    if isinstance(sel, tuple) and len(sel) == 2:
        #        start, end = sel
        #        if isinstance(start, int | None) and isinstance(end, int | None):
        #            l = len(string)
        #            span = (
        #                0 if start is None else clamp_index(start, l),
        #                l if end is None else clamp_index(end, l)
        #            )
        #            return [span]
        #    return None

        #result = find_spans_by_single_selector(selector)
        #if result is None:
        #    if not isinstance(selector, Iterable):
        #        raise TypeError(f"Invalid selector: '{selector}'")
        #    result = []
        #    for sel in selector:
        #        spans = find_spans_by_single_selector(sel)
        #        if spans is None:
        #            raise TypeError(f"Invalid selector: '{sel}'")
        #        result.extend(spans)
        #return list(filter(lambda span: span[0] <= span[1], result))

    @classmethod
    def _span_contains(cls, span_0: Span, span_1: Span) -> bool:
        return span_0[0] <= span_1[0] and span_0[1] >= span_1[1]

    @classmethod
    def _color_to_int(cls, color: ColorType) -> int:
        c = Color()
        c.rgb, _ = ColorUtils.decompose_color(color)  # TODO
        return int(c.hex_l[1:], 16)

    #@staticmethod
    #def color_to_hex(color: RGBAInt) -> str:
    #    return rgb_to_hex(color_to_rgb(color))

    #@staticmethod
    #def hex_to_int(rgb_hex: str) -> int:
    #    return int(rgb_hex[1:], 16)

    @classmethod
    def _int_to_hex(cls, rgb_int: int) -> str:
        return f"#{rgb_int:06x}".upper()

    # Parsing

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
        # TODO: use Enums
        def get_substr(span: Span) -> str:
            return string[slice(*span)]

        #configured_items = self.get_configured_items()
        #isolated_spans = list(cls._iter_spans_by_selector(isolate, string))
        #protected_spans = list(cls._iter_spans_by_selector(protect, string))
        #command_matches = cls.get_command_matches(string)

        #def get_key(category: int, i: int, flag: int) -> tuple:
        #    def get_span_by_category(category: int, i: int) -> Span:
        #        if category == 0:
        #            return configured_items[i][0]
        #        if category == 1:
        #            return isolated_spans[i]
        #        if category == 2:
        #            return protected_spans[i]
        #        return command_matches[i].span()

        #    index, paired_index = get_span_by_category(category, i)[::flag]
        #    return (
        #        index,
        #        flag * (2 if index != paired_index else -1),
        #        -paired_index,
        #        flag * category,
        #        flag * i
        #    )

        def get_key(
            index_item: tuple[ConfiguredItem | IsolatedItem | ProtectedItem | CommandItem, SpanEdgeFlag, int, int]
        ) -> tuple[int, int, int, int, int]:
            span_item, edge_flag, priority, i = index_item
            #priority = (
            #    ConfiguredItem,
            #    IsolatedItem,
            #    ProtectedItem,
            #    CommandItem
            #).index(span_item.__class__) + 1
            flag_value = int(edge_flag.value)
            index, paired_index = span_item.span[::flag_value]
            return (
                index,
                flag_value * (2 if index != paired_index else -1),
                -paired_index,
                flag_value * priority,
                flag_value * i
            )

        index_items: list[tuple[ConfiguredItem | IsolatedItem | ProtectedItem | CommandItem, SpanEdgeFlag, int, int]] = sorted((
            (span_item, edge_flag, priority, i)
            for priority, span_item_iter in enumerate((
                (ConfiguredItem(span=span, attr_dict=attr_dict) for span, attr_dict in configured_items_generator),
                (IsolatedItem(span=span) for span in cls._iter_spans_by_selector(isolate, string)),
                (ProtectedItem(span=span) for span in cls._iter_spans_by_selector(protect, string)),
                (CommandItem(match_obj=match_obj) for match_obj in cls._iter_command_matches(string))
            ), start=1)
            for i, span_item in enumerate(span_item_iter)
            for edge_flag in SpanEdgeFlag
        ), key=get_key)
        #index_items = sorted([
        #    (category, i, flag)
        #    for category, item_length in enumerate((
        #        len(configured_items),
        #        len(isolated_spans),
        #        len(protected_spans),
        #        len(command_matches)
        #    ))
        #    for i in range(item_length)
        #    for flag in (1, -1)
        #], key=lambda t: get_key(*t))

        labelled_items: list[LabelledItem] = []
        reconstruct_items: list[CommandItem | LabelledInsertionItem] = []
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
            reconstruct_items.insert(pos, LabelledInsertionItem(
                labelled_item=labelled_item,
                edge_flag=SpanEdgeFlag.START
            ))
            reconstruct_items.append(LabelledInsertionItem(
                labelled_item=labelled_item,
                edge_flag=SpanEdgeFlag.STOP
            ))

        for span_item, edge_flag, _, _ in index_items:
            if isinstance(span_item, ProtectedItem | CommandItem):
                protect_level += edge_flag.value
                if isinstance(span_item, ProtectedItem):
                    continue
                if edge_flag == SpanEdgeFlag.START:
                    continue
                command_item = span_item
                command_flag = cls._get_command_flag(command_item.match_obj)
                if command_flag == CommandFlag.OPEN:
                    bracket_count += 1
                    bracket_stack.append(bracket_count)
                    reconstruct_items.append(command_item)
                    open_command_stack.append((len(reconstruct_items), command_item))
                elif command_flag == CommandFlag.OTHER:
                    reconstruct_items.append(command_item)
                else:
                    pos, open_command_item = open_command_stack.pop()
                    bracket_stack.pop()
                    attr_dict = cls._get_attr_dict_from_command_pair(
                        open_command_item.match_obj, command_item.match_obj
                    )
                    if attr_dict is not None:
                        add_labelled_item(LabelledItem(
                            label=next(label_counter),
                            span=(open_command_item.span[1], command_item.span[0]),
                            attr_dict=attr_dict
                        ), pos)
                        #labelled_items.append(labelled_item)
                        #reconstruct_items.insert(pos, LabelledInsertionItem(
                        #    labelled_item=labelled_item,
                        #    edge_flag=SpanEdgeFlag.START
                        #))
                        #reconstruct_items.append(LabelledInsertionItem(
                        #    labelled_item=labelled_item,
                        #    edge_flag=SpanEdgeFlag.STOP
                        #))
                        #label += 1
                    reconstruct_items.append(command_item)
                continue
            if edge_flag == SpanEdgeFlag.START:
                open_stack.append((
                    len(reconstruct_items), span_item, protect_level, bracket_stack.copy()
                ))
                continue
            span = span_item.span
            #attr_dict = span_item.attr_dict if isinstance(span_item, ConfiguredItem) else {}
            #span, attr_dict = configured_items[i] if category == 0 else (isolated_spans[i], {})
            pos, open_span_item, open_protect_level, open_bracket_stack = open_stack.pop()
            #if category_ != category or i_ != i:
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
            #labelled_items.append(labelled_item)
            #reconstruct_items.insert(pos, LabelledInsertionItem(
            #    labelled_item=labelled_item,
            #    edge_flag=SpanEdgeFlag.START
            #))
            #reconstruct_items.append(LabelledInsertionItem(
            #    labelled_item=labelled_item,
            #    edge_flag=SpanEdgeFlag.STOP
            #))
            #label += 1
        add_labelled_item(LabelledItem(
            label=0,
            span=(0, len(string)),
            attr_dict={}
        ), 0)
        #labelled_items.append(labelled_item)
        #reconstruct_items.insert(0, LabelledInsertionItem(
        #    labelled_item=labelled_item,
        #    edge_flag=SpanEdgeFlag.START
        #))
        #reconstruct_items.append(LabelledInsertionItem(
        #    labelled_item=labelled_item,
        #    edge_flag=SpanEdgeFlag.STOP
        #))

        #inserted_items = []
        #labelled_items = []
        #overlapping_spans = []
        #level_mismatched_spans = []

        #label = 1
        #protect_level = 0
        #bracket_stack = [0]
        #bracket_count = 0
        #open_command_stack = []
        #open_stack = []
        #for category, i, flag in index_items:
        #    if category >= 2:
        #        protect_level += flag
        #        if flag == 1 or category == 2:
        #            continue
        #        inserted_items.append((i, 0))
        #        command_match = command_matches[i]
        #        command_flag = cls.get_command_flag(command_match)
        #        if command_flag == CommandFlag.OPEN:
        #            bracket_count += 1
        #            bracket_stack.append(bracket_count)
        #            open_command_stack.append((len(inserted_items), i))
        #            continue
        #        if command_flag == CommandFlag.OTHER:
        #            continue
        #        pos, i_ = open_command_stack.pop()
        #        bracket_stack.pop()
        #        open_command_match = command_matches[i_]
        #        attr_dict = cls.get_attr_dict_from_command_pair(
        #            open_command_match, command_match
        #        )
        #        if attr_dict is None:
        #            continue
        #        span = (open_command_match.end(), command_match.start())
        #        labelled_items.append((span, attr_dict))
        #        inserted_items.insert(pos, (label, 1))
        #        inserted_items.insert(-1, (label, -1))
        #        label += 1
        #        continue
        #    if flag == 1:
        #        open_stack.append((
        #            len(inserted_items), category, i, protect_level, bracket_stack.copy()
        #        ))
        #        continue
        #    span, attr_dict = configured_items[i] if category == 0 else (isolated_spans[i], {})
        #    pos, category_, i_, protect_level_, bracket_stack_ = open_stack.pop()
        #    if category_ != category or i_ != i:
        #        overlapping_spans.append(span)
        #        continue
        #    if protect_level_ or protect_level:
        #        continue
        #    if bracket_stack_ != bracket_stack:
        #        level_mismatched_spans.append(span)
        #        continue
        #    labelled_items.append((span, attr_dict))
        #    inserted_items.insert(pos, (label, 1))
        #    inserted_items.append((label, -1))
        #    label += 1
        #labelled_items.insert(0, ((0, len(string)), {}))
        #inserted_items.insert(0, (0, 1))
        #inserted_items.append((0, -1))

        if overlapping_spans:
            warnings.warn(
                "Partly overlapping substrings detected: {0}".format(
                    ", ".join(
                        f"'{get_substr(span)}'"
                        for span in overlapping_spans
                    )
                )
            )
        if level_mismatched_spans:
            warnings.warn(
                "Cannot handle substrings: {0}".format(
                    ", ".join(
                        f"'{get_substr(span)}'"
                        for span in level_mismatched_spans
                    )
                )
            )

        #def get_replaced_str(
        #    reconstruct_item: CommandItem | LabelledInsertionItem,
        #    command_replace_func: Callable[[re.Match[str]], str],
        #    command_insert_func: Callable[[int, SpanEdgeFlag, dict[str, str]], str]
        #) -> str:
        #    if isinstance(reconstruct_item, CommandItem):
        #        return command_replace_func(reconstruct_item.match_obj)
        #    return command_insert_func(
        #        reconstruct_item.labelled_item.label,
        #        reconstruct_item.edge_flag,
        #        reconstruct_item.labelled_item.attr_dict
        #    )

        def get_replaced_pieces(
            command_replace_func: Callable[[re.Match[str]], str],
            command_insert_func: Callable[[int, SpanEdgeFlag, dict[str, str]], str]
        ) -> list[str]:
            return [
                command_replace_func(reconstruct_item.match_obj)
                if isinstance(reconstruct_item, CommandItem)
                else command_insert_func(
                    reconstruct_item.labelled_item.label,
                    reconstruct_item.edge_flag,
                    reconstruct_item.labelled_item.attr_dict
                )
                for reconstruct_item in reconstruct_items
            ]

        #content_labelled_pieces, content_unlabelled_pieces = (
        #    get_replaced_pieces(
        #        cls._replace_for_content,
        #        lambda label, edge_flag, attr_dict: cls._get_command_string(
        #            attr_dict,
        #            edge_flag=edge_flag,
        #            label=label if is_labelled else None
        #        )
        #    )
        #    for is_labelled in (True, False)
        #)


        labelled_insertion_item_to_index_dict = {
            (reconstruct_item.labelled_item.label, reconstruct_item.edge_flag): index
            for index, reconstruct_item in enumerate(reconstruct_items)
            if isinstance(reconstruct_item, LabelledInsertionItem)
        }

        def reconstruct_string(
            original_pieces: list[str],
            replaced_pieces: list[str],
            start_item: tuple[int, SpanEdgeFlag],
            stop_item: tuple[int, SpanEdgeFlag]
        ) -> str:
            start_index = labelled_insertion_item_to_index_dict[start_item]
            stop_index = labelled_insertion_item_to_index_dict[stop_item]
            return "".join(it.chain(*zip(
                original_pieces[start_index:stop_index],
                (*replaced_pieces[start_index + 1:stop_index], ""),
                strict=True
            )))

        reconstruct_spans = [reconstruct_item.span for reconstruct_item in reconstruct_items]
        original_pieces = [
            get_substr((start, stop))
            for start, stop in zip(
                [interval_stop for (_, interval_stop) in reconstruct_spans[:-1]],
                [interval_start for (interval_start, _) in reconstruct_spans[1:]]
            )
        ]

        def get_content(is_labelled: bool) -> str:
            content_replaced_pieces = get_replaced_pieces(
                command_replace_func=cls._replace_for_content,
                command_insert_func=lambda label, edge_flag, attr_dict: cls._get_command_string(
                    attr_dict,
                    edge_flag=edge_flag,
                    label=label if is_labelled else None
                )
            )
            content = reconstruct_string(
                original_pieces=original_pieces,
                replaced_pieces=content_replaced_pieces,
                start_item=(0, SpanEdgeFlag.START),
                stop_item=(0, SpanEdgeFlag.STOP)
            )
            prefix, suffix = get_content_prefix_and_suffix(is_labelled)
            return "".join((prefix, content, suffix))


        #def reconstruct_string(
        #    start_item: tuple[int, SpanEdgeFlag],
        #    stop_item: tuple[int, SpanEdgeFlag],
        #    command_replace_func: Callable[[re.Match[str]], str],
        #    command_insert_func: Callable[[int, SpanEdgeFlag, dict[str, str]], str]
        #) -> str:
        #    def get_replace_item(reconstruct_item: CommandItem | LabelledInsertionItem) -> tuple[Span, str]:
        #        if isinstance(reconstruct_item, CommandItem):
        #            return (
        #                reconstruct_item.span,
        #                command_replace_func(reconstruct_item.match_obj)
        #            )
        #        label = reconstruct_item.labelled_item.label
        #        edge_flag = reconstruct_item.edge_flag
        #        labelled_item = reconstruct_item.labelled_item
        #        index = labelled_item.span[edge_flag.value < 0]
        #        return (
        #            (index, index),
        #            command_insert_func(label, edge_flag, labelled_item.attr_dict)
        #        )

        #        #if flag == 0:
        #        #    match_obj = command_matches[i]
        #        #    return (
        #        #        match_obj.span(),
        #        #        command_replace_func(match_obj)
        #        #    )
        #        #span, attr_dict = labelled_items[i]
        #        #index = span[flag < 0]
        #        #return (
        #        #    (index, index),
        #        #    command_insert_func(i, flag, attr_dict)
        #        #)

        #    #items = [
        #    #    get_edge_item(i, flag)
        #    #    for i, flag in inserted_items[slice(
        #    #        inserted_items.index(start_item),
        #    #        inserted_items.index(end_item) + 1
        #    #    )]
        #    #]

        #    replace_items = [
        #        #(
        #        #    reconstruct_item.span,
        #        #    command_replace_func(reconstruct_item.match_obj)
        #        #)
        #        #if isinstance(reconstruct_item, CommandItem)
        #        #else (
        #        #    (index := (labelled_item := labelled_items[reconstruct_item.label]).span[reconstruct_item.edge_flag.value < 0], index),
        #        #    command_insert_func(reconstruct_item.label, reconstruct_item.edge_flag, labelled_item.attr_dict)
        #        #)
        #        #(
        #        #    reconstruct_item.span,
        #        #    command_replace_func(reconstruct_item.match_obj)
        #        #    if isinstance(reconstruct_item, CommandItem)
        #        #    else command_insert_func(
        #        #        labelled_items.index(reconstruct_item.labelled_item),
        #        #        reconstruct_item.edge_flag,
        #        #        reconstruct_item.labelled_item.attr_dict
        #        #    )
        #        #)
        #        get_replace_item(reconstruct_item)
        #        for reconstruct_item in reconstruct_items[slice(
        #            reconstruct_items.index(start_item),
        #            reconstruct_items.index(stop_item) + 1
        #        )]
        #    ]
        #    pieces = [
        #        get_substr((start, stop))
        #        for start, stop in zip(
        #            [interval_stop for (_, interval_stop), _ in replace_items[:-1]],
        #            [interval_start for (interval_start, _), _ in replace_items[1:]]
        #        )
        #    ]
        #    interval_pieces = [piece for _, piece in replace_items[1:-1]]
        #    return "".join(it.chain(*zip(pieces, (*interval_pieces, ""))))

        #return [labelled_span.span for labelled_span in labelled_items], reconstruct_string
        #self.labelled_spans: list[Span] = [span for span, _ in labelled_items]
        #self.reconstruct_string: Callable[[
        #    tuple[int, int],
        #    tuple[int, int],
        #    Callable[[re.Match[str]], str],
        #    Callable[[int, int, dict[str, str]], str]
        #], str] = reconstruct_string


        plain_svg = SVGMobject(
            file_path=get_svg_path(get_content(is_labelled=False)),
            width=width,
            height=height,
            frame_scale=frame_scale
        )

        labels_count = len(labelled_items)
        if labels_count == 1:
            labelled_shape_items = [
                LabelledShapeItem(
                    label=0,
                    shape_mobject=plain_shape
                )
                for plain_shape in plain_svg._shape_mobjects
            ]
            #return [0] * len(self._shape_mobjects)

        else:
            #labelled_content = self.get_content(is_labelled=True)
            #file_path = self.get_file_path_by_content(labelled_content)
            #labelled_svg = SVGMobject(
            #    file_path=file_path,
            #    #paint_settings={
            #    #    "fill_color": Color("white"),
            #    #    "fill_opacity": 1.0,
            #    #    "disable_stroke": True
            #    #    #"stroke_width": 0.0,
            #    #    #"stroke_opacity": 0.0
            #    #}
            #)
            labelled_svg = SVGMobject(
                file_path=get_svg_path(get_content(is_labelled=True))
            )
            if len(plain_svg._shape_mobjects) != len(labelled_svg._shape_mobjects):
                warnings.warn(
                    "Cannot align children of the labelled svg to the original svg. Skip the labelling process."
                )
                labelled_shape_items = [
                    LabelledShapeItem(
                        label=0,
                        shape_mobject=plain_shape
                    )
                    for plain_shape in plain_svg._shape_mobjects
                ]
                #return [0] * len(plain_svg._shape_mobjects)

            else:
                cls._rearrange_labelled_shapes_by_positions(plain_svg, labelled_svg)
                unrecognizable_colors: list[int] = []
                labelled_shape_items: list[LabelledShapeItem] = []
                for plain_shape, labelled_shape in zip(plain_svg._shape_mobjects, labelled_svg._shape_mobjects, strict=True):
                    #mobject_color = mobject._color_
                    #assert not isinstance(mobject_color, Callable)
                    label = cls._color_to_int(labelled_shape._color_)
                    if label >= labels_count:
                        unrecognizable_colors.append(label)
                        label = 0
                    labelled_shape_items.append(LabelledShapeItem(
                        label=label,
                        shape_mobject=plain_shape
                    ))

                if unrecognizable_colors:
                    warnings.warn(
                        "Unrecognizable color labels detected ({0}). The result could be unexpected.".format(
                            ", ".join(
                                cls._int_to_hex(color)  # TODO
                                for color in unrecognizable_colors
                            )
                        )
                    )

        label_to_span_dict = {
            labelled_item.label: labelled_item.span
            for labelled_item in labelled_items
        }
        shape_items = [
            ShapeItem(
                span=label_to_span_dict[labelled_shape_item.label],
                shape_mobject=labelled_shape_item.shape_mobject
            )
            for labelled_shape_item in labelled_shape_items
        ]
        specified_part_items = [
            (
                string[slice(*labelled_item.span)],
                cls._get_shape_mobject_list_by_span(labelled_item.span, shape_items)
            )
            for labelled_item in labelled_items
        ]


        if not labelled_shape_items:
            group_part_items: list[tuple[str, list[ShapeMobject]]] = []

        else:
            range_lens, group_labels = zip(*(
                (len(list(grouper)), val)
                for val, grouper in it.groupby(labelled_shape_item.label for labelled_shape_item in labelled_shape_items)
            ))
            #child_indices_lists = [
            #    list(range(*child_range))
            #    for child_range in it.pairwise((0, *it.accumulate(range_lens)))
            #]
            #labelled_spans = self.labelled_spans
            start_items = [
                (group_labels[0], SpanEdgeFlag.START),
                *(
                    (curr_label, SpanEdgeFlag.START)
                    if cls._span_contains(
                        label_to_span_dict[prev_label], label_to_span_dict[curr_label]
                    )
                    else (prev_label, SpanEdgeFlag.STOP)
                    for prev_label, curr_label in it.pairwise(group_labels)
                )
            ]
            stop_items = [
                *(
                    (curr_label, SpanEdgeFlag.STOP)
                    if cls._span_contains(
                        label_to_span_dict[next_label], label_to_span_dict[curr_label]
                    )
                    else (next_label, SpanEdgeFlag.START)
                    for curr_label, next_label in it.pairwise(group_labels)
                ),
                (group_labels[-1], SpanEdgeFlag.STOP)
            ]
            matching_replaced_pieces = get_replaced_pieces(
                command_replace_func=cls._replace_for_matching,
                command_insert_func=lambda label, flag, attr_dict: ""
            )
            group_substrs = [
                re.sub(r"\s+", "", reconstruct_string(
                    original_pieces=original_pieces,
                    replaced_pieces=matching_replaced_pieces,
                    start_item=start_item,
                    stop_item=stop_item
                ))
                for start_item, stop_item in zip(start_items, stop_items)
            ]
            group_part_items = list(zip(group_substrs, [
                [
                    labelled_shape_item.shape_mobject
                    for labelled_shape_item in labelled_shape_items[slice(*part_range)]
                ]
                for part_range in it.pairwise((0, *it.accumulate(range_lens)))
            ]))

        return ParsingResult(
            shape_items=shape_items,
            specified_part_items=specified_part_items,
            group_part_items=group_part_items
        )


    #def get_content(self, is_labelled: bool) -> str:
    #    content = self.reconstruct_string(
    #        (0, 1), (0, -1),
    #        self.replace_for_content,
    #        lambda label, flag, attr_dict: self.get_command_string(
    #            attr_dict,
    #            is_stop=flag < 0,
    #            label=label if is_labelled else None
    #        )
    #    )
    #    prefix, suffix = self.get_content_prefix_and_suffix(
    #        is_labelled=is_labelled
    #    )
    #    return "".join((prefix, content, suffix))

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
        cls, attr_dict: dict[str, str], edge_flag: SpanEdgeFlag, label: int | None
    ) -> str:
        pass

    #@abstractmethod
    #def get_configured_items(self) -> list[ConfiguredItem]:
    #    pass

    #@abstractmethod
    #def get_content_prefix_and_suffix(
    #    self, is_labelled: bool
    #) -> tuple[str, str]:
    #    pass

    # Selector

    #def get_child_indices_list_by_span(
    #    self, arbitrary_span: Span
    #) -> list[int]:
    #    return [
    #        child_index
    #        for child_index, label in enumerate(self.labels)
    #        if self._span_contains(arbitrary_span, self.labelled_spans[label])
    #    ]

    @classmethod
    def _get_shape_mobject_list_by_span(cls, arbitrary_span: Span, shape_items: list[ShapeItem]) -> list[ShapeMobject]:
        return [
            shape_item.shape_mobject
            for shape_item in shape_items
            if cls._span_contains(arbitrary_span, shape_item.span)
        ]

    #@classmethod
    #def _get_specified_part_items(cls, string: str, labelled_items: list[LabelledItem]) -> list[tuple[str, list[ShapeMobject]]]:
    #    return 
        #return [
        #    (
        #        self.string[slice(*span)],
        #        self.get_child_indices_list_by_span(span)
        #    )
        #    for span in self.labelled_spans[1:]
        #]

    #@classmethod
    #def _get_group_part_items(cls, ) -> list[tuple[str, list[ShapeMobject]]]:

        #if not self.labels:
        #    return []

        #range_lens, group_labels = zip(*(
        #    (len(list(grouper)), val)
        #    for val, grouper in it.groupby(self.labels)
        #))
        #child_indices_lists = [
        #    list(range(*child_range))
        #    for child_range in it.pairwise((0, *it.accumulate(range_lens)))
        #]
        #labelled_spans = self.labelled_spans
        #start_items = [
        #    (group_labels[0], 1),
        #    *(
        #        (curr_label, 1)
        #        if self._span_contains(
        #            labelled_spans[prev_label], labelled_spans[curr_label]
        #        )
        #        else (prev_label, -1)
        #        for prev_label, curr_label in it.pairwise(group_labels)
        #    )
        #]
        #stop_items = [
        #    *(
        #        (curr_label, -1)
        #        if self._span_contains(
        #            labelled_spans[next_label], labelled_spans[curr_label]
        #        )
        #        else (next_label, 1)
        #        for curr_label, next_label in it.pairwise(group_labels)
        #    ),
        #    (group_labels[-1], -1)
        #]
        #group_substrs = [
        #    re.sub(r"\s+", "", self.reconstruct_string(
        #        start_item, stop_item,
        #        self.replace_for_matching,
        #        lambda label, flag, attr_dict: ""
        #    ))
        #    for start_item, stop_item in zip(start_items, stop_items)
        #]
        #return list(zip(group_substrs, child_indices_lists))

    #def get_child_indices_lists_by_selector(
    #    self, selector: Selector
    #) -> list[list[int]]:
    #    return list(filter(
    #        lambda indices_list: indices_list,
    #        [
    #            self._get_shape_mobject_list_by_span(span, self._parsing_result.labelled_shape_items)
    #            for span in self._iter_spans_by_selector(selector, self._string)
    #        ]
    #    ))

    #def build_part_from_indices_list(
    #    self, indices_list: list[int]
    #) -> ShapeMobject:
    #    return ShapeMobject().add(*(
    #        self._shape_mobjects[child_index]
    #        for child_index in indices_list
    #    ))

    #def build_parts_from_indices_lists(
    #    self, indices_lists: list[list[int]]
    #) -> ShapeMobject:
    #    return ShapeMobject().add(*(
    #        self.build_part_from_indices_list(indices_list)
    #        for indices_list in indices_lists
    #    ))

    #def build_groups(self) -> ShapeMobject:
    #    return self.build_parts_from_indices_lists([
    #        indices_list
    #        for _, indices_list in self.get_group_part_items()
    #    ])

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
        #return self.build_parts_from_indices_lists(
        #    self.get_child_indices_lists_by_selector(selector)
        #)

    def select_part(self, selector: Selector, index: int = 0) -> ShapeMobject:
        return ShapeMobject().add(*(
            list(self._iter_shape_mobject_lists_by_selector(selector))[index]
        ))

    def set_parts_color(self, selector: Selector, color: ColorType):
        self.select_parts(selector).set_fill(color=color)
        return self

    #def set_parts_color_by_dict(self, color_map: dict[Selector, ColorType]):
    #    for selector, color in color_map.items():
    #        self.set_parts_color(selector, color)
    #    return self

    #def get_string(self) -> str:
    #    return self.string
