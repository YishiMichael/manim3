from abc import abstractmethod
from colour import Color
import itertools as it
import re
from typing import Any, Callable, Iterable, Union
import warnings

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..mobjects.path_mobject import PathGroup
from ..mobjects.svg_mobject import SVGMobject
from ..custom_typing import *

#from manimlib.constants import WHITE
#from manimlib.logger import log
#from manimlib.mobject.svg.svg_mobject import SVGMobject
#from manimlib.mobject.types.vectorized_mobject import VGroup
#from manimlib.utils.color import color_to_rgb
#from manimlib.utils.color import rgb_to_hex
#from manimlib.utils.config_ops import digest_config

#from typing import TYPE_CHECKING

#if TYPE_CHECKING:
#    from colour import Color
#    from typing import Callable, Iterable, Union


__all__ = ["StringMobject"]


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
        self: Self,
        string: str,
        *,
        isolate: Selector = (),
        protect: Selector = (),
        **kwargs
    ):
        self.string: str = string
        #digest_config(self, kwargs)
        #if self.base_color is None:
        self.base_color: Color = Color("white")  # TODO

        configured_items = self.get_configured_items()
        labelled_spans, reconstruct_string = self.parse(
            string, isolate, protect, configured_items
        )
        self.labelled_spans: list[Span] = labelled_spans
        self.reconstruct_string: Callable[[
            tuple[int, int],
            tuple[int, int],
            Callable[[re.Match], str],
            Callable[[int, int, dict[str, str]], str]
        ], str] = reconstruct_string
        original_content = self.get_content(is_labelled=False)
        super().__init__(
            file_path=self.get_file_path_by_content(original_content),
            paint_settings={
                "fill_color": Color("white"),
                "fill_opacity": 1.0,
                "stroke_width": 0.0
            },
            **kwargs
        )
        self.labels = self.get_labels()

    #def get_file_path(self) -> str:
    #    original_content = self.get_content(is_labelled=False)
    #    return self.get_file_path_by_content(original_content)

    @abstractmethod
    def get_file_path_by_content(self: Self, content: str) -> str:
        raise NotImplementedError

    def get_labels(self: Self) -> list[int]:
        labels_count = len(self.labelled_spans)
        if labels_count == 1:
            return [0] * len(self.children)

        labelled_content = self.get_content(is_labelled=True)
        file_path = self.get_file_path_by_content(labelled_content)
        labelled_svg = SVGMobject(file_path)
        if len(self.children) != len(labelled_svg.children):
            warnings.warn(
                "Cannot align children of the labelled svg to the original svg. Skip the labelling process."
            )
            return [0] * len(self.children)

        self.rearrange_children_by_positions(labelled_svg)
        unrecognizable_colors = []
        labels = []
        for child in labelled_svg.children:
            label = self.color_to_int(child.fill_paint.color)
            if label >= labels_count:
                unrecognizable_colors.append(label)
                label = 0
            labels.append(label)

        if unrecognizable_colors:
            warnings.warn(
                "Unrecognizable color labels detected ({0}). The result could be unexpected.".format(
                    ", ".join(
                        self.int_to_hex(color)  # TODO
                        for color in unrecognizable_colors
                    )
                )
            )
        return labels

    def rearrange_children_by_positions(
        self: Self, labelled_svg: SVGMobject
    ) -> None:
        # Rearrange children of `labelled_svg` so that
        # each child is labelled by the nearest one of `labelled_svg`.
        # The correctness cannot be ensured, since the svg may
        # change significantly after inserting color commands.
        if not labelled_svg.children:
            return

        bb_0 = self.get_bounding_box()
        bb_1 = labelled_svg.get_bounding_box()
        labelled_svg.move_to(self).scale(bb_1.radius / bb_0.radius)

        distance_matrix = cdist(
            [child.get_center() for child in self.children],
            [child.get_center() for child in labelled_svg.children]
        )
        _, indices = linear_sum_assignment(distance_matrix)
        new_children = [
            labelled_svg.children[index]
            for index in indices
        ]
        labelled_svg.set_children(new_children)

    # Toolkits

    @staticmethod
    def find_spans_by_selector(selector: Selector, string: str) -> list[Span]:
        def clamp_index(index: int, l: int) -> int:
            return min(index, l) if index >= 0 else max(index + l, 0)

        def find_spans_by_single_selector(sel: Any) -> list[Span] | None:
            if isinstance(sel, str):
                return [
                    match_obj.span()
                    for match_obj in re.finditer(re.escape(sel), string)
                ]
            if isinstance(sel, re.Pattern):
                return [
                    match_obj.span()
                    for match_obj in sel.finditer(string)
                ]
            if isinstance(sel, tuple) and len(sel) == 2:
                start, end = sel
                if isinstance(start, int | None) and isinstance(end, int | None):
                    l = len(string)
                    span = (
                        0 if start is None else clamp_index(start, l),
                        l if end is None else clamp_index(end, l)
                    )
                    return [span]
            return None

        result = find_spans_by_single_selector(selector)
        if result is None:
            if not isinstance(selector, Iterable):
                raise TypeError(f"Invalid selector: '{selector}'")
            result = []
            for sel in selector:
                spans = find_spans_by_single_selector(sel)
                if spans is None:
                    raise TypeError(f"Invalid selector: '{sel}'")
                result.extend(spans)
        return list(filter(lambda span: span[0] <= span[1], result))

    @staticmethod
    def span_contains(span_0: Span, span_1: Span) -> bool:
        return span_0[0] <= span_1[0] and span_0[1] >= span_1[1]

    @staticmethod
    def color_to_int(color: Color) -> int:
        return int(color.hex_l[1:], 16)

    #@staticmethod
    #def color_to_hex(color: RGBAInt) -> str:
    #    return rgb_to_hex(color_to_rgb(color))

    #@staticmethod
    #def hex_to_int(rgb_hex: str) -> int:
    #    return int(rgb_hex[1:], 16)

    #@staticmethod
    #def int_to_hex(rgb_int: int) -> str:
    #    return f"#{rgb_int:06x}".upper()

    # Parsing

    @classmethod
    def parse(
        cls,
        string: str,
        isolate: Selector,
        protect: Selector,
        configured_items: list[tuple[Span, dict[str, str]]]
    ) -> tuple[
        list[Span],
        Callable[[
            tuple[int, int],
            tuple[int, int],
            Callable[[re.Match], str],
            Callable[[int, int, dict[str, str]], str]
        ], str]
    ]:
        # TODO: use Enums
        def get_substr(span: Span) -> str:
            return string[slice(*span)]

        #configured_items = self.get_configured_items()
        isolated_spans = cls.find_spans_by_selector(isolate, string)
        protected_spans = cls.find_spans_by_selector(protect, string)
        command_matches = cls.get_command_matches(string)

        def get_key(category: int, i: int, flag: int) -> tuple:
            def get_span_by_category(category: int, i: int) -> Span:
                if category == 0:
                    return configured_items[i][0]
                if category == 1:
                    return isolated_spans[i]
                if category == 2:
                    return protected_spans[i]
                return command_matches[i].span()

            index, paired_index = get_span_by_category(category, i)[::flag]
            return (
                index,
                flag * (2 if index != paired_index else -1),
                -paired_index,
                flag * category,
                flag * i
            )

        index_items = sorted([
            (category, i, flag)
            for category, item_length in enumerate((
                len(configured_items),
                len(isolated_spans),
                len(protected_spans),
                len(command_matches)
            ))
            for i in range(item_length)
            for flag in (1, -1)
        ], key=lambda t: get_key(*t))

        inserted_items = []
        labelled_items = []
        overlapping_spans = []
        level_mismatched_spans = []

        label = 1
        protect_level = 0
        bracket_stack = [0]
        bracket_count = 0
        open_command_stack = []
        open_stack = []
        for category, i, flag in index_items:
            if category >= 2:
                protect_level += flag
                if flag == 1 or category == 2:
                    continue
                inserted_items.append((i, 0))
                command_match = command_matches[i]
                command_flag = cls.get_command_flag(command_match)
                if command_flag == 1:
                    bracket_count += 1
                    bracket_stack.append(bracket_count)
                    open_command_stack.append((len(inserted_items), i))
                    continue
                if command_flag == 0:
                    continue
                pos, i_ = open_command_stack.pop()
                bracket_stack.pop()
                open_command_match = command_matches[i_]
                attr_dict = cls.get_attr_dict_from_command_pair(
                    open_command_match, command_match
                )
                if attr_dict is None:
                    continue
                span = (open_command_match.end(), command_match.start())
                labelled_items.append((span, attr_dict))
                inserted_items.insert(pos, (label, 1))
                inserted_items.insert(-1, (label, -1))
                label += 1
                continue
            if flag == 1:
                open_stack.append((
                    len(inserted_items), category, i, protect_level, bracket_stack.copy()
                ))
                continue
            span, attr_dict = configured_items[i] if category == 0 else (isolated_spans[i], {})
            pos, category_, i_, protect_level_, bracket_stack_ = open_stack.pop()
            if category_ != category or i_ != i:
                overlapping_spans.append(span)
                continue
            if protect_level_ or protect_level:
                continue
            if bracket_stack_ != bracket_stack:
                level_mismatched_spans.append(span)
                continue
            labelled_items.append((span, attr_dict))
            inserted_items.insert(pos, (label, 1))
            inserted_items.append((label, -1))
            label += 1
        labelled_items.insert(0, ((0, len(string)), {}))
        inserted_items.insert(0, (0, 1))
        inserted_items.append((0, -1))

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

        def reconstruct_string(
            start_item: tuple[int, int],
            end_item: tuple[int, int],
            command_replace_func: Callable[[re.Match], str],
            command_insert_func: Callable[[int, int, dict[str, str]], str]
        ) -> str:
            def get_edge_item(i: int, flag: int) -> tuple[Span, str]:
                if flag == 0:
                    match_obj = command_matches[i]
                    return (
                        match_obj.span(),
                        command_replace_func(match_obj)
                    )
                span, attr_dict = labelled_items[i]
                index = span[flag < 0]
                return (
                    (index, index),
                    command_insert_func(i, flag, attr_dict)
                )

            items = [
                get_edge_item(i, flag)
                for i, flag in inserted_items[slice(
                    inserted_items.index(start_item),
                    inserted_items.index(end_item) + 1
                )]
            ]
            pieces = [
                get_substr((start, end))
                for start, end in zip(
                    [interval_end for (_, interval_end), _ in items[:-1]],
                    [interval_start for (interval_start, _), _ in items[1:]]
                )
            ]
            interval_pieces = [piece for _, piece in items[1:-1]]
            return "".join(it.chain(*zip(pieces, (*interval_pieces, ""))))

        return [span for span, _ in labelled_items], reconstruct_string
        #self.labelled_spans: list[Span] = [span for span, _ in labelled_items]
        #self.reconstruct_string: Callable[[
        #    tuple[int, int],
        #    tuple[int, int],
        #    Callable[[re.Match], str],
        #    Callable[[int, int, dict[str, str]], str]
        #], str] = reconstruct_string

    def get_content(self: Self, is_labelled: bool) -> str:
        content = self.reconstruct_string(
            (0, 1), (0, -1),
            self.replace_for_content,
            lambda label, flag, attr_dict: self.get_command_string(
                attr_dict,
                is_end=flag < 0,
                label=label if is_labelled else None
            )
        )
        prefix, suffix = self.get_content_prefix_and_suffix(
            is_labelled=is_labelled
        )
        return "".join((prefix, content, suffix))

    @classmethod
    @abstractmethod
    def get_command_matches(cls, string: str) -> list[re.Match]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_command_flag(cls, match_obj: re.Match) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def replace_for_content(cls, match_obj: re.Match) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def replace_for_matching(cls, match_obj: re.Match) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_attr_dict_from_command_pair(
        cls, open_command: re.Match, close_command: re.Match,
    ) -> dict[str, str] | None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_command_string(
        cls, attr_dict: dict[str, str], is_end: bool, label: int | None
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_configured_items(self: Self) -> list[tuple[Span, dict[str, str]]]:
        raise NotImplementedError

    @abstractmethod
    def get_content_prefix_and_suffix(
        self: Self, is_labelled: bool
    ) -> tuple[str, str]:
        raise NotImplementedError

    # Selector

    def get_child_indices_list_by_span(
        self: Self, arbitrary_span: Span
    ) -> list[int]:
        return [
            child_index
            for child_index, label in enumerate(self.labels)
            if self.span_contains(arbitrary_span, self.labelled_spans[label])
        ]

    def get_specified_part_items(self: Self) -> list[tuple[str, list[int]]]:
        return [
            (
                self.string[slice(*span)],
                self.get_child_indices_list_by_span(span)
            )
            for span in self.labelled_spans[1:]
        ]

    def get_group_part_items(self: Self) -> list[tuple[str, list[int]]]:
        if not self.labels:
            return []

        range_lens, group_labels = zip(*(
            (len(list(grouper)), val)
            for val, grouper in it.groupby(self.labels)
        ))
        child_indices_lists = [
            list(range(*child_range))
            for child_range in it.pairwise((0, *it.accumulate(range_lens)))
        ]
        labelled_spans = self.labelled_spans
        start_items = [
            (group_labels[0], 1),
            *(
                (curr_label, 1)
                if self.span_contains(
                    labelled_spans[prev_label], labelled_spans[curr_label]
                )
                else (prev_label, -1)
                for prev_label, curr_label in it.pairwise(group_labels)
            )
        ]
        end_items = [
            *(
                (curr_label, -1)
                if self.span_contains(
                    labelled_spans[next_label], labelled_spans[curr_label]
                )
                else (next_label, 1)
                for curr_label, next_label in it.pairwise(group_labels)
            ),
            (group_labels[-1], -1)
        ]
        group_substrs = [
            re.sub(r"\s+", "", self.reconstruct_string(
                start_item, end_item,
                self.replace_for_matching,
                lambda label, flag, attr_dict: ""
            ))
            for start_item, end_item in zip(start_items, end_items)
        ]
        return list(zip(group_substrs, child_indices_lists))

    def get_child_indices_lists_by_selector(
        self: Self, selector: Selector
    ) -> list[list[int]]:
        return list(filter(
            lambda indices_list: indices_list,
            [
                self.get_child_indices_list_by_span(span)
                for span in self.find_spans_by_selector(selector, self.string)
            ]
        ))

    def build_parts_from_indices_lists(
        self: Self, indices_lists: list[list[int]]
    ) -> PathGroup:
        return PathGroup(*(
            PathGroup(*(
                self.children[child_index]
                for child_index in indices_list
            ))
            for indices_list in indices_lists
        ))

    def build_groups(self: Self) -> PathGroup:
        return self.build_parts_from_indices_lists([
            indices_list
            for _, indices_list in self.get_group_part_items()
        ])

    def select_parts(self: Self, selector: Selector) -> PathGroup:
        return self.build_parts_from_indices_lists(
            self.get_child_indices_lists_by_selector(selector)
        )

    def select_part(self: Self, selector: Selector, index: int = 0) -> PathGroup:
        return self.select_parts(selector)[index]

    def set_parts_color(self: Self, selector: Selector, color: ColorType):
        self.select_parts(selector).set_fill(color=color)
        return self

    #def set_parts_color_by_dict(self: Self, color_map: dict[Selector, ColorType]):
    #    for selector, color in color_map.items():
    #        self.set_parts_color(selector, color)
    #    return self

    def get_string(self: Self) -> str:
        return self.string
