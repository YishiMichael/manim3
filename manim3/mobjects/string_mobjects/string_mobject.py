import itertools as it
import pathlib
import re
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    Iterable,
    Iterator,
    TypedDict
)

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ...constants.custom_typing import SelectorT
from ...utils.color_utils import ColorUtils
from ..mobject_io import MobjectIO
from ..shape_mobjects.shape_mobject import ShapeMobject
from ..svg_mobject import (
    SVGMobjectIO,
    ShapeMobjectJSON
)


class CommandFlag(Enum):
    OPEN = 1
    CLOSE = -1
    OTHER = 0


class EdgeFlag(Enum):
    START = 1
    STOP = -1

    def negate(self) -> "EdgeFlag":
        return EdgeFlag(-self.value)


@dataclass(
    unsafe_hash=True,
    frozen=True,
    slots=True
)
class Span:
    start: int
    stop: int

    def contains(
        self,
        span: "Span"
    ) -> bool:
        return self.start <= span.start and self.stop >= span.stop

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)

    def get_edge_index(
        self,
        edge_flag: EdgeFlag
    ) -> int:
        return self.start if edge_flag == EdgeFlag.START else self.stop


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ConfiguredItem:
    span: Span
    attrs: dict[str, str]


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
class LabelledInsertionItem:
    label: int
    edge_flag: EdgeFlag
    attrs: dict[str, str]
    index: int

    @property
    def span(self) -> Span:
        index = self.index
        return Span(index, index)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class StringMobjectInputData:
    string: str
    isolate: list[Span]
    protect: list[Span]


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class StringMobjectOutputData:
    shape_mobjects: list[ShapeMobject]
    spans: list[Span]
    labelled_part_items: list[tuple[str, list[int]]]
    group_part_items: list[tuple[str, list[int]]]


class StringMobjectJSON(TypedDict):
    shape_mobjects: list[ShapeMobjectJSON]
    spans: list[list[int]]
    labelled_part_strings: list[str]
    labelled_part_indices: list[list[int]]
    group_part_strings: list[str]
    group_part_indices: list[list[int]]


class StringMobjectIO(MobjectIO[StringMobjectInputData, StringMobjectOutputData, StringMobjectJSON]):
    __slots__ = ()

    @classmethod
    def generate(
        cls,
        input_data: StringMobjectInputData,
        temp_path: pathlib.Path
    ) -> StringMobjectOutputData:
        string = input_data.string
        global_attrs = cls._get_global_attrs(input_data, temp_path)
        local_attrs = cls._get_local_attrs(input_data, temp_path)

        label_to_span_dict, replaced_items = cls._get_label_to_span_dict_and_replaced_items(
            string=string,
            isolate=input_data.isolate,
            protect=input_data.protect,
            global_attrs=global_attrs,
            local_attrs=local_attrs
        )
        original_pieces = cls._get_original_pieces(
            replaced_items=replaced_items,
            string=string
        )
        labels, shape_mobjects = cls._get_labels_and_shape_mobjects(
            unlabelled_content=cls._get_content(
                original_pieces=original_pieces,
                replaced_items=replaced_items,
                is_labelled=False
            ),
            labelled_content=cls._get_content(
                original_pieces=original_pieces,
                replaced_items=replaced_items,
                is_labelled=True
            ),
            requires_labelling=len(label_to_span_dict) > 1,
            input_data=input_data,
            temp_path=temp_path
        )
        spans = [label_to_span_dict[label] for label in labels]

        return StringMobjectOutputData(
            shape_mobjects=shape_mobjects,
            spans=spans,
            labelled_part_items=cls._get_labelled_part_items(
                string=string,
                spans=spans,
                label_to_span_dict=label_to_span_dict
            ),
            group_part_items=cls._get_group_part_items(
                labels=labels,
                label_to_span_dict=label_to_span_dict,
                original_pieces=original_pieces,
                replaced_items=replaced_items
            )
        )

    @classmethod
    def dump_json(
        cls,
        output_data: StringMobjectOutputData
    ) -> StringMobjectJSON:
        return StringMobjectJSON(
            shape_mobjects=[
                SVGMobjectIO._shape_mobject_to_json(shape_mobject)
                for shape_mobject in output_data.shape_mobjects
            ],
            spans=[[span.start, span.stop] for span in output_data.spans],
            labelled_part_strings=[string for string, _ in output_data.labelled_part_items],
            labelled_part_indices=[indices for _, indices in output_data.labelled_part_items],
            group_part_strings=[string for string, _ in output_data.group_part_items],
            group_part_indices=[indices for _, indices in output_data.group_part_items]
        )

    @classmethod
    def load_json(
        cls,
        json_data: StringMobjectJSON
    ) -> StringMobjectOutputData:
        return StringMobjectOutputData(
            shape_mobjects=[
                SVGMobjectIO._json_to_shape_mobject(shape_mobject_json)
                for shape_mobject_json in json_data["shape_mobjects"]
            ],
            spans=[Span(*span_values) for span_values in json_data["spans"]],
            labelled_part_items=list(zip(
                json_data["labelled_part_strings"],
                json_data["labelled_part_indices"],
                strict=True
            )),
            group_part_items=list(zip(
                json_data["group_part_strings"],
                json_data["group_part_indices"],
                strict=True
            ))
        )

    @classmethod
    def _get_labels_and_shape_mobjects(
        cls,
        unlabelled_content: str,
        labelled_content: str,
        requires_labelling: bool,
        input_data: StringMobjectInputData,
        temp_path: pathlib.Path
    ) -> tuple[list[int], list[ShapeMobject]]:
        unlabelled_shapes = cls._get_shape_mobjects(unlabelled_content, input_data, temp_path)
        if not requires_labelling or not unlabelled_shapes:
            return [0] * len(unlabelled_shapes), unlabelled_shapes

        labelled_shapes = cls._get_shape_mobjects(labelled_content, input_data, temp_path)
        assert len(unlabelled_shapes) == len(labelled_shapes)

        ShapeMobject().add(*labelled_shapes).match_bounding_box(
            ShapeMobject().add(*unlabelled_shapes)
        )
        distance_matrix = cdist(
            [shape.get_centroid() for shape in unlabelled_shapes],
            [shape.get_centroid() for shape in labelled_shapes]
        )
        unlabelled_indices, labelled_indices = linear_sum_assignment(distance_matrix)
        return [
            int(ColorUtils.color_to_hex(labelled_shapes[labelled_index]._color_)[1:], 16)
            for labelled_index in labelled_indices
        ], [
            unlabelled_shapes[unlabelled_index]
            for unlabelled_index in unlabelled_indices
        ]

    @classmethod
    @abstractmethod
    def _get_shape_mobjects(
        cls,
        content: str,
        input_data: StringMobjectInputData,
        temp_path: pathlib.Path
    ) -> list[ShapeMobject]:
        svg_path = temp_path.with_suffix(".svg")
        try:
            cls._create_svg(
                content=content,
                input_data=input_data,
                svg_path=svg_path
            )
            shape_mobjects = list(SVGMobjectIO._iter_shape_mobject_from_svg(
                svg_path=svg_path,
                frame_scale=cls._get_svg_frame_scale(input_data)
            ))
        finally:
            svg_path.unlink(missing_ok=True)

        return shape_mobjects

    @classmethod
    @abstractmethod
    def _get_global_attrs(
        cls,
        input_data: StringMobjectInputData,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        pass

    @classmethod
    @abstractmethod
    def _get_local_attrs(
        cls,
        input_data: StringMobjectInputData,
        temp_path: pathlib.Path
    ) -> dict[Span, dict[str, str]]:
        pass

    @classmethod
    @abstractmethod
    def _create_svg(
        cls,
        content: str,
        input_data: StringMobjectInputData,
        svg_path: pathlib.Path
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def _get_svg_frame_scale(
        cls,
        input_data: StringMobjectInputData
    ) -> float:
        # `font_size=30` shall make the height of "x" become roughly 0.30.
        pass

    # parsing

    @classmethod
    def _get_label_to_span_dict_and_replaced_items(
        cls,
        string: str,
        isolate: list[Span],
        protect: list[Span],
        global_attrs: dict[str, str],
        local_attrs: dict[Span, dict[str, str]]
    ) -> tuple[dict[int, Span], list[CommandItem | LabelledInsertionItem]]:

        def get_key(
            index_item: tuple[ConfiguredItem | IsolatedItem | ProtectedItem | CommandItem, EdgeFlag, int, int]
        ) -> tuple[int, int, int, int, int]:
            span_item, edge_flag, priority, item_index = index_item
            flag_value = edge_flag.value
            index = span_item.span.get_edge_index(edge_flag)
            paired_index = span_item.span.get_edge_index(edge_flag.negate())
            return (
                index,
                flag_value * (2 if index != paired_index else -1),
                -paired_index,
                flag_value * priority,
                flag_value * item_index
            )

        def add_labelled_item(
            label_to_span_dict: dict[int, Span],
            replaced_items: list[CommandItem | LabelledInsertionItem],
            label: int,
            span: Span,
            attrs: dict[str, str],
            insert_index: int
        ) -> None:
            label_to_span_dict[label] = span
            replaced_items.insert(insert_index, LabelledInsertionItem(
                label=label,
                edge_flag=EdgeFlag.START,
                attrs=attrs,
                index=span.start
            ))
            replaced_items.append(LabelledInsertionItem(
                label=label,
                edge_flag=EdgeFlag.STOP,
                attrs=attrs,
                index=span.stop
            ))

        index_items = sorted((
            (span_item, edge_flag, priority, item_index)
            for priority, span_item_iterator in enumerate((
                (
                    ConfiguredItem(span=span, attrs=attrs)
                    for span, attrs in local_attrs.items()
                ),
                (
                    IsolatedItem(span=span)
                    for span in isolate
                ),
                (
                    ProtectedItem(span=span)
                    for span in protect
                ),
                (
                    CommandItem(match_obj=match_obj)
                    for match_obj in cls._iter_command_matches(string)
                )
            ), start=1)
            for item_index, span_item in enumerate(span_item_iterator)
            for edge_flag in EdgeFlag
        ), key=get_key)

        label_to_span_dict: dict[int, Span] = {}
        replaced_items: list[CommandItem | LabelledInsertionItem] = []
        label_counter: it.count[int] = it.count(start=1)
        bracket_counter: it.count[int] = it.count()
        protect_level: int = 0
        bracket_stack: list[int] = []
        open_command_stack: list[tuple[int, CommandItem]] = []
        open_stack: list[tuple[int, ConfiguredItem | IsolatedItem, int, list[int]]] = []

        for span_item, edge_flag, _, _ in index_items:
            if isinstance(span_item, ProtectedItem | CommandItem):
                protect_level += edge_flag.value
                if isinstance(span_item, ProtectedItem):
                    continue
                if edge_flag == EdgeFlag.START:
                    continue
                command_item = span_item
                command_flag = cls._get_command_flag(match_obj=command_item.match_obj)
                if command_flag == CommandFlag.OPEN:
                    bracket_stack.append(next(bracket_counter))
                    open_command_stack.append((len(replaced_items) + 1, command_item))
                elif command_flag == CommandFlag.CLOSE:
                    insert_index, open_command_item = open_command_stack.pop()
                    bracket_stack.pop()
                    attrs = cls._get_attrs_from_command_pair(
                        open_command=open_command_item.match_obj,
                        close_command=command_item.match_obj
                    )
                    if attrs is not None:
                        add_labelled_item(
                            label_to_span_dict=label_to_span_dict,
                            replaced_items=replaced_items,
                            label=next(label_counter),
                            span=Span(open_command_item.span.stop, command_item.span.start),
                            attrs=attrs,
                            insert_index=insert_index
                        )
                replaced_items.append(command_item)
                continue
            if edge_flag == EdgeFlag.START:
                open_stack.append((
                    len(replaced_items), span_item, protect_level, bracket_stack.copy()
                ))
                continue
            span = span_item.span
            insert_index, open_span_item, open_protect_level, open_bracket_stack = open_stack.pop()
            assert open_span_item is span_item, \
                "Partly overlapping substrings detected: " + \
                f"'{string[open_span_item.span.as_slice()]}', '{string[span.as_slice()]}'"
            if open_protect_level or protect_level:
                continue
            assert open_bracket_stack == bracket_stack, \
                f"Cannot handle substring: '{string[span.as_slice()]}'"
            add_labelled_item(
                label_to_span_dict=label_to_span_dict,
                replaced_items=replaced_items,
                label=next(label_counter),
                span=span,
                attrs=span_item.attrs if isinstance(span_item, ConfiguredItem) else {},
                insert_index=insert_index
            )
        add_labelled_item(
            label_to_span_dict=label_to_span_dict,
            replaced_items=replaced_items,
            label=0,
            span=Span(0, len(string)),
            attrs=global_attrs,
            insert_index=0
        )

        assert not bracket_stack
        assert not open_command_stack
        assert not open_stack
        return label_to_span_dict, replaced_items

    @classmethod
    def _get_original_pieces(
        cls,
        replaced_items: list[CommandItem | LabelledInsertionItem],
        string: str
    ) -> list[str]:
        replaced_spans = [replaced_item.span for replaced_item in replaced_items]
        return [
            string[start:stop]
            for start, stop in zip(
                [interval_span.stop for interval_span in replaced_spans[:-1]],
                [interval_span.start for interval_span in replaced_spans[1:]],
                strict=True
            )
        ]

    @classmethod
    def _replace_string(
        cls,
        original_pieces: list[str],
        replaced_pieces: list[str],
        start_index: int,
        stop_index: int
    ) -> str:
        return "".join(it.chain.from_iterable(zip(
            ("", *original_pieces[start_index:stop_index - 1]),
            replaced_pieces[start_index:stop_index],
            strict=True
        )))

    @classmethod
    def _get_content(
        cls,
        original_pieces: list[str],
        replaced_items: list[CommandItem | LabelledInsertionItem],
        is_labelled: bool
    ) -> str:
        content_replaced_pieces = [
            cls._replace_for_content(match_obj=replaced_item.match_obj)
            if isinstance(replaced_item, CommandItem)
            else cls._get_command_string(
                label=replaced_item.label if is_labelled else None,
                edge_flag=replaced_item.edge_flag,
                attrs=replaced_item.attrs
            )
            for replaced_item in replaced_items
        ]
        return cls._replace_string(
            original_pieces=original_pieces,
            replaced_pieces=content_replaced_pieces,
            start_index=0,
            stop_index=len(content_replaced_pieces)
        )

    @classmethod
    def _get_indices_by_span(
        cls,
        specified_span: Span,
        spans: list[Span]
    ) -> list[int]:
        return [
            index
            for index, span in enumerate(spans)
            if specified_span.contains(span)
        ]

    @classmethod
    def _get_labelled_part_items(
        cls,
        string: str,
        spans: list[Span],
        label_to_span_dict: dict[int, Span]
    ) -> list[tuple[str, list[int]]]:
        return [
            (string[span.as_slice()], cls._get_indices_by_span(span, spans))
            for span in label_to_span_dict.values()
        ]

    @classmethod
    def _get_group_part_items(
        cls,
        labels: list[int],
        label_to_span_dict: dict[int, Span],
        original_pieces: list[str],
        replaced_items: list[CommandItem | LabelledInsertionItem]
    ) -> list[tuple[str, list[int]]]:

        def iter_boundary_item_pairs(
            label_iterator: Iterator[int],
            label_to_span_dict: dict[int, Span]
        ) -> Iterator[tuple[tuple[int, EdgeFlag], tuple[int, EdgeFlag]]]:
            prev_label = next(label_iterator)
            prev_span = label_to_span_dict[prev_label]
            start_item = (prev_label, EdgeFlag.START)
            for next_label in label_iterator:
                next_span = label_to_span_dict[next_label]
                prev_stop = (prev_label, EdgeFlag.STOP)
                next_start = (next_label, EdgeFlag.START)
                stop_item = next_start if prev_span.contains(next_span) else prev_stop
                yield (start_item, stop_item)
                start_item = prev_stop if next_span.contains(prev_span) else next_start
                prev_label = next_label
                prev_span = next_span
            stop_item = (prev_label, EdgeFlag.STOP)
            yield (start_item, stop_item)

        label_groupers = list(it.groupby(
            enumerate(labels),
            key=lambda label_item: label_item[1]
        ))
        boundary_item_to_index_dict = {
            (replaced_item.label, replaced_item.edge_flag): index
            for index, replaced_item in enumerate(replaced_items)
            if isinstance(replaced_item, LabelledInsertionItem)
        }

        matching_replaced_pieces = [
            cls._replace_for_matching(match_obj=replaced_item.match_obj)
            if isinstance(replaced_item, CommandItem)
            else ""
            for replaced_item in replaced_items
        ]
        return [
            (
                re.sub(r"\s+", "", cls._replace_string(
                    original_pieces=original_pieces,
                    replaced_pieces=matching_replaced_pieces,
                    start_index=start_index,
                    stop_index=stop_index
                )),
                [label_item[0] for label_item in grouper]
            )
            for grouper, (start_item, stop_item) in zip(
                (grouper for _, grouper in label_groupers),
                iter_boundary_item_pairs(
                    label_iterator=(label for label, _ in label_groupers),
                    label_to_span_dict=label_to_span_dict
                ),
                strict=True
            )
            if (
                (start_index := boundary_item_to_index_dict[start_item])
                < (stop_index := boundary_item_to_index_dict[stop_item])
            )
        ]

    @classmethod
    @abstractmethod
    def _iter_command_matches(
        cls,
        string: str
    ) -> Iterator[re.Match[str]]:
        pass

    @classmethod
    @abstractmethod
    def _get_command_flag(
        cls,
        match_obj: re.Match[str]
    ) -> CommandFlag:
        pass

    @classmethod
    @abstractmethod
    def _replace_for_content(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        pass

    @classmethod
    @abstractmethod
    def _replace_for_matching(
        cls,
        match_obj: re.Match[str]
    ) -> str:
        pass

    @classmethod
    @abstractmethod
    def _get_attrs_from_command_pair(
        cls,
        open_command: re.Match[str],
        close_command: re.Match[str]
    ) -> dict[str, str] | None:
        pass

    @classmethod
    @abstractmethod
    def _get_command_string(
        cls,
        label: int | None,
        edge_flag: EdgeFlag,
        attrs: dict[str, str]
    ) -> str:
        pass


class StringMobject(ShapeMobject):
    """
    An abstract base class for `Tex` and `MarkupText`.

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
    so that each child of the original svg will be labelled
    by the color of its paired child from the additional svg.
    """
    __slots__ = (
        "_string",
        "_shape_mobjects",
        "_spans",
        "_labelled_part_items",
        "_group_part_items"
    )

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()

        cls = type(self)
        input_data = cls._input_data_cls(**kwargs)
        output_data = cls._io_cls.get(input_data)

        shape_mobjects = output_data.shape_mobjects
        spans = output_data.spans
        labelled_part_items = output_data.labelled_part_items
        group_part_items = output_data.group_part_items
        self._string: str = input_data.string
        self._shape_mobjects: list[ShapeMobject] = shape_mobjects
        self._spans: list[Span] = spans
        self._labelled_part_items: list[tuple[str, list[int]]] = labelled_part_items
        self._group_part_items: list[tuple[str, list[int]]] = group_part_items
        self.add(*shape_mobjects)

    @classmethod
    @property
    @abstractmethod
    def _io_cls(cls) -> type[StringMobjectIO]:
        pass

    @classmethod
    @property
    @abstractmethod
    def _input_data_cls(cls) -> type[StringMobjectInputData]:
        pass

    @classmethod
    def _iter_spans_by_selector(
        cls,
        selector: SelectorT,
        string: str
    ) -> Iterator[Span]:
        match selector:
            case str():
                substr_len = len(selector)
                if not substr_len:
                    return
                index = 0
                while True:
                    index = string.find(selector, index)
                    if index == -1:
                        break
                    yield Span(index, index + substr_len)
                    index += substr_len
            case re.Pattern():
                for match_obj in selector.finditer(string):
                    yield Span(*match_obj.span())
            case (int() as start, int() as stop):
                assert start <= stop
                yield Span(start, stop)

    @classmethod
    def _get_spans_by_selectors(
        cls,
        selectors: Iterable[SelectorT],
        string: str
    ) -> list[Span]:
        return list(it.chain.from_iterable(
            cls._iter_spans_by_selector(selector, string)
            for selector in selectors
        ))

    def _build_from_indices(
        self,
        indices: list[int]
    ) -> ShapeMobject:
        return ShapeMobject().add(*(
            self._shape_mobjects[index]
            for index in indices
        ))

    def _build_from_indices_list(
        self,
        indices_list: list[list[int]]
    ) -> ShapeMobject:
        return ShapeMobject().add(*(
            self._build_from_indices(indices)
            for indices in indices_list
        ))

    def select_part(
        self,
        selector: SelectorT,
        index: int = 0
    ) -> ShapeMobject:
        cls = type(self)
        return self._build_from_indices([
            StringMobjectIO._get_indices_by_span(specified_span, self._spans)
            for specified_span in cls._iter_spans_by_selector(selector, self._string)
        ][index])

    def select_parts(
        self,
        selector: SelectorT
    ) -> ShapeMobject:
        cls = type(self)
        return self._build_from_indices_list([
            StringMobjectIO._get_indices_by_span(specified_span, self._spans)
            for specified_span in cls._iter_spans_by_selector(selector, self._string)
        ])
