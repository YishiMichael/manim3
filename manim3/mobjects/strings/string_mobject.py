from abc import (
    ABC,
    abstractmethod
)
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import itertools as it
import pathlib
import re
from typing import (
    Any,
    ClassVar,
    Iterable,
    Iterator
)

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ...config import Config
from ...custom_typing import (
    SelectorT,
    NP_3f8
)
from ...utils.color import ColorUtils
from ...utils.iterables import IterUtils
from ..shape_mobject import ShapeMobject
from ..svg_mobject import SVGMobject


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
    __slots__ = (
        "start",
        "stop"
    )

    def __init__(
        self,
        start: int,
        stop: int
    ) -> None:
        assert start <= stop, f"Invalid span: ({start}, {stop})"
        self.start: int = start
        self.stop: int = stop

    def __contains__(
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
class LabelledShapeMobject:
    label: int
    shape_mobject: ShapeMobject


class StringFileWriter(ABC):
    __slots__ = ("_parameters",)

    _dir_name: ClassVar[str]

    def __init__(
        self,
        **parameters: Any
    ) -> None:
        super().__init__()
        self._parameters: dict[str, Any] = parameters

    def get_svg_file(
        self,
        content: str
    ) -> pathlib.Path:
        parameters = self._parameters
        cls = type(self)
        hash_content = str((content, *parameters.values()))
        svg_path = cls.get_svg_path(hash_content)
        if not svg_path.exists():
            with cls.display_during_execution(content):
                cls.create_svg_file(content, svg_path, **parameters)
        return svg_path

    @classmethod
    @abstractmethod
    def create_svg_file(
        cls,
        content: str,
        svg_path: pathlib.Path,
        **parameters: Any
    ) -> None:
        pass

    @classmethod
    def get_svg_path(
        cls,
        hash_content: str
    ) -> pathlib.Path:
        # Truncating at 16 bytes for cleanliness.
        hex_string = hashlib.sha256(hash_content.encode()).hexdigest()[:16]
        svg_dir = Config().path.get_output_subdir(cls._dir_name)
        return svg_dir.joinpath(f"{hex_string}.svg")

    @classmethod
    @contextmanager
    def display_during_execution(
        cls,
        string: str
    ) -> Iterator[None]:
        max_characters = 60
        summary = string.replace("\n", "")
        if len(summary) > max_characters:
            summary = f"{summary[:max_characters - 3]}..."
        message = f"Writing \"{summary}\""
        try:
            print(message, end="\r")
            yield
        finally:
            print(" " * len(message), end="\r")


class StringParser(ABC):
    __slots__ = (
        "_string",
        "_label_to_span_dict",
        "_replaced_items",
        "_original_pieces",
        "_labelled_shape_mobjects"
    )

    def __init__(
        self,
        string: str,
        isolate: Iterable[SelectorT],
        protect: Iterable[SelectorT],
        local_attrs: dict[SelectorT, dict[str, str]],
        global_attrs: dict[str, str],
        file_writer: StringFileWriter,
        frame_scale: float
    ) -> None:
        super().__init__()
        cls = type(self)
        label_to_span_dict, replaced_items = cls._get_label_to_span_dict_and_replaced_items(
            string=string,
            isolate=isolate,
            protect=protect,
            local_attrs=local_attrs,
            global_attrs=global_attrs
        )
        original_pieces = cls._get_original_pieces(
            replaced_items=replaced_items,
            string=string
        )
        labelled_shape_mobjects = list(cls._iter_labelled_shape_mobjects(
            original_pieces=original_pieces,
            replaced_items=replaced_items,
            labels_count=len(label_to_span_dict),
            file_writer=file_writer,
            frame_scale=frame_scale
        ))

        self._string: str = string
        self._label_to_span_dict: dict[int, Span] = label_to_span_dict
        self._replaced_items: list[CommandItem | LabelledInsertionItem] = replaced_items
        self._original_pieces: list[str] = original_pieces
        self._labelled_shape_mobjects: list[LabelledShapeMobject] = labelled_shape_mobjects

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
                yield Span(start, stop)

    @classmethod
    def _get_label_to_span_dict_and_replaced_items(
        cls,
        string: str,
        isolate: Iterable[SelectorT],
        protect: Iterable[SelectorT],
        local_attrs: dict[SelectorT, dict[str, str]],
        global_attrs: dict[str, str]
    ) -> tuple[dict[int, Span], list[CommandItem | LabelledInsertionItem]]:

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

        index_items = sorted((
            (span_item, edge_flag, priority, i)
            for priority, span_item_iterator in enumerate((
                (
                    ConfiguredItem(span=span, attrs=attrs)
                    for selector, attrs in local_attrs.items()
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                (
                    IsolatedItem(span=span)
                    for selector in isolate
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                (
                    ProtectedItem(span=span)
                    for selector in protect
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                (
                    CommandItem(match_obj=match_obj)
                    for match_obj in cls._iter_command_matches(string)
                )
            ), start=1)
            for i, span_item in enumerate(span_item_iterator)
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

        def add_labelled_item(
            label: int,
            span: Span,
            attrs: dict[str, str],
            pos: int
        ) -> None:
            label_to_span_dict[label] = span
            replaced_items.insert(pos, LabelledInsertionItem(
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

        for span_item, edge_flag, _, _ in index_items:
            if isinstance(span_item, ProtectedItem | CommandItem):
                protect_level += edge_flag.get_value()
                if isinstance(span_item, ProtectedItem):
                    continue
                if edge_flag == EdgeFlag.START:
                    continue
                command_item = span_item
                command_flag = cls._get_command_flag(match_obj=command_item.match_obj)
                if command_flag == CommandFlag.OPEN:
                    bracket_stack.append(next(bracket_counter))
                    open_command_stack.append((len(replaced_items), command_item))
                elif command_flag == CommandFlag.CLOSE:
                    pos, open_command_item = open_command_stack.pop()
                    bracket_stack.pop()
                    attrs = cls._get_attrs_from_command_pair(
                        open_command=open_command_item.match_obj,
                        close_command=command_item.match_obj
                    )
                    if attrs is not None:
                        add_labelled_item(
                            label=next(label_counter),
                            span=Span(open_command_item.span.stop, command_item.span.start),
                            attrs=attrs,
                            pos=pos + 1
                        )
                replaced_items.append(command_item)
                continue
            if edge_flag == EdgeFlag.START:
                open_stack.append((
                    len(replaced_items), span_item, protect_level, bracket_stack.copy()
                ))
                continue
            span = span_item.span
            pos, open_span_item, open_protect_level, open_bracket_stack = open_stack.pop()
            assert open_span_item is span_item, \
                "Partly overlapping substrings detected: " + \
                f"'{string[open_span_item.span.as_slice()]}', '{string[span.as_slice()]}'"
            if open_protect_level or protect_level:
                continue
            assert open_bracket_stack == bracket_stack, \
                f"Cannot handle substring: '{string[span.as_slice()]}'"
            add_labelled_item(
                label=next(label_counter),
                span=span,
                attrs=span_item.attrs if isinstance(span_item, ConfiguredItem) else {},
                pos=pos
            )
        add_labelled_item(
            label=0,
            span=Span(0, len(string)),
            attrs=global_attrs,
            pos=0
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
    def _iter_matched_shape_mobjects(
        cls,
        plain_shapes: list[ShapeMobject],
        labelled_shapes: list[ShapeMobject]
    ) -> Iterator[tuple[ShapeMobject, ShapeMobject]]:
        # Rearrange `labelled_shapes` so that
        # each mobject is labelled by the nearest one of `labelled_shapes`.
        # The correctness cannot be ensured, since the svg may
        # change significantly after inserting color commands.

        def get_matched_position_indices(
            positions_0: list[NP_3f8],
            positions_1: list[NP_3f8]
        ) -> tuple[list[int], list[int]]:
            distance_matrix = cdist(positions_0, positions_1)
            return linear_sum_assignment(distance_matrix)

        assert len(plain_shapes) == len(labelled_shapes)
        if not labelled_shapes:
            return

        SVGMobject().add(*labelled_shapes).match_bounding_box(
            SVGMobject().add(*plain_shapes)
        )

        for plain_index, labelled_index in zip(*get_matched_position_indices(
            [shape.get_center() for shape in plain_shapes],
            [shape.get_center() for shape in labelled_shapes]
        ), strict=True):
            yield plain_shapes[plain_index], labelled_shapes[labelled_index]

    @classmethod
    def _iter_labelled_shape_mobjects(
        cls,
        original_pieces: list[str],
        replaced_items: list[CommandItem | LabelledInsertionItem],
        labels_count: int,
        file_writer: StringFileWriter,
        frame_scale: float
    ) -> Iterator[LabelledShapeMobject]:

        def iter_shape_mobjects(
            is_labelled: bool
        ) -> Iterator[ShapeMobject]:
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
            content = cls._replace_string(
                original_pieces=original_pieces,
                replaced_pieces=content_replaced_pieces,
                start_index=0,
                stop_index=len(content_replaced_pieces)
            )
            svg_path = file_writer.get_svg_file(content)
            svg_mobject = SVGMobject(
                file_path=svg_path,
                frame_scale=frame_scale
            )
            for mobject in svg_mobject:
                if isinstance(mobject, ShapeMobject):
                    yield mobject
            #return .iter_children_by_type(mobject_type=ShapeMobject)

        plain_shapes_iterator = iter_shape_mobjects(is_labelled=False)
        if labels_count == 1:
            for plain_shape in plain_shapes_iterator:
                yield LabelledShapeMobject(
                    label=0,
                    shape_mobject=plain_shape
                )
            return

        labelled_shapes_iterator = iter_shape_mobjects(is_labelled=True)
        for plain_shape, labelled_shape in cls._iter_matched_shape_mobjects(
            list(plain_shapes_iterator), list(labelled_shapes_iterator)
        ):
            label = int(ColorUtils.color_to_hex(labelled_shape._color_)[1:], 16)
            yield LabelledShapeMobject(
                label=label,
                shape_mobject=plain_shape
            )

    def iter_shape_mobjects(self) -> Iterator[ShapeMobject]:
        for labelled_shape_mobject in self._labelled_shape_mobjects:
            yield labelled_shape_mobject.shape_mobject

    def iter_shape_mobjects_by_span(
        self,
        span: Span
    ) -> Iterator[ShapeMobject]:
        for labelled_shape_mobject in self._labelled_shape_mobjects:
            if self._label_to_span_dict[labelled_shape_mobject.label] in span:
                yield labelled_shape_mobject.shape_mobject

    def iter_iter_shape_mobjects_by_selector(
        self,
        selector: SelectorT
    ) -> Iterator[Iterator[ShapeMobject]]:
        for span in type(self)._iter_spans_by_selector(selector, self._string):
            yield self.iter_shape_mobjects_by_span(span)

    def iter_specified_part_items(self) -> Iterator[tuple[str, Iterator[ShapeMobject]]]:
        for labelled_shape_mobject in self._labelled_shape_mobjects:
            span = self._label_to_span_dict[labelled_shape_mobject.label]
            yield self._string[span.as_slice()], self.iter_shape_mobjects_by_span(span)

    def iter_group_part_items(self) -> Iterator[tuple[str, Iterator[ShapeMobject]]]:

        def iter_boundary_items(
            label_iterator: Iterator[int],
            label_to_span_dict: dict[int, Span]
        ) -> Iterator[tuple[int, EdgeFlag]]:
            prev_label = next(label_iterator)
            prev_span = label_to_span_dict[prev_label]
            yield (prev_label, EdgeFlag.START)
            for next_label in label_iterator:
                next_span = label_to_span_dict[next_label]
                prev_stop = (prev_label, EdgeFlag.STOP)
                next_start = (next_label, EdgeFlag.START)
                yield next_start if next_span in prev_span else prev_stop
                yield prev_stop if prev_span in next_span else next_start
                prev_label = next_label
                prev_span = next_span
            yield (prev_label, EdgeFlag.STOP)

        if not self._labelled_shape_mobjects:
            return

        label_iterator, grouper_iterator = IterUtils.unzip_pairs(it.groupby(
            self._labelled_shape_mobjects,
            key=lambda labelled_shape_item: labelled_shape_item.label
        ))
        boundary_item_to_index_dict = {
            (replaced_item.label, replaced_item.edge_flag): index
            for index, replaced_item in enumerate(self._replaced_items)
            if isinstance(replaced_item, LabelledInsertionItem)
        }
        index_iterator = (
            boundary_item_to_index_dict[boundary_item]
            for boundary_item in iter_boundary_items(
                label_iterator=label_iterator,
                label_to_span_dict=self._label_to_span_dict
            )
        )

        cls = type(self)
        matching_replaced_pieces = [
            cls._replace_for_matching(match_obj=replaced_item.match_obj)
            if isinstance(replaced_item, CommandItem)
            else ""
            for replaced_item in self._replaced_items
        ]
        for grouper in grouper_iterator:
            start_index = next(index_iterator)
            stop_index = next(index_iterator)
            if start_index >= stop_index:
                continue
            yield re.sub(r"\s+", "", cls._replace_string(
                original_pieces=self._original_pieces,
                replaced_pieces=matching_replaced_pieces,
                start_index=start_index,
                stop_index=stop_index
            )), (labelled_shape_mobject.shape_mobject for labelled_shape_mobject in grouper)

    # Implemented in subclasses.

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


class StringMobject(SVGMobject):
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
    so that each child of the original `SVGMobject` will be labelled
    by the color of its paired child from the additional `SVGMobject`.
    """
    __slots__ = ("_parser",)

    def __init__(
        self,
        *,
        parser: StringParser
    ) -> None:
        super().__init__()
        self._parser: StringParser = parser
        self.add(*parser.iter_shape_mobjects())

    def select_parts(
        self,
        selector: SelectorT
    ) -> ShapeMobject:
        return ShapeMobject().add(*(
            ShapeMobject().add(*shape_mobject_iterator)
            for shape_mobject_iterator in self._parser.iter_iter_shape_mobjects_by_selector(selector)
        ))

    def select_part(
        self,
        selector: SelectorT,
        index: int = 0
    ) -> ShapeMobject:
        return ShapeMobject().add(*(
            list(self._parser.iter_iter_shape_mobjects_by_selector(selector))[index]
        ))
