from __future__ import annotations


import itertools
import pathlib
import re
from abc import abstractmethod
from enum import Enum
from typing import (
    Iterable,
    Iterator,
    Self,
    TypedDict
)

import attrs

import scipy.optimize
import scipy.spatial.distance

from ...animatables.shape import Shape
from ...constants.custom_typing import (
    ColorType,
    SelectorType
)
from ...utils.color_utils import ColorUtils
from ..shape_mobjects.shape_mobject import ShapeMobject
from ..mobject import Mobject
from ..mobject_io import (
    MobjectIO,
    MobjectInput,
    MobjectJSON,
    MobjectOutput
)
from ..svg_mobject import (
    SVGMobjectIO,
    ShapeMobjectJSON
)


class BoundaryFlag(Enum):
    START = 1
    STOP = -1

    def negate(
        self: Self
    ) -> BoundaryFlag:
        return BoundaryFlag(-self.value)


class Span:
    __slots__ = (
        "_start",
        "_stop"
    )

    def __init__(
        self: Self,
        start: int,
        stop: int
    ) -> None:
        assert 0 <= start <= stop
        super().__init__()
        self._start: int = start
        self._stop: int = stop

    def contains(
        self: Self,
        span: Span
    ) -> bool:
        return self._start <= span._start and self._stop >= span._stop

    def as_slice(
        self: Self
    ) -> slice:
        return slice(self._start, self._stop)

    def get_boundary_index(
        self: Self,
        boundary_flag: BoundaryFlag
    ) -> int:
        return self._start if boundary_flag == BoundaryFlag.START else self._stop


class CommandFlag(Enum):
    STANDALONE = 0
    OPEN = 1
    CLOSE = -1


class CommandInfo:
    __slots__ = (
        "_command_items",
        "_attributes_item"
    )

    def __init__(
        self: Self,
        match_obj_items: tuple[tuple[re.Match[str], str | None, CommandFlag], ...],
        attributes_item: tuple[dict[str, str], bool] | None
    ) -> None:
        super().__init__()
        self._command_items: tuple[tuple[Span, str, CommandFlag], ...] = tuple(
            (
                Span(*match.span()),
                replacement if replacement is not None else match.group(),
                command_flag
            )
            for match, replacement, command_flag in match_obj_items
        )
        self._attributes_item: tuple[dict[str, str], bool] | None = attributes_item


class StandaloneCommandInfo(CommandInfo):
    __slots__ = ()

    def __init__(
        self: Self,
        match: re.Match[str],
        replacement: str | None = None
    ) -> None:
        super().__init__(
            match_obj_items=(
                (match, replacement, CommandFlag.STANDALONE),
            ),
            attributes_item=None
        )


class BalancedCommandInfo(CommandInfo):
    __slots__ = (
        "_other_match_obj_item",
        "_attributes_item"
    )

    def __init__(
        self: Self,
        attributes: dict[str, str],
        isolated: bool,
        open_match_obj: re.Match[str],
        close_match_obj: re.Match[str],
        open_replacement: str | None = None,
        close_replacement: str | None = None
    ) -> None:
        super().__init__(
            match_obj_items=(
                (open_match_obj, open_replacement, CommandFlag.OPEN),
                (close_match_obj, close_replacement, CommandFlag.CLOSE)
            ),
            attributes_item=(attributes, isolated)
        )


@attrs.frozen(kw_only=True)
class SpanInfo:
    span: Span
    isolated: bool | None = None
    attributes: dict[str, str] | None = None
    local_color: ColorType | None = None
    command_item: tuple[CommandInfo, str, CommandFlag] | None = None


class ReplacementRecord:
    __slots__ = (
        "_span",
        "_unlabelled_replacement",
        "_labelled_replacement"
    )

    def __init__(
        self: Self,
        span: Span
    ) -> None:
        super().__init__()
        self._span: Span = span
        self._unlabelled_replacement: str = ""
        self._labelled_replacement: str = ""

    def write_replacements(
        self: Self,
        *,
        unlabelled_replacement: str,
        labelled_replacement: str
    ) -> None:
        self._unlabelled_replacement = unlabelled_replacement
        self._labelled_replacement = labelled_replacement


class InsertionRecord(ReplacementRecord):
    __slots__ = ()

    def __init__(
        self: Self,
        index: int
    ) -> None:
        super().__init__(Span(index, index))


class StringMobjectKwargs(TypedDict, total=False):
    local_colors: dict[SelectorType, ColorType]
    isolate: Iterable[SelectorType]
    protect: Iterable[SelectorType]
    concatenate: bool


@attrs.frozen(kw_only=True)
class StringMobjectInput(MobjectInput):
    string: str
    local_colors: dict[SelectorType, ColorType] = attrs.field(factory=dict)
    isolate: Iterable[SelectorType] = ()
    protect: Iterable[SelectorType] = ()
    concatenate: bool = False


@attrs.frozen(kw_only=True)
class StringMobjectOutput(MobjectOutput):
    shape_mobjects: tuple[ShapeMobject, ...]
    string: str
    labels: tuple[int, ...]
    isolated_spans: tuple[Span, ...]


class StringMobjectJSON(MobjectJSON):
    shape_mobjects: tuple[ShapeMobjectJSON, ...]
    string: str
    labels: tuple[int, ...]
    isolated_spans: tuple[tuple[int, ...], ...]


class StringMobjectIO[StringMobjectInputT: StringMobjectInput](
    MobjectIO[StringMobjectInputT, StringMobjectOutput, StringMobjectJSON]
):
    __slots__ = ()

    @classmethod
    def generate(
        cls: type[Self],
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> StringMobjectOutput:

        def replace_string(
            original_pieces: tuple[str, ...],
            replacement_pieces: tuple[str, ...]
        ) -> str:
            return "".join(itertools.chain.from_iterable(zip(
                ("", *original_pieces),
                replacement_pieces,
                strict=True
            )))

        string = input_data.string
        isolated_items, replacement_records = cls._get_isolated_items_and_replacement_records(
            string=string,
            span_infos=(
                SpanInfo(
                    span=Span(0, len(string)),
                    isolated=True,
                    attributes=cls._get_global_span_attributes(input_data, temp_path)
                ),
                *(
                    SpanInfo(
                        span=span,
                        isolated=False,
                        attributes=attributes
                    )
                    for span, attributes in cls._iter_local_span_attributes(input_data, temp_path)
                ),
                *(
                    SpanInfo(
                        span=span,
                        isolated=True,
                        local_color=local_color
                    )
                    for selector, local_color in input_data.local_colors.items()
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                *(
                    SpanInfo(
                        span=span,
                        isolated=True
                    )
                    for selector in input_data.isolate
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                *(
                    SpanInfo(
                        span=span
                    )
                    for selector in input_data.protect
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                *(
                    SpanInfo(
                        span=span,
                        command_item=(command_info, replacement, command_flag)
                    )
                    for command_info in cls._iter_command_infos(string)
                    for span, replacement, command_flag in command_info._command_items
                )
            )
        )
        original_pieces = tuple(
            string[start:stop]
            for start, stop in zip(
                (replacement_record._span._stop for replacement_record in replacement_records[:-1]),
                (replacement_record._span._start for replacement_record in replacement_records[1:]),
                strict=True
            )
        )
        shape_mobject_items = tuple(cls._iter_shape_mobject_items(
            unlabelled_content=replace_string(
                original_pieces=original_pieces,
                replacement_pieces=tuple(
                    replacement_record._unlabelled_replacement
                    for replacement_record in replacement_records
                )
            ),
            labelled_content=replace_string(
                original_pieces=original_pieces,
                replacement_pieces=tuple(
                    replacement_record._labelled_replacement
                    for replacement_record in replacement_records
                )
            ),
            requires_labelling=len(isolated_items) > 1,
            input_data=input_data,
            temp_path=temp_path
        ))

        for shape_mobject, label in shape_mobject_items:
            _, local_color = isolated_items[label]
            if local_color is not None:
                shape_mobject._color_._array_ = ColorUtils.color_to_array(local_color)

        return StringMobjectOutput(
            shape_mobjects=tuple(shape_mobject for shape_mobject, _ in shape_mobject_items),
            string=string,
            labels=tuple(label for _, label in shape_mobject_items),
            isolated_spans=tuple(isolated_span for isolated_span, _ in isolated_items)
        )

    @classmethod
    def dump_json(
        cls: type[Self],
        output_data: StringMobjectOutput
    ) -> StringMobjectJSON:
        return StringMobjectJSON(
            shape_mobjects=tuple(
                SVGMobjectIO._shape_mobject_to_json(shape_mobject)
                for shape_mobject in output_data.shape_mobjects
            ),
            string=output_data.string,
            labels=output_data.labels,
            isolated_spans=tuple((span._start, span._stop) for span in output_data.isolated_spans)
        )

    @classmethod
    def load_json(
        cls: type[Self],
        json_data: StringMobjectJSON
    ) -> StringMobjectOutput:
        return StringMobjectOutput(
            shape_mobjects=tuple(
                SVGMobjectIO._json_to_shape_mobject(shape_mobject_json)
                for shape_mobject_json in json_data["shape_mobjects"]
            ),
            string=json_data["string"],
            labels=json_data["labels"],
            isolated_spans=tuple(Span(*span_values) for span_values in json_data["isolated_spans"])
        )

    @classmethod
    def _iter_shape_mobject_items(
        cls: type[Self],
        unlabelled_content: str,
        labelled_content: str,
        requires_labelling: bool,
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> Iterator[tuple[ShapeMobject, int]]:
        unlabelled_shape_mobjects = cls._get_shape_mobjects(unlabelled_content, input_data, temp_path)
        if input_data.concatenate:
            yield ShapeMobject(Shape().concatenate(tuple(
                shape_mobject._shape_ for shape_mobject in unlabelled_shape_mobjects
            ))), 0
            return

        if not requires_labelling or not unlabelled_shape_mobjects:
            for unlabelled_shape_mobject in unlabelled_shape_mobjects:
                yield unlabelled_shape_mobject, 0
            return

        labelled_shape_mobjects = cls._get_shape_mobjects(labelled_content, input_data, temp_path)
        assert len(unlabelled_shape_mobjects) == len(labelled_shape_mobjects)

        unlabelled_radii = Mobject().add(*unlabelled_shape_mobjects).box.get_radii()
        labelled_radii = Mobject().add(*labelled_shape_mobjects).box.get_radii()
        scale_factor = unlabelled_radii / labelled_radii
        distance_matrix = scipy.spatial.distance.cdist(
            [shape.box.get() for shape in unlabelled_shape_mobjects],
            [shape.box.get() * scale_factor for shape in labelled_shape_mobjects]
        )
        for unlabelled_index, labelled_index in zip(*scipy.optimize.linear_sum_assignment(distance_matrix), strict=True):
            yield (
                unlabelled_shape_mobjects[unlabelled_index],
                int(ColorUtils.color_to_hex(labelled_shape_mobjects[labelled_index]._color_._array_)[1:], 16)
            )

    @classmethod
    def _get_shape_mobjects(
        cls: type[Self],
        content: str,
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> tuple[ShapeMobject, ...]:
        svg_path = temp_path.with_suffix(".svg")
        try:
            cls._create_svg(
                content=content,
                input_data=input_data,
                svg_path=svg_path
            )
            shape_mobjects = SVGMobjectIO._get_shape_mobjects_from_svg_path(
                svg_path=svg_path,
                frame_scale=cls._get_svg_frame_scale(input_data)
            )
        finally:
            svg_path.unlink(missing_ok=True)

        return shape_mobjects

    @classmethod
    def _get_global_span_attributes(
        cls: type[Self],
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        return {}

    @classmethod
    def _iter_local_span_attributes(
        cls: type[Self],
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> Iterator[tuple[Span, dict[str, str]]]:
        yield from ()

    @classmethod
    @abstractmethod
    def _create_svg(
        cls: type[Self],
        content: str,
        input_data: StringMobjectInputT,
        svg_path: pathlib.Path
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def _get_svg_frame_scale(
        cls: type[Self],
        input_data: StringMobjectInputT
    ) -> float:
        # `font_size=30` shall make the height of "x" become roughly 0.30.
        pass

    # parsing

    @classmethod
    def _get_isolated_items_and_replacement_records(
        cls: type[Self],
        string: str,
        span_infos: tuple[SpanInfo, ...]
    ) -> tuple[tuple[tuple[Span, ColorType | None], ...], tuple[ReplacementRecord, ...]]:

        def get_sorting_key(
            span_boundary: tuple[SpanInfo, BoundaryFlag]
        ) -> tuple[int, int, int]:
            span_info, boundary_flag = span_boundary
            span = span_info.span
            index = span.get_boundary_index(boundary_flag)
            paired_index = span.get_boundary_index(boundary_flag.negate())
            return (
                index,
                (boundary_flag.value) * (-1 if index == paired_index else 2),
                -paired_index
            )

        insertion_record_items: list[tuple[InsertionRecord, InsertionRecord, dict[str, str], bool, ColorType | None]] = []
        replacement_records: list[ReplacementRecord] = []
        bracket_counter = itertools.count()
        protect_level = 0
        bracket_stack: list[int] = []
        open_command_stack: list[tuple[InsertionRecord, CommandInfo]] = []
        start_stack: list[tuple[SpanInfo, InsertionRecord, int, tuple[int, ...]]] = []
        local_color_stack: list[ColorType] = []

        for span_info, boundary_flag in sorted((
            (span_info, boundary_flag)
            for boundary_flag in (BoundaryFlag.STOP, BoundaryFlag.START)
            for span_info in span_infos[::boundary_flag.value]
        ), key=get_sorting_key):
            span = span_info.span
            if span_info.isolated is None:
                protect_level += boundary_flag.value
                if span_info.command_item is None:
                    continue
                if boundary_flag == BoundaryFlag.START:
                    continue
                command_info, command_replacement, command_flag = span_info.command_item
                command_replacement_record = ReplacementRecord(span)
                command_replacement_record.write_replacements(
                    unlabelled_replacement=command_replacement,
                    labelled_replacement=command_replacement
                )
                if command_flag == CommandFlag.OPEN:
                    bracket_stack.append(next(bracket_counter))
                    open_insertion_record = InsertionRecord(span._stop)
                    replacement_records.append(command_replacement_record)
                    replacement_records.append(open_insertion_record)
                    open_command_stack.append((open_insertion_record, command_info))
                elif command_flag == CommandFlag.CLOSE:
                    bracket_stack.pop()
                    close_insertion_record = InsertionRecord(span._start)
                    replacement_records.append(close_insertion_record)
                    replacement_records.append(command_replacement_record)
                    open_insertion_record, open_command_info = open_command_stack.pop()
                    assert open_command_info is command_info
                    assert (attributes_item := command_info._attributes_item) is not None
                    attributes, isolated = attributes_item
                    insertion_record_items.append((
                        open_insertion_record,
                        close_insertion_record,
                        attributes,
                        isolated,
                        None
                    ))
                elif command_flag == CommandFlag.STANDALONE:
                    replacement_records.append(command_replacement_record)
                continue

            if boundary_flag == BoundaryFlag.START:
                start_insertion_record = InsertionRecord(span._start)
                replacement_records.append(start_insertion_record)
                start_stack.append((span_info, start_insertion_record, protect_level, tuple(bracket_stack)))
                if span_info.local_color is not None:
                    local_color_stack.append(span_info.local_color)
            elif boundary_flag == BoundaryFlag.STOP:
                stop_insertion_record = InsertionRecord(span._stop)
                replacement_records.append(stop_insertion_record)
                start_span_info, start_insertion_record, start_protect_level, start_bracket_stack = start_stack.pop()
                if start_protect_level or protect_level:
                    continue
                if start_span_info is not span_info:
                    raise ValueError(
                        f"Partly overlapping substrings detected: '{string[start_span_info.span.as_slice()]}', '{string[span.as_slice()]}'"
                    )
                if start_bracket_stack != tuple(bracket_stack):
                    raise ValueError(
                        f"Cannot handle substring: '{string[span.as_slice()]}'"
                    )
                if span_info.local_color is not None:
                    local_color = span_info.local_color
                    local_color_stack.pop()
                else:
                    local_color = local_color_stack[-1] if local_color_stack else None
                insertion_record_items.append((
                    start_insertion_record,
                    stop_insertion_record,
                    span_info.attributes if span_info.attributes is not None else {},
                    span_info.isolated,
                    local_color
                ))

        assert protect_level == 0
        assert not bracket_stack
        assert not open_command_stack
        assert not start_stack
        assert not local_color_stack

        label_counter = itertools.count()
        isolated_items: list[tuple[Span, ColorType | None]] = []
        for start_insertion_record, stop_insertion_record, attributes, isolated, local_color in insertion_record_items:
            if isolated:
                label = next(label_counter)
                isolated_items.append((
                    Span(start_insertion_record._span._stop, stop_insertion_record._span._start),
                    local_color
                ))
            else:
                label = None
            labelled_attributes = cls._convert_attributes_for_labelling(attributes, label)
            start_unlabelled_insertion, stop_unlabelled_insertion = cls._get_command_pair(attributes)
            start_labelled_insertion, stop_labelled_insertion = cls._get_command_pair(labelled_attributes)
            start_insertion_record.write_replacements(
                unlabelled_replacement=start_unlabelled_insertion,
                labelled_replacement=start_labelled_insertion
            )
            stop_insertion_record.write_replacements(
                unlabelled_replacement=stop_unlabelled_insertion,
                labelled_replacement=stop_labelled_insertion
            )

        return tuple(isolated_items), tuple(replacement_records)

    @classmethod
    def _iter_spans_by_selector(
        cls: type[Self],
        selector: SelectorType,
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
                for match in selector.finditer(string):
                    yield Span(*match.span())
            case slice(start=int(start), stop=int(stop)):
                l = len(string)
                start = min(start, l) if start >= 0 else max(start + l, 0)
                stop = min(stop, l) if stop >= 0 else max(stop + l, 0)
                if start <= stop:
                    yield Span(start, stop)

    @classmethod
    @abstractmethod
    def _get_command_pair(
        cls: type[Self],
        attributes: dict[str, str]
    ) -> tuple[str, str]:
        pass

    @classmethod
    @abstractmethod
    def _convert_attributes_for_labelling(
        cls: type[Self],
        attributes: dict[str, str],
        label: int | None
    ) -> dict[str, str]:
        pass

    @classmethod
    @abstractmethod
    def _iter_command_infos(
        cls: type[Self],
        string: str
    ) -> Iterator[CommandInfo]:
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
        "_shape_mobjects",
        "_string",
        "_labels",
        "_isolated_spans"
    )

    def __init__(
        self: Self,
        output_data: StringMobjectOutput
    ) -> None:
        super().__init__()

        shape_mobjects = output_data.shape_mobjects
        self._shape_mobjects: tuple[ShapeMobject, ...] = shape_mobjects
        self._string: str = output_data.string
        self._labels: tuple[int, ...] = output_data.labels
        self._isolated_spans: tuple[Span, ...] = output_data.isolated_spans
        self.add(*shape_mobjects)

    def _get_indices_tuple_by_selector(
        self: Self,
        selector: SelectorType
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(
                index
                for index, label in enumerate(self._labels)
                if specified_span.contains(self._isolated_spans[label])
            )
            for specified_span in StringMobjectIO._iter_spans_by_selector(selector, self._string)
        )

    def _build_from_indices(
        self: Self,
        indices: tuple[int, ...]
    ) -> Mobject:
        return Mobject().add(*(
            self._shape_mobjects[index]
            for index in indices
        ))

    def _build_from_indices_tuple(
        self: Self,
        indices_tuple: tuple[tuple[int, ...], ...]
    ) -> Mobject:
        return Mobject().add(*(
            self._build_from_indices(indices)
            for indices in indices_tuple
        ))

    def select_part(
        self: Self,
        selector: SelectorType,
        index: int = 0
    ) -> Mobject:
        return self._build_from_indices(self._get_indices_tuple_by_selector(selector)[index])

    def select_parts(
        self: Self,
        selector: SelectorType
    ) -> Mobject:
        return self._build_from_indices_tuple(self._get_indices_tuple_by_selector(selector))
