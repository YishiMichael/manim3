from __future__ import annotations


import pathlib
import subprocess
from abc import abstractmethod
from typing import (
    Self,
    TypedDict
)

import attrs

from ...animatables.arrays.animatable_color import AnimatableColor
from ...animatables.model import SetKwargs
from ...animatables.shape import Shape
from ...constants.custom_typing import (
    ColorType,
    SelectorType
)
from ...toplevel.toplevel import Toplevel
from ..shape_mobjects.shape_mobject import ShapeMobject
from ..cached_mobject import (
    CachedMobject,
    CachedMobjectInputs
)
from ..svg_mobject import SVGMobject

#from ..mobject_io import (
#    MobjectIO,
#    MobjectInput,
#    MobjectJSON,
#    MobjectOutput
#)
#from ..svg_mobject import (
#    SVGMobjectIO,
#    ShapeMobjectJSON
#)


class TypstMobjectKwargs(TypedDict, total=False):
    #global_attributes: AttributesT
    #local_attributes: dict[SelectorType, AttributesT]
    #local_colors: dict[SelectorType, ColorType]
    #isolate: Iterable[SelectorType]
    #protect: Iterable[SelectorType]
    preamble: str
    concatenate: bool
    align: str | None
    font: str | tuple[str, ...] | None
    color: ColorType | None


#@attrs.frozen(kw_only=True)
#class TypstMobjectParameters:
#    concatenate: bool = False


@attrs.frozen(kw_only=True)
class TypstMobjectInputs(CachedMobjectInputs):

    @staticmethod
    def _docstring_trim(
        string: str
    ) -> str:
        # Borrowed from `https://peps.python.org/pep-0257/`.
        if not string:
            return ""
        lines = string.splitlines()
        indents = tuple(
            len(line) - len(stripped)
            for line in lines[1:]
            if (stripped := line.lstrip())
        )
        trimmed = [lines[0].strip()]
        if indents:
            indent = min(indents)
            trimmed.extend(
                line[indent:].rstrip()
                for line in lines[1:]
            )
        while trimmed and not trimmed[-1]:
            trimmed.pop()
        while trimmed and not trimmed[0]:
            trimmed.pop(0)
        return "\n".join(trimmed)

    string: str = attrs.field(converter=_docstring_trim)
    preamble: str = attrs.field(factory=lambda: Toplevel._get_config().typst_preamble, converter=_docstring_trim)
    concatenate: bool = False
    align: str | None = attrs.field(factory=lambda: Toplevel._get_config().typst_align)
    font: str | tuple[str, ...] | None = attrs.field(factory=lambda: Toplevel._get_config().typst_font)
    color: ColorType | None = attrs.field(factory=lambda: Toplevel._get_config().default_color)


#class BoundaryFlag(Enum):
#    START = 1
#    STOP = -1

#    def negate(
#        self: Self
#    ) -> BoundaryFlag:
#        return BoundaryFlag(-self.value)


#class Span:
#    __slots__ = (
#        "_start",
#        "_stop"
#    )

#    def __init__(
#        self: Self,
#        start: int,
#        stop: int
#    ) -> None:
#        assert 0 <= start <= stop
#        super().__init__()
#        self._start: int = start
#        self._stop: int = stop

#    def contains(
#        self: Self,
#        span: Span
#    ) -> bool:
#        return self._start <= span._start and self._stop >= span._stop

#    def as_slice(
#        self: Self
#    ) -> slice:
#        return slice(self._start, self._stop)

#    def get_boundary_index(
#        self: Self,
#        boundary_flag: BoundaryFlag
#    ) -> int:
#        return self._start if boundary_flag == BoundaryFlag.START else self._stop


#class CommandFlag(Enum):
#    STANDALONE = 0
#    OPEN = 1
#    CLOSE = -1


#class CommandInfo[AttributesT: TypedDict]:
#    __slots__ = (
#        "_command_items",
#        "_attributes_item"
#    )

#    def __init__(
#        self: Self,
#        match_obj_items: tuple[tuple[re.Match[str], str | None, CommandFlag], ...],
#        attributes_item: tuple[AttributesT, bool] | None
#    ) -> None:
#        super().__init__()
#        self._command_items: tuple[tuple[Span, str, CommandFlag], ...] = tuple(
#            (
#                Span(*match.span()),
#                replacement if replacement is not None else match.group(),
#                command_flag
#            )
#            for match, replacement, command_flag in match_obj_items
#        )
#        self._attributes_item: tuple[AttributesT, bool] | None = attributes_item


#class StandaloneCommandInfo[AttributesT: TypedDict](CommandInfo[AttributesT]):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        match: re.Match[str],
#        replacement: str | None = None
#    ) -> None:
#        super().__init__(
#            match_obj_items=(
#                (match, replacement, CommandFlag.STANDALONE),
#            ),
#            attributes_item=None
#        )


#class BalancedCommandInfo[AttributesT: TypedDict](CommandInfo[AttributesT]):
#    __slots__ = (
#        "_other_match_obj_item",
#        "_attributes_item"
#    )

#    def __init__(
#        self: Self,
#        attributes: AttributesT,
#        isolated: bool,
#        open_match_obj: re.Match[str],
#        close_match_obj: re.Match[str],
#        open_replacement: str | None = None,
#        close_replacement: str | None = None
#    ) -> None:
#        super().__init__(
#            match_obj_items=(
#                (open_match_obj, open_replacement, CommandFlag.OPEN),
#                (close_match_obj, close_replacement, CommandFlag.CLOSE)
#            ),
#            attributes_item=(attributes, isolated)
#        )


#@attrs.frozen(kw_only=True)
#class SpanInfo[AttributesT: TypedDict]:
#    span: Span
#    isolated: bool | None = None
#    attributes: AttributesT | None = None
#    local_color: ColorType | None = None
#    command_item: tuple[CommandInfo[AttributesT], str, CommandFlag] | None = None


#class ReplacementRecord:
#    __slots__ = (
#        "_span",
#        "_unlabelled_replacement",
#        "_labelled_replacement"
#    )

#    def __init__(
#        self: Self,
#        span: Span
#    ) -> None:
#        super().__init__()
#        self._span: Span = span
#        self._unlabelled_replacement: str = ""
#        self._labelled_replacement: str = ""

#    def write_replacements(
#        self: Self,
#        *,
#        unlabelled_replacement: str,
#        labelled_replacement: str
#    ) -> Self:
#        self._unlabelled_replacement = unlabelled_replacement
#        self._labelled_replacement = labelled_replacement
#        return self


#class InsertionRecord(ReplacementRecord):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        index: int
#    ) -> None:
#        super().__init__(Span(index, index))


#class StringMobjectKwargs(TypedDict, total=False):
#    #global_attributes: AttributesT
#    #local_attributes: dict[SelectorType, AttributesT]
#    #local_colors: dict[SelectorType, ColorType]
#    #isolate: Iterable[SelectorType]
#    #protect: Iterable[SelectorType]
#    concatenate: bool


#@attrs.frozen(kw_only=True)
#class StringMobjectCacheInput(CacheInput):
#    string: str
#    #global_attributes: AttributesT = attrs.field(factory=dict)
#    #local_attributes: dict[SelectorType, AttributesT] = attrs.field(factory=dict)
#    #local_colors: dict[SelectorType, ColorType] = attrs.field(factory=dict)
#    #isolate: Iterable[SelectorType] = ()
#    #protect: Iterable[SelectorType] = ()
#    concatenate: bool = False


#@attrs.frozen(kw_only=True)
#class StringMobjectOutput(MobjectOutput):
#    shape_mobjects: tuple[ShapeMobject, ...]
#    #labels: tuple[int, ...]
#    string: str
#    #isolated_spans: tuple[Span, ...]


#class StringMobjectJSON(MobjectJSON):
#    shape_mobjects: tuple[ShapeMobjectJSON, ...]
#    #labels: tuple[int, ...]
#    string: str
#    #isolated_spans: tuple[tuple[int, ...], ...]


#class StringMobjectIO[StringMobjectInputT: StringMobjectInput](MobjectIO[StringMobjectInputT, StringMobjectOutput, StringMobjectJSON]):
#    __slots__ = ()
#
#    @classmethod
#    def generate(
#        cls: type[Self],
#        input_data: StringMobjectInputT,
#        temp_path: pathlib.Path
#    ) -> StringMobjectOutput:

        #def replace_string(
        #    original_pieces: tuple[str, ...],
        #    replacement_pieces: tuple[str, ...]
        #) -> str:
        #    return "".join(itertools.chain.from_iterable(zip(
        #        ("", *original_pieces),
        #        replacement_pieces,
        #        strict=True
        #    )))

        #typst_path = temp_path.with_suffix(".typ")
        #svg_path = temp_path.with_suffix(".svg")

        #string = input_data.string
        #environment_prefix, environment_suffix = cls._get_environment_command_pair(input_data)
        #content = f"{environment_prefix}{string}{environment_suffix}"
        #typst_path.write_text(content, encoding="utf-8")

        #try:
        #    if subprocess.run((
        #        "typst",
        #        "compile",
        #        typst_path,
        #        svg_path
        #    ), stdout=subprocess.DEVNULL).returncode:
        #        raise OSError(f"StringMobjectIO: Failed to execute typst command")

        #    shape_mobjects = SVGMobjectIO._get_shape_mobjects_from_svg_path(
        #        svg_path=svg_path,
        #        scale=5.0  # TODO
        #    )

        #finally:
        #    for path in (typst_path, svg_path):
        #        path.unlink(missing_ok=True)


        #isolated_items, replacement_records = cls._get_isolated_items_and_replacement_records(
        #    string=string,
        #    environment_command_pair=cls._get_environment_command_pair(input_data),
        #    global_attributes_items=tuple(cls._iter_global_span_attributes(input_data, temp_path)),
        #    local_span_infos=tuple(itertools.chain((
        #        SpanInfo(
        #            span=span,
        #            isolated=False,
        #            attributes=attributes
        #        )
        #        for span, attributes in cls._iter_local_span_attributes(input_data, temp_path)
        #    ), (
        #        SpanInfo(
        #            span=span,
        #            isolated=True,
        #            attributes=cls._get_empty_attributes(),
        #            local_color=local_color
        #        )
        #        for selector, local_color in input_data.local_colors.items()
        #        for span in cls._iter_spans_by_selector(selector, string)
        #    ), (
        #        SpanInfo(
        #            span=span,
        #            isolated=True,
        #            attributes=cls._get_empty_attributes()
        #        )
        #        for selector in input_data.isolate
        #        for span in cls._iter_spans_by_selector(selector, string)
        #    ), (
        #        SpanInfo(
        #            span=span
        #        )
        #        for selector in input_data.protect
        #        for span in cls._iter_spans_by_selector(selector, string)
        #    ), (
        #        SpanInfo(
        #            span=span,
        #            command_item=(command_info, replacement, command_flag)
        #        )
        #        for command_info in cls._iter_command_infos(string)
        #        for span, replacement, command_flag in command_info._command_items
        #    )))
        #)
        #original_pieces = tuple(
        #    string[start:stop]
        #    for start, stop in zip(
        #        (replacement_record._span._stop for replacement_record in replacement_records[:-1]),
        #        (replacement_record._span._start for replacement_record in replacement_records[1:]),
        #        strict=True
        #    )
        #)
        #shape_mobject_items = tuple(cls._iter_shape_mobject_items(
        #    unlabelled_content=replace_string(
        #        original_pieces=original_pieces,
        #        replacement_pieces=tuple(
        #            replacement_record._unlabelled_replacement
        #            for replacement_record in replacement_records
        #        )
        #    ),
        #    labelled_content=replace_string(
        #        original_pieces=original_pieces,
        #        replacement_pieces=tuple(
        #            replacement_record._labelled_replacement
        #            for replacement_record in replacement_records
        #        )
        #    ),
        #    requires_labelling=len(isolated_items) > 1,
        #    input_data=input_data,
        #    temp_path=temp_path
        #))

        #for shape_mobject, label in shape_mobject_items:
        #    _, local_color = isolated_items[label]
        #    if local_color is not None:
        #        shape_mobject._color_._array_ = AnimatableColor._color_to_array(local_color)

        #return StringMobjectOutput(
        #    shape_mobjects=shape_mobjects,
        #    #labels=tuple(label for shape_mobject in shape_mobjects),
        #    string=string
        #    #isolated_spans=tuple(isolated_span for isolated_span, _ in isolated_items)
        #)

    #@classmethod
    #def dump_json(
    #    cls: type[Self],
    #    output_data: StringMobjectOutput
    #) -> StringMobjectJSON:
    #    return StringMobjectJSON(
    #        shape_mobjects=tuple(
    #            SVGMobjectIO._shape_mobject_to_json(shape_mobject)
    #            for shape_mobject in output_data.shape_mobjects
    #        ),
    #        #labels=output_data.labels,
    #        string=output_data.string
    #        #isolated_spans=tuple((span._start, span._stop) for span in output_data.isolated_spans)
    #    )

    #@classmethod
    #def load_json(
    #    cls: type[Self],
    #    json_data: StringMobjectJSON
    #) -> StringMobjectOutput:
    #    return StringMobjectOutput(
    #        shape_mobjects=tuple(
    #            SVGMobjectIO._json_to_shape_mobject(shape_mobject_json)
    #            for shape_mobject_json in json_data["shape_mobjects"]
    #        ),
    #        #labels=json_data["labels"],
    #        string=json_data["string"]
    #        #isolated_spans=tuple(Span(*span_values) for span_values in json_data["isolated_spans"])
    #    )

    #@classmethod
    #def _get_isolated_items_and_replacement_records(
    #    cls: type[Self],
    #    string: str,
    #    environment_command_pair: tuple[str, str],
    #    global_attributes_items: tuple[AttributesT, ...],
    #    local_span_infos: tuple[SpanInfo[AttributesT], ...]
    #) -> tuple[tuple[tuple[Span, ColorType | None], ...], tuple[ReplacementRecord, ...]]:

    #    def get_sorting_key(
    #        span_boundary: tuple[SpanInfo[AttributesT], BoundaryFlag]
    #    ) -> tuple[int, int, int]:
    #        span_info, boundary_flag = span_boundary
    #        span = span_info.span
    #        index = span.get_boundary_index(boundary_flag)
    #        paired_index = span.get_boundary_index(boundary_flag.negate())
    #        # All local spans are guaranteed to have non-zero widths.
    #        return (
    #            index,
    #            boundary_flag.value,
    #            -paired_index
    #        )

    #    insertion_record_items: list[tuple[InsertionRecord, InsertionRecord, AttributesT, bool, ColorType | None]] = []
    #    replacement_records: collections.deque[ReplacementRecord] = collections.deque()
    #    bracket_counter = itertools.count()
    #    protect_level = 0
    #    bracket_stack: list[int] = []
    #    open_command_stack: list[tuple[InsertionRecord, CommandInfo[AttributesT]]] = []
    #    start_stack: list[tuple[SpanInfo[AttributesT], InsertionRecord, int, tuple[int, ...]]] = []
    #    local_color_stack: list[ColorType] = []

    #    for span_info, boundary_flag in sorted((
    #        (span_info, boundary_flag)
    #        for boundary_flag in (BoundaryFlag.STOP, BoundaryFlag.START)
    #        for span_info in local_span_infos[::boundary_flag.value]
    #    ), key=get_sorting_key):
    #        span = span_info.span
    #        if span_info.isolated is None:
    #            protect_level += boundary_flag.value
    #            if span_info.command_item is None:
    #                continue
    #            if boundary_flag == BoundaryFlag.START:
    #                continue
    #            command_info, command_replacement, command_flag = span_info.command_item
    #            command_replacement_record = ReplacementRecord(span).write_replacements(
    #                unlabelled_replacement=command_replacement,
    #                labelled_replacement=command_replacement
    #            )
    #            if command_flag == CommandFlag.OPEN:
    #                bracket_stack.append(next(bracket_counter))
    #                open_insertion_record = InsertionRecord(span._stop)
    #                replacement_records.append(command_replacement_record)
    #                replacement_records.append(open_insertion_record)
    #                open_command_stack.append((open_insertion_record, command_info))
    #            elif command_flag == CommandFlag.CLOSE:
    #                bracket_stack.pop()
    #                close_insertion_record = InsertionRecord(span._start)
    #                replacement_records.append(close_insertion_record)
    #                replacement_records.append(command_replacement_record)
    #                open_insertion_record, open_command_info = open_command_stack.pop()
    #                assert open_command_info is command_info
    #                assert command_info._attributes_item is not None
    #                attributes, isolated = command_info._attributes_item
    #                insertion_record_items.append((
    #                    open_insertion_record,
    #                    close_insertion_record,
    #                    attributes,
    #                    isolated,
    #                    None
    #                ))
    #            elif command_flag == CommandFlag.STANDALONE:
    #                replacement_records.append(command_replacement_record)
    #            continue

    #        if boundary_flag == BoundaryFlag.START:
    #            start_insertion_record = InsertionRecord(span._start)
    #            replacement_records.append(start_insertion_record)
    #            start_stack.append((span_info, start_insertion_record, protect_level, tuple(bracket_stack)))
    #            if span_info.local_color is not None:
    #                local_color_stack.append(span_info.local_color)
    #        elif boundary_flag == BoundaryFlag.STOP:
    #            stop_insertion_record = InsertionRecord(span._stop)
    #            replacement_records.append(stop_insertion_record)
    #            start_span_info, start_insertion_record, start_protect_level, start_bracket_stack = start_stack.pop()
    #            if start_protect_level or protect_level:
    #                continue
    #            if start_span_info is not span_info:
    #                raise ValueError(
    #                    f"Partly overlapping substrings detected: '{string[start_span_info.span.as_slice()]}', '{string[span.as_slice()]}'"
    #                )
    #            if start_bracket_stack != tuple(bracket_stack):
    #                raise ValueError(
    #                    f"Cannot handle substring: '{string[span.as_slice()]}'"
    #                )
    #            if span_info.local_color is not None:
    #                local_color = span_info.local_color
    #                local_color_stack.pop()
    #            else:
    #                local_color = local_color_stack[-1] if local_color_stack else None
    #            assert span_info.attributes is not None
    #            insertion_record_items.append((
    #                start_insertion_record,
    #                stop_insertion_record,
    #                span_info.attributes,
    #                span_info.isolated,
    #                local_color
    #            ))

    #    assert protect_level == 0
    #    assert not bracket_stack
    #    assert not open_command_stack
    #    assert not start_stack
    #    assert not local_color_stack

    #    label_counter = itertools.count()
    #    isolated_items: list[tuple[Span, ColorType | None]] = []

    #    global_label = next(label_counter)
    #    global_span = Span(0, len(string))
    #    isolated_items.append((global_span, None))

    #    start_environment_command, stop_environment_command = environment_command_pair
    #    replacement_records.appendleft(InsertionRecord(global_span._start).write_replacements(
    #        unlabelled_replacement=start_environment_command,
    #        labelled_replacement=start_environment_command
    #    ))
    #    replacement_records.append(InsertionRecord(global_span._stop).write_replacements(
    #        unlabelled_replacement=stop_environment_command,
    #        labelled_replacement=stop_environment_command
    #    ))

    #    for attributes in reversed(global_attributes_items):
    #        labelled_attributes = cls._convert_attributes_for_labelling(attributes, global_label)
    #        start_unlabelled_insertion, stop_unlabelled_insertion = cls._get_command_pair(attributes)
    #        start_labelled_insertion, stop_labelled_insertion = cls._get_command_pair(labelled_attributes)
    #        replacement_records.appendleft(InsertionRecord(global_span._start).write_replacements(
    #            unlabelled_replacement=start_unlabelled_insertion,
    #            labelled_replacement=start_labelled_insertion
    #        ))
    #        replacement_records.append(InsertionRecord(global_span._stop).write_replacements(
    #            unlabelled_replacement=stop_unlabelled_insertion,
    #            labelled_replacement=stop_labelled_insertion
    #        ))

    #    for start_insertion_record, stop_insertion_record, attributes, isolated, local_color in insertion_record_items:
    #        if isolated:
    #            local_label = next(label_counter)
    #            local_span = Span(start_insertion_record._span._stop, stop_insertion_record._span._start)
    #            isolated_items.append((local_span, local_color))
    #        else:
    #            local_label = None
    #        labelled_attributes = cls._convert_attributes_for_labelling(attributes, local_label)
    #        start_unlabelled_insertion, stop_unlabelled_insertion = cls._get_command_pair(attributes)
    #        start_labelled_insertion, stop_labelled_insertion = cls._get_command_pair(labelled_attributes)
    #        start_insertion_record.write_replacements(
    #            unlabelled_replacement=start_unlabelled_insertion,
    #            labelled_replacement=start_labelled_insertion
    #        )
    #        stop_insertion_record.write_replacements(
    #            unlabelled_replacement=stop_unlabelled_insertion,
    #            labelled_replacement=stop_labelled_insertion
    #        )

    #    return tuple(isolated_items), tuple(replacement_records)

    #@classmethod
    #def _iter_spans_by_selector(
    #    cls: type[Self],
    #    selector: SelectorType,
    #    string: str
    #) -> Iterator[Span]:
    #    match selector:
    #        case str():
    #            substr_len = len(selector)
    #            if not substr_len:
    #                return
    #            index = 0
    #            while True:
    #                index = string.find(selector, index)
    #                if index == -1:
    #                    break
    #                yield Span(index, index + substr_len)
    #                index += substr_len
    #        case re.Pattern():
    #            for match in selector.finditer(string):
    #                start, stop = match.span()
    #                if start < stop:
    #                    yield Span(start, stop)
    #        case slice(start=int(start), stop=int(stop)):
    #            l = len(string)
    #            start = min(start, l) if start >= 0 else max(start + l, 0)
    #            stop = min(stop, l) if stop >= 0 else max(stop + l, 0)
    #            if start < stop:
    #                yield Span(start, stop)

    #@classmethod
    #def _iter_shape_mobject_items(
    #    cls: type[Self],
    #    unlabelled_content: str,
    #    labelled_content: str,
    #    requires_labelling: bool,
    #    input_data: StringMobjectInputT,
    #    temp_path: pathlib.Path
    #) -> Iterator[tuple[ShapeMobject, int]]:
    #    unlabelled_shape_mobjects = cls._get_shape_mobjects(unlabelled_content, input_data, temp_path)
    #    if input_data.concatenate:
    #        yield ShapeMobject(Shape().concatenate(tuple(
    #            shape_mobject._shape_ for shape_mobject in unlabelled_shape_mobjects
    #        ))), 0
    #        return

    #    if not requires_labelling or not unlabelled_shape_mobjects:
    #        for unlabelled_shape_mobject in unlabelled_shape_mobjects:
    #            yield unlabelled_shape_mobject, 0
    #        return

    #    labelled_shape_mobjects = cls._get_shape_mobjects(labelled_content, input_data, temp_path)
    #    assert len(unlabelled_shape_mobjects) == len(labelled_shape_mobjects)

    #    unlabelled_radii = Mobject().add(*unlabelled_shape_mobjects).box.get_radii()
    #    labelled_radii = Mobject().add(*labelled_shape_mobjects).box.get_radii()
    #    scale_factor = unlabelled_radii / labelled_radii
    #    distance_matrix = scipy.spatial.distance.cdist(
    #        tuple(shape.box.get() for shape in unlabelled_shape_mobjects),
    #        tuple(shape.box.get() * scale_factor for shape in labelled_shape_mobjects)
    #    )
    #    for unlabelled_index, labelled_index in zip(*scipy.optimize.linear_sum_assignment(distance_matrix), strict=True):
    #        yield (
    #            unlabelled_shape_mobjects[unlabelled_index],
    #            int(AnimatableColor._array_to_hex(labelled_shape_mobjects[labelled_index]._color_._array_)[1:], 16)
    #        )

    #@classmethod
    #def _get_shape_mobjects(
    #    cls: type[Self],
    #    content: str,
    #    input_data: StringMobjectInputT,
    #    temp_path: pathlib.Path
    #) -> tuple[ShapeMobject, ...]:
    #    svg_path = temp_path.with_suffix(".svg")
    #    try:
    #        cls._create_svg(
    #            content=content,
    #            input_data=input_data,
    #            svg_path=svg_path
    #        )
    #        shape_mobjects = SVGMobjectIO._get_shape_mobjects_from_svg_path(
    #            svg_path=svg_path,
    #            scale=cls._get_adjustment_scale()
    #        )
    #    finally:
    #        svg_path.unlink(missing_ok=True)

    #    return shape_mobjects

    #@classmethod
    #@abstractmethod
    #def _create_svg(
    #    cls: type[Self],
    #    content: str,
    #    input_data: StringMobjectInputT,
    #    svg_path: pathlib.Path
    #) -> None:
    #    pass

    #@classmethod
    #@abstractmethod
    #def _get_adjustment_scale(
    #    cls: type[Self]
    #) -> float:
    #    # The line height shall be roughly equal to 1.0 for default fonts.
    #    pass

    #@classmethod
    #def _get_environment_command_pair(
    #    cls: type[Self],
    #    input_data: StringMobjectInputT
    #) -> tuple[str, str]:
    #    return "", ""

    #@classmethod
    #def _iter_global_span_attributes(
    #    cls: type[Self],
    #    input_data: StringMobjectInputT,
    #    temp_path: pathlib.Path
    #) -> Iterator[AttributesT]:
    #    yield input_data.global_attributes

    #@classmethod
    #def _iter_local_span_attributes(
    #    cls: type[Self],
    #    input_data: StringMobjectInputT,
    #    temp_path: pathlib.Path
    #) -> Iterator[tuple[Span, AttributesT]]:
    #    for selector, attributes in input_data.local_attributes.items():
    #        for span in cls._iter_spans_by_selector(selector, input_data.string):
    #            yield span, attributes

    #@classmethod
    #@abstractmethod
    #def _get_empty_attributes(
    #    cls: type[Self]
    #) -> AttributesT:
    #    pass

    #@classmethod
    #@abstractmethod
    #def _get_command_pair(
    #    cls: type[Self],
    #    attributes: AttributesT
    #) -> tuple[str, str]:
    #    pass

    #@classmethod
    #@abstractmethod
    #def _convert_attributes_for_labelling(
    #    cls: type[Self],
    #    attributes: AttributesT,
    #    label: int | None
    #) -> AttributesT:
    #    pass

    #@classmethod
    #@abstractmethod
    #def _iter_command_infos(
    #    cls: type[Self],
    #    string: str
    #) -> Iterator[CommandInfo[AttributesT]]:
    #    pass


class TypstMobject[TypstMobjectInputsT: TypstMobjectInputs](CachedMobject[TypstMobjectInputsT]):
    __slots__ = (
        "_inputs",
        #"_preamble",
        #"_parameters",
        #"_shape_mobjects",
        "_selector_to_indices_dict"
        #"_labels",
        #"_isolated_spans"
    )

    def __init__(
        self: Self,
        inputs: TypstMobjectInputsT
        #string: str,
        #preamble: str,
        #parameters: TypstMobjectParametersT
        #output_data: TypstMobjectOutput
    ) -> None:
        #string = cls._docstring_trim(string)
        #preamble = cls._docstring_trim(preamble)
        super().__init__(inputs)
        self._inputs: TypstMobjectInputsT = inputs
        #self._preamble: str = preamble
        #self._parameters: TypstMobjectParametersT = parameters
        self._selector_to_indices_dict: dict[SelectorType, list[int]] = {}
        self.scale(0.1)
        #output_data = type(self)._get_output_data(input_data)
        #self._input_data: TypstMobjectInputT = input_data
        #self._shape_mobjects: tuple[ShapeMobject, ...] = output_data.shape_mobjects
        #self._string: str = output_data.string
        ##self._labels: tuple[int, ...] = output_data.labels
        ##self._isolated_spans: tuple[Span, ...] = output_data.isolated_spans
        #self.add(*output_data.shape_mobjects)

    @classmethod
    def _generate_shape_mobjects(
        cls: type[Self],
        inputs: TypstMobjectInputsT,
        temp_path: pathlib.Path
    ) -> tuple[ShapeMobject, ...]:
        preamble = cls._get_preamble_from_inputs(inputs)
        environment_begin, environment_end = cls._get_environment_pair_from_inputs(inputs)
        content = "\n".join(filter(None, (
            preamble,
            inputs.preamble,
            f"{environment_begin}{inputs.string}{environment_end}"
        )))

        svg_path = temp_path.with_suffix(".svg")
        typst_path = temp_path.with_suffix(".typ")
        typst_path.write_text(content, encoding="utf-8")

        try:
            if (stdout := subprocess.check_output((
                "typst",
                "compile",
                typst_path,
                svg_path
            ))):
                error = OSError("Typst error")
                error.add_note(stdout.decode())
                raise error

            shape_mobjects = SVGMobject._generate_shape_mobjects_from_svg(svg_path)

        finally:
            for path in (svg_path, typst_path):
                path.unlink(missing_ok=True)

        if inputs.concatenate:
            shape_mobjects = (ShapeMobject(Shape().concatenate(tuple(
                shape_mobject._shape_ for shape_mobject in shape_mobjects
            ))),)
        return shape_mobjects

    @classmethod
    def _get_preamble_from_inputs(
        cls: type[Self],
        inputs: TypstMobjectInputsT
    ) -> str:
        return "\n".join(filter(None, (
            f"""#set align({
                inputs.align
            })""" if inputs.align is not None else "",
            f"""#set text(font: {
                f"\"{inputs.font.replace("\"", "\\\"")}\""
                if isinstance(inputs.font, str)
                else f"({", ".join(f"\"{font.replace("\"", "\\\"")}\"" for font in inputs.font)})"
            })""" if inputs.font is not None else "",
            f"""#set text(fill: rgb({
                ", ".join(f"{component * 100.0}%" for component in AnimatableColor._color_to_array(inputs.color))
            }))""" if inputs.color is not None else ""
        )))

    @classmethod
    def _get_environment_pair_from_inputs(
        cls: type[Self],
        inputs: TypstMobjectInputsT
    ) -> tuple[str, str]:
        return "", ""

    @classmethod
    @abstractmethod
    def _get_labelled_inputs(
        cls: type[Self],
        inputs: TypstMobjectInputsT,
        label_to_selector_dict: dict[int, SelectorType]
    ) -> TypstMobjectInputsT:
        pass

    def _build_from_selector(
        self: Self,
        selector: SelectorType
    ) -> ShapeMobject:
        return ShapeMobject().add(*(
            self._shape_mobjects[index]
            for index in self._selector_to_indices_dict[selector]
        ))

    def probe(
        self: Self,
        selectors: tuple[SelectorType, ...]
    ) -> None:
        selectors = tuple(
            selector for selector in selectors
            if selector not in self._selector_to_indices_dict
        )
        if not selectors:
            return

        # Label by 255 opacity values (opacity 0xFF is reserved).
        assert len(selectors) <= 255
        cls = type(self)
        label_to_selector_dict = {
            label: selector
            for label, selector in enumerate(selectors)
        }
        labelled_inputs = cls._get_labelled_inputs(
            inputs=self._inputs,
            label_to_selector_dict=label_to_selector_dict
        )
        labelled_shape_mobjects = cls._get_shape_mobjects(labelled_inputs)
        assert len(self._shape_mobjects) == len(labelled_shape_mobjects)

        self._selector_to_indices_dict.update((selector, []) for selector in selectors)
        for index, labelled_shape_mobject in enumerate(labelled_shape_mobjects):
            label = int(labelled_shape_mobject._opacity_._array_ * 255.0)
            if label == 255:
                continue
            self._selector_to_indices_dict[label_to_selector_dict[label]].append(index)

    def select(
        self: Self,
        selector: SelectorType
    ) -> ShapeMobject:
        self.probe((selector,))
        return self._build_from_selector(selector)

    def set_local_styles(
        self: Self,
        selector_to_kwargs_dict: dict[SelectorType, SetKwargs]
    ) -> Self:
        self.probe(tuple(selector_to_kwargs_dict))
        for selector, kwargs in selector_to_kwargs_dict.items():
            self._build_from_selector(selector).set(**kwargs)
        return self

    def set_local_colors(
        self: Self,
        selector_to_color_dict: dict[SelectorType, ColorType]
    ) -> Self:
        self.set_local_styles({
            selector: {"color": color}
            for selector, color in selector_to_color_dict.items()
        })
        return self

    #@classmethod
    #def _get_output_data(
    #    cls: type[Self],
    #    input_data: StringMobjectInputT
    #) -> StringMobjectOutput:
    #    return StringMobjectIO.get(input_data)

    #def _get_indices_tuple_by_selector(
    #    self: Self,
    #    selector: SelectorType
    #) -> tuple[tuple[int, ...], ...]:
    #    return tuple(
    #        tuple(
    #            index
    #            for index, label in enumerate(self._labels)
    #            if specified_span.contains(self._isolated_spans[label])
    #        )
    #        for specified_span in StringMobjectIO._iter_spans_by_selector(selector, self._string)
    #    )

    #def _build_from_indices(
    #    self: Self,
    #    indices: tuple[int, ...]
    #) -> Mobject:
    #    return Mobject().add(*(
    #        self._shape_mobjects[index]
    #        for index in indices
    #    ))

    #def _build_from_indices_tuple(
    #    self: Self,
    #    indices_tuple: tuple[tuple[int, ...], ...]
    #) -> Mobject:
    #    return Mobject().add(*(
    #        self._build_from_indices(indices)
    #        for indices in indices_tuple
    #    ))

    #def select_part(
    #    self: Self,
    #    selector: SelectorType,
    #    index: int = 0
    #) -> Mobject:
    #    return self._build_from_indices(self._get_indices_tuple_by_selector(selector)[index])

    #def select_parts(
    #    self: Self,
    #    selector: SelectorType
    #) -> Mobject:
    #    return self._build_from_indices_tuple(self._get_indices_tuple_by_selector(selector))
