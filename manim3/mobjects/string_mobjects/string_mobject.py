from __future__ import annotations


import itertools
import pathlib
import re
from abc import abstractmethod
from enum import Enum
from typing import (
    Iterator,
    Self,
    TypedDict
)

import attrs

import scipy.optimize
import scipy.spatial.distance

#from ...animatables.arrays.animatable_color import AnimatableColor
from ...animatables.shape import Shape
from ...constants.custom_typing import (
    ColorT,
    SelectorT
)
#from ...toplevel.toplevel import Toplevel
from ...utils.color_utils import ColorUtils
from ..mobject import Mobject
from ..mobject_io import (
    MobjectIO,
    MobjectInput,
    MobjectJSON,
    MobjectOutput
)
from ..shape_mobjects.shape_mobject import ShapeMobject
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
        "_attribs_item"
    )

    def __init__(
        self: Self,
        match_obj_items: tuple[tuple[re.Match[str], str | None, CommandFlag], ...],
        #command_flag: CommandFlag = CommandFlag.OTHER,
        attribs_item: tuple[dict[str, str], bool] | None,
        #attribs: dict[str, str] | None = None  # Only takes effect for open commands.
    ) -> None:
        super().__init__()
        self._command_items: tuple[tuple[Span, str, CommandFlag], ...] = tuple(
            (
                Span(*match_obj.span()),
                replacement if replacement is not None else match_obj.group(),
                command_flag
            )
            for match_obj, replacement, command_flag in match_obj_items
        )
        self._attribs_item: tuple[dict[str, str], bool] | None = attribs_item

    #@abstractmethod
    #def _iter_match_obj_items(
    #    self: Self
    #) -> Iterator[tuple[re.Match[str], str | None]]:
    #    pass

    #@abstractmethod
    #def _iter_attribs_items(
    #    self: Self
    #) -> Iterator[tuple[bool, dict[str, str]]]:
    #    pass

    #def _iter_span_items(
    #    self: Self
    #) -> Iterator[tuple[Span, str]]:
    #    for match_obj, replacement in self._iter_match_obj_items():
    #        yield Span(*match_obj.span()), replacement if replacement is not None else match_obj.group()


class StandaloneCommandInfo(CommandInfo):
    __slots__ = ()

    def __init__(
        self: Self,
        match_obj: re.Match[str],
        replacement: str | None = None
    ) -> None:
        super().__init__(
            match_obj_items=(
                (match_obj, replacement, CommandFlag.STANDALONE),
            ),
            attribs_item=None
        )


class BalancedCommandInfo(CommandInfo):
    __slots__ = (
        "_other_match_obj_item",
        "_attribs_item"
    )

    def __init__(
        self: Self,
        attribs: dict[str, str],
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
            attribs_item=(attribs, isolated)
        )


@attrs.frozen(kw_only=True)
class SpanInfo:
    span: Span
    isolated: bool | None = None
    attribs: dict[str, str] | None = None
    local_color: ColorT | None = None
    command_item: tuple[CommandInfo, str, CommandFlag] | None = None
    #(span, True, global_attribs, None)


#class SpanBoundary:
#    __slots__ = (
#        "_span",
#        "_boundary_flag"
#    )

#    def __init__(
#        self,
#        span: Span,
#        boundary_flag: BoundaryFlag
#    ) -> None:
#        super().__init__()
#        self._span: Span = span
#        self._boundary_flag: BoundaryFlag = boundary_flag

#    def get_sorting_key(
#        self: Self
#    ) -> tuple[int, int, int]:
#        span = self._span
#        boundary_flag = self._boundary_flag
#        flag_value = boundary_flag.value
#        index = span.get_boundary_index(boundary_flag)
#        paired_index = span.get_boundary_index(boundary_flag.negate())
#        # All spans have nonzero widths.
#        return (
#            index,
#            flag_value,
#            -paired_index
#        )


#class IsolatedSpanBoundary(SpanBoundary):
#    __slots__ = ()


#class AttributedSpanBoundary(IsolatedSpanBoundary):
#    __slots__ = ("_attribs",)

#    def __init__(
#        self,
#        span: Span,
#        boundary_flag: BoundaryFlag,
#        attribs: dict[str, str]
#    ) -> None:
#        super().__init__(span, boundary_flag)
#        self._attribs: dict[str, str] = attribs


#class ProtectedSpanBoundary(SpanBoundary):
#    __slots__ = ()


#class CommandSpanBoundary(ProtectedSpanBoundary):
#    __slots__ = ("_command_info",)

#    def __init__(
#        self,
#        command_info: CommandInfo,
#        boundary_flag: BoundaryFlag
#    ) -> None:
#        super().__init__(Span(*command_info.span), boundary_flag)
#        self._command_info: CommandInfo = command_info


class ReplacementRecord:
    __slots__ = (
        "_span",
        "_unlabelled_replacement",
        "_labelled_replacement"
    )

    def __init__(
        self,
        span: Span
        #unlabelled_replacement: str,
        #labelled_replacement: str
    ) -> None:
        super().__init__()
        self._span: Span = span
        self._unlabelled_replacement: str = ""
        self._labelled_replacement: str = ""

    def write_replacements(
        self: Self,
        *,
        #label: int,
        #boundary_flag: BoundaryFlag,
        unlabelled_replacement: str,
        labelled_replacement: str
    ) -> None:
        #self._label = label
        #self._boundary_flag = boundary_flag
        self._unlabelled_replacement = unlabelled_replacement
        self._labelled_replacement = labelled_replacement
        #self._activated = True


class InsertionRecord(ReplacementRecord):
    __slots__ = (
        #"_label",
        #"_boundary_flag",
        #"_activated",
    )

    def __init__(
        self,
        index: int
    ) -> None:
        super().__init__(Span(index, index))
        #self._label: int = NotImplemented
        #self._boundary_flag: BoundaryFlag = NotImplemented
        #self._activated = False


class StringMobjectKwargs(TypedDict, total=False):
    local_colors: dict[SelectorT, ColorT]
    isolate: list[SelectorT]
    protect: list[SelectorT]
    #color: ColorT
    #global_attribs: dict[str, str]
    #local_attribs: dict[SelectorT, dict[str, str]]
    concatenate: bool


@attrs.frozen(kw_only=True)
class StringMobjectInput(MobjectInput):
    string: str
    local_colors: dict[SelectorT, ColorT] = attrs.field(factory=dict)
    isolate: list[SelectorT] = attrs.field(factory=list)
    protect: list[SelectorT] = attrs.field(factory=list)
    #color: ColorT = attrs.field(factory=lambda: Toplevel.config.default_color)
    #global_attribs: dict[str, str] = attrs.field(factory=dict)
    #local_attribs: dict[SelectorT, dict[str, str]] = attrs.field(factory=dict)
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
        #global_span_attribs = cls._get_global_span_attribs(input_data, temp_path)
        #local_span_attribs = cls._get_local_span_attribs(input_data, temp_path)
        #global_span_color = input_data.global_color
        #local_span_color = {
        #    span: color
        #    for selector, color in input_data.local_color.items()
        #    for span in cls._iter_spans_by_selector(selector, string)
        #}
        #print(local_span_color)  # TODO

        isolated_items, replacement_records = cls._get_isolated_items_and_replacement_records(
            string=string,
            span_infos=(
                SpanInfo(span=Span(0, len(string)), isolated=True, attribs=cls._get_global_span_attribs(input_data, temp_path)),
                *(
                    SpanInfo(span=span, isolated=False, attribs=attribs)
                    for span, attribs in cls._iter_local_span_attribs(input_data, temp_path)
                ),
                *(
                    SpanInfo(span=span, isolated=True, local_color=local_color)
                    for selector, local_color in input_data.local_colors.items()
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                *(
                    SpanInfo(span=span, isolated=True)
                    for selector in input_data.isolate
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                *(
                    SpanInfo(span=span)
                    for selector in input_data.protect
                    for span in cls._iter_spans_by_selector(selector, string)
                ),
                *(
                    SpanInfo(span=span, command_item=(command_info, replacement, command_flag))
                    for command_info in cls._iter_command_infos(string)
                    for span, replacement, command_flag in command_info._command_items
                )
            )
            #isolate=input_data.isolate,
            #protect=input_data.protect,
            #local_colors=input_data.local_colors,
            #colored_span_items=tuple(
            #    (span, color)
            #    for selector, color in input_data.local_colors.items()
            #    for span in cls._iter_spans_by_selector(selector, string)
            #),
            #isolated_spans=tuple(itertools.chain.from_iterable(
            #    cls._iter_spans_by_selector(selector, string)
            #    for selector in input_data.isolate
            #)),
            #protected_spans=tuple(itertools.chain.from_iterable(
            #    cls._iter_spans_by_selector(selector, string)
            #    for selector in input_data.protect
            #)),
            #command_infos=tuple(cls._iter_command_infos(string)),
            ##command_info_pairs=tuple(cls._iter_command_info_pairs(string)),
            #global_span_attribs=cls._get_global_span_attribs(input_data, temp_path),
            #local_span_attribs_items=tuple(cls._iter_local_span_attribs(input_data, temp_path))
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
            concatenate=input_data.concatenate,
            requires_labelling=len(isolated_items) > 1,
            input_data=input_data,
            temp_path=temp_path
        ))

        #for shape_mobject in shape_mobjects:
        #    shape_mobject._color_ = AnimatableColor(global_span_color)
        #for span, color in local_span_color.items():
        #    for index in cls._get_indices_by_span(span, labels, isolated_spans):
        #        shape_mobjects[index]._color_ = AnimatableColor(color)
        for shape_mobject, label in shape_mobject_items:
            _, local_color = isolated_items[label]
            if local_color is not None:
                shape_mobject._color_._array_ = ColorUtils.standardize_color(local_color)

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
        concatenate: bool,
        requires_labelling: bool,
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> Iterator[tuple[ShapeMobject, int]]:
        unlabelled_shape_mobjects = cls._get_shape_mobjects(unlabelled_content, input_data, temp_path)
        if concatenate:
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

        unlabelled_radii = ShapeMobject().add(*unlabelled_shape_mobjects).box.get_radii()
        labelled_radii = ShapeMobject().add(*labelled_shape_mobjects).box.get_radii()
        scale_factor = labelled_radii / unlabelled_radii
        distance_matrix = scipy.spatial.distance.cdist(
            [shape.box.get() for shape in unlabelled_shape_mobjects],
            [shape.box.get() * scale_factor for shape in labelled_shape_mobjects]
        )
        #unlabelled_indices, labelled_indices = scipy.optimize.linear_sum_assignment(distance_matrix)
        #unlabelled_indices = tuple(int(index) for index in unlabelled_indices)
        #labelled_indices = tuple(int(index) for index in labelled_indices)
        for unlabelled_index, labelled_index in zip(*scipy.optimize.linear_sum_assignment(distance_matrix)):
            yield (
                unlabelled_shape_mobjects[unlabelled_index],
                int(ColorUtils.color_to_hex(labelled_shape_mobjects[labelled_index]._color_._array_)[1:], 16)
            )
        #return tuple(
        #    int(ColorUtils.color_to_hex(labelled_shape_mobjects[labelled_index]._color_._array_)[1:], 16)
        #    for labelled_index in labelled_indices
        #), tuple(
        #    unlabelled_shape_mobjects[unlabelled_index]
        #    for unlabelled_index in unlabelled_indices
        #)

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
    def _get_global_span_attribs(
        cls: type[Self],
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        return {}
        #global_span_attribs: dict[str, str] = {}
        #global_span_attribs.update(input_data.global_attribs)
        #return global_span_attribs

    @classmethod
    def _iter_local_span_attribs(
        cls: type[Self],
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> Iterator[tuple[Span, dict[str, str]]]:
        yield from ()
        #local_span_attribs: dict[Span, dict[str, str]] = {}
        #for selector, local_attribs in input_data.local_attribs.items():
        #    for span in cls._iter_spans_by_selector(selector, input_data.string):
        #        local_span_attribs.setdefault(span, {}).update(local_attribs)
        #return local_span_attribs

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
        #isolate: list[SelectorT],
        #protect: list[SelectorT],
        #local_colors: dict[SelectorT, ColorT],
        #colored_span_items: tuple[tuple[Span, ColorT], ...],
        #isolated_spans: tuple[Span, ...],
        #protected_spans: tuple[Span, ...],
        #command_infos: tuple[CommandInfo, ...],
        #command_info_pairs: tuple[tuple[CommandInfo, CommandInfo, dict[str, str], bool], ...],
        #global_span_attribs: dict[str, str],
        #local_span_attribs_items: tuple[tuple[Span, dict[str, str]], ...]
    ) -> tuple[tuple[tuple[Span, ColorT | None], ...], tuple[ReplacementRecord, ...]]:
        #tuple[Span, BoundaryFlag, bool, dict[str, str] | None, ]
        #(span, True, global_attribs, None)
        #(span, False, attribs, None)
        #(span, True, None, None)
        #(span, None, None, None)
        #(span, None, None, command_info)

        def get_sorting_key(
            span_boundary: tuple[SpanInfo, BoundaryFlag]
        ) -> tuple[int, int, int]:
            span_info, boundary_flag = span_boundary
            span = span_info.span
            flag_value = boundary_flag.value
            index = span.get_boundary_index(boundary_flag)
            paired_index = span.get_boundary_index(boundary_flag.negate())
            # All spans have nonzero widths.
            return (
                index,
                flag_value,
                -paired_index
            )

        #span_boundaries = sorted((
        #    (span_info, boundary_flag)
        #    for boundary_flag in (BoundaryFlag.STOP, BoundaryFlag.START)
        #    for span_info in span_infos[::boundary_flag.value]
        #), key=get_sorting_key)
        #span_boundaries = sorted(itertools.chain.from_iterable(
        #    tuple(itertools.chain(
        #        (
        #            AttributedSpanBoundary(span=span, boundary_flag=boundary_flag, attribs=attribs)
        #            for span, attribs in local_span_attribs.items()
        #        ),
        #        (
        #            IsolatedSpanBoundary(span=span, boundary_flag=boundary_flag)
        #            for span in isolated_spans
        #        ),
        #        (
        #            ProtectedSpanBoundary(span=span, boundary_flag=boundary_flag)
        #            for span in protected_spans
        #        ),
        #        (
        #            CommandSpanBoundary(command_info=command_info, boundary_flag=boundary_flag)
        #            for command_info in cls._iter_command_infos(string)
        #        )
        #    ))[::boundary_flag.value]
        #    for boundary_flag in (BoundaryFlag.STOP, BoundaryFlag.START)
        #), key=SpanBoundary.get_sorting_key)

        insertion_record_items: list[tuple[InsertionRecord, InsertionRecord, dict[str, str], bool, ColorT | None]] = []
        replacement_records: list[ReplacementRecord] = []
        bracket_counter = itertools.count()
        protect_level = 0
        bracket_stack: list[int] = []
        open_command_stack: list[tuple[InsertionRecord, CommandInfo]] = []
        start_stack: list[tuple[SpanInfo, InsertionRecord, int, tuple[int, ...]]] = []
        local_color_stack: list[ColorT] = []
        #global_start_insertion_record = InsertionRecord(0)
        #global_stop_insertion_record = InsertionRecord(len(string))
        #labelled_items.append((
        #    global_start_insertion_record,
        #    global_stop_insertion_record,
        #    global_span_attribs
        #))
        #replacement_records.append(global_start_insertion_record)

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
                    assert (attribs_item := command_info._attribs_item) is not None
                    attribs, isolated = attribs_item
                    insertion_record_items.append((
                        open_insertion_record,
                        close_insertion_record,
                        attribs,
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
                #if not span_info.isolated:
                #    continue
                if span_info.local_color is not None:
                    local_color = span_info.local_color
                    local_color_stack.pop()
                else:
                    local_color = local_color_stack[-1] if local_color_stack else None
                insertion_record_items.append((
                    start_insertion_record,
                    stop_insertion_record,
                    span_info.attribs if span_info.attribs is not None else {},
                    span_info.isolated,
                    local_color
                    #span_boundary._attribs if isinstance(span_boundary, AttributedSpanBoundary) else {}
                ))



            #span = span_boundary._span
            #if isinstance(span_boundary, ProtectedSpanBoundary):
            #    protect_level += span_boundary._boundary_flag.value
            #    if not isinstance(span_boundary, CommandSpanBoundary):
            #        continue
            #    if span_boundary._boundary_flag == BoundaryFlag.START:
            #        continue
            #    command_replacement = span_boundary._command_info.replacement
            #    command_replacement_record = ReplacementRecord(
            #        span=span,
            #        unlabelled_replacement=command_replacement,
            #        labelled_replacement=command_replacement
            #    )
            #    command_flag = span_boundary._command_info.flag
            #    if command_flag == CommandFlag.OPEN:
            #        bracket_stack.append(next(bracket_counter))
            #        open_insertion_record = InsertionRecord(span._stop)
            #        replacement_records.append(command_replacement_record)
            #        replacement_records.append(open_insertion_record)
            #        open_command_stack.append((span_boundary, open_insertion_record))
            #    elif command_flag == CommandFlag.CLOSE:
            #        bracket_stack.pop()
            #        close_insertion_record = InsertionRecord(span._start)
            #        replacement_records.append(close_insertion_record)
            #        replacement_records.append(command_replacement_record)
            #        open_span_boundary, open_insertion_record = open_command_stack.pop()
            #        if (attribs := open_span_boundary._command_info.attribs) is not None:
            #            labelled_items.append((
            #                open_insertion_record,
            #                close_insertion_record,
            #                attribs
            #            ))
            #    else:
            #        replacement_records.append(command_replacement_record)
            #    continue

            #if span_boundary._boundary_flag == BoundaryFlag.START:
            #    start_insertion_record = InsertionRecord(span._start)
            #    replacement_records.append(start_insertion_record)
            #    start_stack.append((protect_level, span, tuple(bracket_stack), start_insertion_record))
            #elif span_boundary._boundary_flag == BoundaryFlag.STOP:
            #    stop_insertion_record = InsertionRecord(span._stop)
            #    replacement_records.append(stop_insertion_record)
            #    start_protect_level, start_span, start_bracket_stack, start_insertion_record = start_stack.pop()

            #    if not start_protect_level and not protect_level:
            #        assert start_span is span, \
            #            f"Partly overlapping substrings detected: '{string[start_span.as_slice()]}', '{string[span.as_slice()]}'"
            #        assert start_bracket_stack == tuple(bracket_stack), \
            #            f"Cannot handle substring: '{string[span.as_slice()]}'"
            #        labelled_items.append((
            #            start_insertion_record,
            #            stop_insertion_record,
            #            span_boundary._attribs if isinstance(span_boundary, AttributedSpanBoundary) else {}
            #        ))

        #replacement_records.append(global_stop_insertion_record)

        assert protect_level == 0
        assert not bracket_stack
        assert not open_command_stack
        assert not start_stack
        assert not local_color_stack

        label_counter = itertools.count()
        isolated_items: list[tuple[Span, ColorT | None]] = []
        for start_insertion_record, stop_insertion_record, attribs, isolated, local_color in insertion_record_items:
            if isolated:
                label = next(label_counter)
                isolated_items.append((
                    Span(start_insertion_record._span._stop, stop_insertion_record._span._start),
                    local_color
                ))
            else:
                label = None
            labelled_attribs = cls._convert_attribs_for_labelling(attribs, label)
            start_unlabelled_insertion, stop_unlabelled_insertion = cls._get_command_pair(attribs)
            start_labelled_insertion, stop_labelled_insertion = cls._get_command_pair(labelled_attribs)
            start_insertion_record.write_replacements(
                #label=label,
                #boundary_flag=BoundaryFlag.START,
                unlabelled_replacement=start_unlabelled_insertion,
                labelled_replacement=start_labelled_insertion
            )
            stop_insertion_record.write_replacements(
                #label=label,
                #boundary_flag=BoundaryFlag.STOP,
                unlabelled_replacement=stop_unlabelled_insertion,
                labelled_replacement=stop_labelled_insertion
            )
            #for insertion_record, boundary_flag in (
            #    (start_insertion_record, BoundaryFlag.START),
            #    (stop_insertion_record, BoundaryFlag.STOP)
            #):
            #    insertion_record.write_replacements(
            #        label=label,
            #        boundary_flag=boundary_flag,
            #        unlabelled_replacement=cls._get_command_string(
            #            label=None,
            #            boundary_flag=boundary_flag,
            #            attribs=attribs
            #        ),
            #        labelled_replacement=cls._get_command_string(
            #            label=label,
            #            boundary_flag=boundary_flag,
            #            attribs=attribs
            #        )
            #    )

        return tuple(isolated_items), tuple(replacement_records)
        #return tuple(isolated_spans), tuple(
        #    replacement_record for replacement_record in replacement_records
        #    if not isinstance(replacement_record, InsertionRecord) or replacement_record._activated
        #)

    @classmethod
    def _iter_spans_by_selector(
        cls: type[Self],
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
                    start, stop = match_obj.span()
                    if start < stop:
                        yield Span(start, stop)
            case slice(start=int(start), stop=int(stop)):
                l = len(string)
                start = min(start, l) if start >= 0 else max(start + l, 0)
                stop = min(stop, l) if stop >= 0 else max(stop + l, 0)
                if start < stop:
                    yield Span(start, stop)

    @classmethod
    @abstractmethod
    def _get_command_pair(
        cls: type[Self],
        attribs: dict[str, str]
    ) -> tuple[str, str]:
        pass

    @classmethod
    @abstractmethod
    def _convert_attribs_for_labelling(
        cls: type[Self],
        attribs: dict[str, str],
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

    #@classmethod
    #@abstractmethod
    #def _iter_command_info_pairs(
    #    cls: type[Self],
    #    string: str
    #) -> Iterator[tuple[CommandInfo, CommandInfo, dict[str, str], bool]]:
    #    pass


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
        selector: SelectorT
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
        selector: SelectorT,
        index: int = 0
    ) -> Mobject:
        return self._build_from_indices(self._get_indices_tuple_by_selector(selector)[index])

    def select_parts(
        self: Self,
        selector: SelectorT
    ) -> Mobject:
        return self._build_from_indices_tuple(self._get_indices_tuple_by_selector(selector))
