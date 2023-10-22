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

from ...animatables.arrays.animatable_color import AnimatableColor
from ...animatables.geometries.shape import Shape
from ...constants.custom_typing import (
    ColorT,
    SelectorT
)
from ...toplevel.toplevel import Toplevel
from ...utils.color_utils import ColorUtils
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


class CommandFlag(Enum):
    OPEN = 1
    CLOSE = -1
    OTHER = 0


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


class SpanBoundary:
    __slots__ = (
        "_span",
        "_boundary_flag"
    )

    def __init__(
        self,
        span: Span,
        boundary_flag: BoundaryFlag
    ) -> None:
        super().__init__()
        self._span: Span = span
        self._boundary_flag: BoundaryFlag = boundary_flag

    def get_sorting_key(
        self: Self
    ) -> tuple[int, int, int]:
        span = self._span
        boundary_flag = self._boundary_flag
        flag_value = boundary_flag.value
        index = span.get_boundary_index(boundary_flag)
        paired_index = span.get_boundary_index(boundary_flag.negate())
        # All spans have nonzero widths.
        return (
            index,
            flag_value,
            -paired_index
            #flag_value * item_index
        )


class IsolatedSpanBoundary(SpanBoundary):
    __slots__ = ()


class AttributedSpanBoundary(IsolatedSpanBoundary):
    __slots__ = ("_attrs",)

    def __init__(
        self,
        span: Span,
        boundary_flag: BoundaryFlag,
        attrs: dict[str, str]
    ) -> None:
        super().__init__(span, boundary_flag)
        self._attrs: dict[str, str] = attrs


class ProtectedSpanBoundary(SpanBoundary):
    __slots__ = ()


class CommandSpanBoundary(ProtectedSpanBoundary):
    __slots__ = ("_match_obj",)

    def __init__(
        self,
        match_obj: re.Match[str],
        boundary_flag: BoundaryFlag
    ) -> None:
        super().__init__(Span(*match_obj.span()), boundary_flag)
        self._match_obj: re.Match[str] = match_obj


class ReplacementRecord:
    __slots__ = (
        "_span",
        "_unlabelled_replacement",
        "_labelled_replacement",
        "_matching_replacement"
    )

    def __init__(
        self,
        span: Span,
        unlabelled_replacement: str,
        labelled_replacement: str,
        matching_replacement: str
    ) -> None:
        super().__init__()
        self._span: Span = span
        self._unlabelled_replacement: str = unlabelled_replacement
        self._labelled_replacement: str = labelled_replacement
        self._matching_replacement: str = matching_replacement


class InsertionRecord(ReplacementRecord):
    __slots__ = (
        #"_index",
        "_label",
        "_boundary_flag",
        "_activated"
    )

    def __init__(
        self,
        index: int
    ) -> None:
        super().__init__(Span(index, index), "", "", "")
        #self._index: int = index
        self._label: int = NotImplemented
        self._boundary_flag: BoundaryFlag = NotImplemented
        self._activated = False

    def write(
        self: Self,
        *,
        label: int,
        boundary_flag: BoundaryFlag,
        unlabelled_replacement: str,
        labelled_replacement: str,
        matching_replacement: str
    ) -> None:
        self._label = label
        self._boundary_flag = boundary_flag
        self._unlabelled_replacement = unlabelled_replacement
        self._labelled_replacement = labelled_replacement
        self._matching_replacement = matching_replacement
        self._activated = True


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class LabelledInsertionItem:
#    label: int
#    boundary_flag: BoundaryFlag
#    attrs: dict[str, str]
#    index: int

#    @property
#    def span(
#        self: Self
#    ) -> Span:
#        index = self.index
#        return Span(index, index)


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class StringMobjectSettings:
#    pass


class StringMobjectKwargs(TypedDict, total=False):
    #string: Required[str]
    #isolate: list[Span]
    #protect: list[Span]
    isolate: list[SelectorT]
    protect: list[SelectorT]
    global_color: ColorT
    local_color: dict[SelectorT, ColorT]
    global_attrs: dict[str, str]
    local_attrs: dict[SelectorT, dict[str, str]]
    concatenate: bool


@attrs.frozen(kw_only=True)
class StringMobjectInput(MobjectInput):
    string: str
    #isolate: list[Span]
    #protect: list[Span]
    isolate: list[SelectorT] = attrs.field(factory=list)
    protect: list[SelectorT] = attrs.field(factory=list)
    global_color: ColorT = attrs.field(factory=lambda: Toplevel.config.default_color)
    local_color: dict[SelectorT, ColorT] = attrs.field(factory=dict)
    global_attrs: dict[str, str] = attrs.field(factory=dict)
    local_attrs: dict[SelectorT, dict[str, str]] = attrs.field(factory=dict)
    concatenate: bool = False
    #settings: StringMobjectSettingsT


@attrs.frozen(kw_only=True)
class StringMobjectOutput(MobjectOutput):
    shape_mobjects: tuple[ShapeMobject, ...]
    string: str
    spans: tuple[Span, ...]
    labelled_part_items: tuple[tuple[str, tuple[int, ...]], ...]
    group_part_items: tuple[tuple[str, tuple[int, ...]], ...]


class StringMobjectJSON(MobjectJSON):
    shape_mobjects: tuple[ShapeMobjectJSON, ...]
    string: str
    spans: tuple[tuple[int, ...], ...]
    labelled_part_strings: tuple[str, ...]
    labelled_part_indices: tuple[tuple[int, ...], ...]
    group_part_strings: tuple[str, ...]
    group_part_indices: tuple[tuple[int, ...], ...]


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
        string = input_data.string
        global_span_attrs = cls._get_global_span_attrs(input_data, temp_path)
        #global_span_attrs.update(input_data.global_attrs)
        local_span_attrs = cls._get_local_span_attrs(input_data, temp_path)
        global_span_color = input_data.global_color
        local_span_color = {
            span: color
            for selector, color in input_data.local_color.items()
            for span in cls._iter_spans_by_selector(selector, string)
        }

        label_to_span_dict, replacement_records = cls._get_label_to_span_dict_and_replacement_records(
            string=string,
            isolated_spans=itertools.chain(
                *(
                    cls._iter_spans_by_selector(selector, string)
                    for selector in input_data.isolate
                ),
                local_span_color
            ),
            protected_spans=itertools.chain.from_iterable(
                cls._iter_spans_by_selector(selector, string)
                for selector in input_data.protect
            ),
            global_span_attrs=global_span_attrs,
            local_span_attrs=local_span_attrs
        )
        original_pieces = tuple(
            string[start:stop]
            for start, stop in zip(
                (replacement_record._span._stop for replacement_record in replacement_records[:-1]),
                (replacement_record._span._start for replacement_record in replacement_records[1:]),
                strict=True
            )
        )
        labels, shape_mobjects = cls._get_labels_and_shape_mobjects(
            unlabelled_content=cls._replace_string(
                original_pieces=original_pieces,
                replacement_pieces=tuple(
                    replacement_record._unlabelled_replacement
                    for replacement_record in replacement_records
                ),
                start_index=0,
                stop_index=len(replacement_records)
            ),
            #labelled_content=cls._get_content(
            #    original_pieces=original_pieces,
            #    replacement_items=replacement_items,
            #    is_labelled=True
            #),
            labelled_content=cls._replace_string(
                original_pieces=original_pieces,
                replacement_pieces=tuple(
                    replacement_record._labelled_replacement
                    for replacement_record in replacement_records
                ),
                start_index=0,
                stop_index=len(replacement_records)
            ),
            concatenate=input_data.concatenate,
            requires_labelling=len(label_to_span_dict) > 1,
            input_data=input_data,
            #input_data=input_data,
            temp_path=temp_path
        )

        spans = tuple(label_to_span_dict[label] for label in labels)
        for shape_mobject in shape_mobjects:
            shape_mobject._color_ = AnimatableColor._convert_input(global_span_color)
        for span, color in local_span_color.items():
            for index in cls._get_indices_by_span(span, spans):
                shape_mobjects[index]._color_ = AnimatableColor._convert_input(color)

        return StringMobjectOutput(
            shape_mobjects=shape_mobjects,
            string=string,
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
                replacement_records=replacement_records
                #replaced_items=replaced_items
            )
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
            spans=tuple((span._start, span._stop) for span in output_data.spans),
            labelled_part_strings=tuple(string for string, _ in output_data.labelled_part_items),
            labelled_part_indices=tuple(indices for _, indices in output_data.labelled_part_items),
            group_part_strings=tuple(string for string, _ in output_data.group_part_items),
            group_part_indices=tuple(indices for _, indices in output_data.group_part_items)
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
            spans=tuple(Span(*span_values) for span_values in json_data["spans"]),
            labelled_part_items=tuple(zip(
                json_data["labelled_part_strings"],
                json_data["labelled_part_indices"],
                strict=True
            )),
            group_part_items=tuple(zip(
                json_data["group_part_strings"],
                json_data["group_part_indices"],
                strict=True
            ))
        )

    @classmethod
    def _get_labels_and_shape_mobjects(
        cls: type[Self],
        unlabelled_content: str,
        labelled_content: str,
        concatenate: bool,
        requires_labelling: bool,
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> tuple[tuple[int, ...], tuple[ShapeMobject, ...]]:
        unlabelled_shape_mobjects = cls._get_shape_mobjects(unlabelled_content, input_data, temp_path)
        if concatenate:
            return (0,), (ShapeMobject(Shape._concatenate(tuple(
                shape_mobject._shape_ for shape_mobject in unlabelled_shape_mobjects
            ))),)

        if not requires_labelling or not unlabelled_shape_mobjects:
            return (0,) * len(unlabelled_shape_mobjects), unlabelled_shape_mobjects

        labelled_shape_mobjects = cls._get_shape_mobjects(labelled_content, input_data, temp_path)
        assert len(unlabelled_shape_mobjects) == len(labelled_shape_mobjects)

        unlabelled_radii = ShapeMobject().add(*unlabelled_shape_mobjects).box.get_radii()
        labelled_radii = ShapeMobject().add(*labelled_shape_mobjects).box.get_radii()
        scale_factor = labelled_radii / unlabelled_radii
        #ShapeMobject().add(*labelled_shape_mobjects).match_box(
        #    ShapeMobject().add(*unlabelled_shape_mobjects)
        #)
        distance_matrix = scipy.spatial.distance.cdist(
            [shape.box.get() for shape in unlabelled_shape_mobjects],
            [shape.box.get() * scale_factor for shape in labelled_shape_mobjects]
        )
        unlabelled_indices, labelled_indices = scipy.optimize.linear_sum_assignment(distance_matrix)
        return tuple(
            int(ColorUtils.color_to_hex(labelled_shape_mobjects[labelled_index]._color_)[1:], 16)
            for labelled_index in labelled_indices
        ), tuple(
            unlabelled_shape_mobjects[unlabelled_index]
            for unlabelled_index in unlabelled_indices
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
    def _get_global_span_attrs(
        cls: type[Self],
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        global_span_attrs: dict[str, str] = {}
        global_span_attrs.update(input_data.global_attrs)
        return global_span_attrs

    @classmethod
    def _get_local_span_attrs(
        cls: type[Self],
        input_data: StringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[Span, dict[str, str]]:
        local_span_attrs: dict[Span, dict[str, str]] = {}
        for selector, local_attrs in input_data.local_attrs.items():
            for span in cls._iter_spans_by_selector(selector, input_data.string):
                local_span_attrs.setdefault(span, {}).update(local_attrs)
        return local_span_attrs

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
    def _get_label_to_span_dict_and_replacement_records(
        cls: type[Self],
        string: str,
        isolated_spans: Iterator[Span],
        protected_spans: Iterator[Span],
        global_span_attrs: dict[str, str],
        local_span_attrs: dict[Span, dict[str, str]]
        #isolate: Iterable[SelectorT],
        #protect: Iterable[SelectorT],
        #global_attrs: dict[str, str],
        #local_attrs: dict[Span, dict[str, str]]
    ) -> tuple[dict[int, Span], tuple[ReplacementRecord, ...]]:

        #def get_key(
        #    index_item: tuple[AttributedItem | IsolatedItem | ProtectedItem | CommandItem, BoundaryFlag]
        #) -> tuple[int, int, int]:
        #    span_item, boundary_flag = index_item
        #    flag_value = boundary_flag.value
        #    index = span_item.span.get_boundary_index(boundary_flag)
        #    paired_index = span_item.span.get_boundary_index(boundary_flag.negate())
        #    # All spans have nonzero widths.
        #    return (
        #        index,
        #        flag_value,
        #        -paired_index
        #        #flag_value * item_index
        #    )

        def write_labelled_item(
            start_insertion_record: InsertionRecord,
            stop_insertion_record: InsertionRecord,
            attrs: dict[str, str],
            label: int,
            label_to_span_dict: dict[int, Span]
        ) -> None:
            label_to_span_dict[label] = Span(start_insertion_record._span._stop, stop_insertion_record._span._start)
            for insertion_record, boundary_flag in (
                (start_insertion_record, BoundaryFlag.START),
                (stop_insertion_record, BoundaryFlag.STOP)
            ):
                insertion_record.write(
                    label=label,
                    boundary_flag=boundary_flag,
                    unlabelled_replacement=cls._get_command_string(
                        label=None,
                        boundary_flag=boundary_flag,
                        attrs=attrs
                    ),
                    labelled_replacement=cls._get_command_string(
                        label=label,
                        boundary_flag=boundary_flag,
                        attrs=attrs
                    ),
                    matching_replacement=""
                )

        #def add_labelled_item(
        #    label_to_span_dict: dict[int, Span],
        #    replaced_items: list[CommandItem | LabelledInsertionItem],
        #    label: int,
        #    span: Span,
        #    attrs: dict[str, str],
        #    insert_index: int
        #) -> None:
        #    label_to_span_dict[label] = span
        #    replaced_items.insert(insert_index, LabelledInsertionItem(
        #        label=label,
        #        boundary_flag=BoundaryFlag.START,
        #        attrs=attrs,
        #        index=span._start
        #    ))
        #    replaced_items.append(LabelledInsertionItem(
        #        label=label,
        #        boundary_flag=BoundaryFlag.STOP,
        #        attrs=attrs,
        #        index=span._stop
        #    ))

        span_boundaries = sorted(itertools.chain.from_iterable(
            itertools.islice(itertools.chain(
                (
                    AttributedSpanBoundary(span=span, boundary_flag=boundary_flag, attrs=attrs)
                    for span, attrs in local_span_attrs.items()
                ),
                (
                    IsolatedSpanBoundary(span=span, boundary_flag=boundary_flag)
                    for span in isolated_spans
                ),
                (
                    ProtectedSpanBoundary(span=span, boundary_flag=boundary_flag)
                    for span in protected_spans
                ),
                (
                    CommandSpanBoundary(match_obj=match_obj, boundary_flag=boundary_flag)
                    for match_obj in cls._iter_command_matches(string)
                )
            ), None, None, boundary_flag.value)
            for boundary_flag in (BoundaryFlag.STOP, BoundaryFlag.START)
        ), key=SpanBoundary.get_sorting_key)
        #index_items = sorted((
        #    (span_item, boundary_flag)
        #    for boundary_flag in (BoundaryFlag.STOP, BoundaryFlag.START)
        #    for span_item in span_items[::boundary_flag.value]
        #), key=get_key)

        label_to_span_dict: dict[int, Span] = {}
        replacement_records: list[ReplacementRecord] = []
        #replaced_items: list[CommandItem | LabelledInsertionItem] = []
        label_counter = itertools.count(start=1)
        bracket_counter = itertools.count()
        protect_level = 0
        bracket_stack: list[int] = []
        open_command_stack: list[tuple[CommandSpanBoundary, InsertionRecord]] = []
        start_stack: list[tuple[int, Span, tuple[int, ...], InsertionRecord]] = []

        global_start_insertion_record = InsertionRecord(0)
        replacement_records.append(global_start_insertion_record)
        #global_label = next(label_counter)
        #label_to_span_dict[global_label] = Span(global_start_insertion_record._index, global_stop_insertion_record._index)

        for span_boundary in span_boundaries:
            if isinstance(span_boundary, ProtectedSpanBoundary):
                protect_level += span_boundary._boundary_flag.value
                if not isinstance(span_boundary, CommandSpanBoundary):
                    continue
                if span_boundary._boundary_flag == BoundaryFlag.START:
                    continue
                content_replacement = cls._replace_for_content(match_obj=span_boundary._match_obj)
                matching_replacement = cls._replace_for_matching(match_obj=span_boundary._match_obj)
                command_replacement_record = ReplacementRecord(
                    span=span_boundary._span,
                    unlabelled_replacement=content_replacement,
                    labelled_replacement=content_replacement,
                    matching_replacement=matching_replacement
                )
                command_flag = cls._get_command_flag(match_obj=span_boundary._match_obj)
                if command_flag == CommandFlag.OPEN:
                    bracket_stack.append(next(bracket_counter))
                    open_insertion_record = InsertionRecord(span_boundary._span._stop)
                    replacement_records.append(command_replacement_record)
                    replacement_records.append(open_insertion_record)
                    #open_command_stack.append((len(replaced_items), command_item))
                    open_command_stack.append((span_boundary, open_insertion_record))
                elif command_flag == CommandFlag.CLOSE:
                    #insert_index, open_command_item = open_command_stack.pop()
                    #bracket_stack.pop()
                    close_insertion_record = InsertionRecord(span_boundary._span._start)
                    replacement_records.append(close_insertion_record)
                    replacement_records.append(command_replacement_record)
                    open_span_boundary, open_insertion_record = open_command_stack.pop()

                    if (attrs := cls._get_attrs_from_command_pair(
                        open_command=open_span_boundary._match_obj,
                        close_command=span_boundary._match_obj
                    )) is not None:
                        write_labelled_item(
                            start_insertion_record=open_insertion_record,
                            stop_insertion_record=close_insertion_record,
                            attrs=attrs,
                            label=next(label_counter),
                            label_to_span_dict=label_to_span_dict
                        )
                        #label = next(label_counter)
                        #label_to_span_dict[label] = Span(open_insertion_record._index, close_insertion_record._index)
                        #for insertion_record, boundary_flag in (
                        #    (open_insertion_record, BoundaryFlag.START),
                        #    (close_insertion_record, BoundaryFlag.STOP)
                        #):
                        #    insertion_record.write(
                        #        unlabelled_replacement=cls._get_command_string(
                        #            label=None,
                        #            boundary_flag=boundary_flag,
                        #            attrs=attrs
                        #        ),
                        #        labelled_replacement=cls._get_command_string(
                        #            label=label,
                        #            boundary_flag=boundary_flag,
                        #            attrs=attrs
                        #        ),
                        #        matching_replacement=""
                        #    )
                        #add_labelled_item(
                        #    label_to_span_dict=label_to_span_dict,
                        #    replaced_items=replaced_items,
                        #    label=next(label_counter),
                        #    span=Span(open_command_item.span._stop, command_item.span._start),
                        #    attrs=attrs,
                        #    insert_index=insert_index
                        #)
                    #replacement_records.append(command_replacement_record)
                else:
                    replacement_records.append(command_replacement_record)
                continue

            if span_boundary._boundary_flag == BoundaryFlag.START:
                start_insertion_record = InsertionRecord(span_boundary._span._start)
                replacement_records.append(start_insertion_record)
                start_stack.append((
                    protect_level, span_boundary._span, tuple(bracket_stack), start_insertion_record
                ))
            elif span_boundary._boundary_flag == BoundaryFlag.STOP:
                stop_insertion_record = InsertionRecord(span_boundary._span._stop)
                replacement_records.append(stop_insertion_record)
                start_protect_level, start_span, start_bracket_stack, start_insertion_record = start_stack.pop()

                if not start_protect_level and not protect_level:
                    span = span_boundary._span
                    assert start_span is span, \
                        f"Partly overlapping substrings detected: '{string[start_span.as_slice()]}', '{string[span.as_slice()]}'"
                    assert start_bracket_stack == bracket_stack, \
                        f"Cannot handle substring: '{string[span.as_slice()]}'"
                    write_labelled_item(
                        start_insertion_record=start_insertion_record,
                        stop_insertion_record=stop_insertion_record,
                        attrs=span_boundary._attrs if isinstance(span_boundary, AttributedSpanBoundary) else {},
                        label=next(label_counter),
                        label_to_span_dict=label_to_span_dict
                    )
                    #attrs = span_boundary._attrs if isinstance(span_boundary, AttributedSpanBoundary) else {}
                    #label = next(label_counter)
                    #label_to_span_dict[label] = Span(start_insertion_record._index, stop_insertion_record._index)
                    #for insertion_record, boundary_flag in (
                    #    (start_insertion_record, BoundaryFlag.START),
                    #    (stop_insertion_record, BoundaryFlag.STOP)
                    #):
                    #    insertion_record.write(
                    #        unlabelled_replacement=cls._get_command_string(
                    #            label=None,
                    #            boundary_flag=boundary_flag,
                    #            attrs=attrs
                    #        ),
                    #        labelled_replacement=cls._get_command_string(
                    #            label=label,
                    #            boundary_flag=boundary_flag,
                    #            attrs=attrs
                    #        ),
                    #        matching_replacement=""
                    #    )
            #add_labelled_item(
            #    label_to_span_dict=label_to_span_dict,
            #    replaced_items=replaced_items,
            #    label=next(label_counter),
            #    span=span,
            #    attrs=span_item.attrs if isinstance(span_item, AttributedItem) else {},
            #    insert_index=insert_index
            #)
        #add_labelled_item(
        #    label_to_span_dict=label_to_span_dict,
        #    replaced_items=replaced_items,
        #    label=0,
        #    span=Span(0, len(string)),
        #    attrs=global_span_attrs,
        #    insert_index=0
        #)
        global_stop_insertion_record = InsertionRecord(len(string))
        replacement_records.append(global_stop_insertion_record)
        write_labelled_item(
            start_insertion_record=global_start_insertion_record,
            stop_insertion_record=global_stop_insertion_record,
            attrs=global_span_attrs,
            label=0,
            label_to_span_dict=label_to_span_dict
        )

        assert protect_level == 0
        assert not bracket_stack
        assert not open_command_stack
        assert not start_stack
        return label_to_span_dict, tuple(
            replacement_record for replacement_record in replacement_records
            if not isinstance(replacement_record, InsertionRecord) or replacement_record._activated
        )

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

    #@classmethod
    #def _iter_spans_by_selectors(
    #    cls: type[Self],
    #    selectors: Iterable[SelectorT],
    #    string: str
    #) -> Iterator[Span]:
    #    for selector in selectors:
    #        yield from cls._iter_spans_by_selector(selector, string)

    #@classmethod
    #def _get_original_pieces(
    #    cls: type[Self],
    #    replaced_items: list[CommandItem | LabelledInsertionItem],
    #    string: str
    #) -> list[str]:
    #    replaced_spans = [replaced_item.span for replaced_item in replaced_items]
    #    return [
    #        string[start:stop]
    #        for start, stop in zip(
    #            [interval_span._stop for interval_span in replaced_spans[:-1]],
    #            [interval_span._start for interval_span in replaced_spans[1:]],
    #            strict=True
    #        )
    #    ]

    @classmethod
    def _replace_string(
        cls: type[Self],
        original_pieces: tuple[str, ...],
        replacement_pieces: tuple[str, ...],
        start_index: int,
        stop_index: int
    ) -> str:
        return "".join(itertools.chain.from_iterable(zip(
            ("", *original_pieces[start_index:stop_index - 1]),
            replacement_pieces[start_index:stop_index],
            strict=True
        )))

    #@classmethod
    #def _get_content(
    #    cls: type[Self],
    #    original_pieces: list[str],
    #    replaced_items: list[CommandItem | LabelledInsertionItem],
    #    is_labelled: bool
    #) -> str:
    #    content_replaced_pieces = [
    #        cls._replace_for_content(match_obj=replaced_item.match_obj)
    #        if isinstance(replaced_item, CommandItem)
    #        else cls._get_command_string(
    #            label=replaced_item.label if is_labelled else None,
    #            boundary_flag=replaced_item.boundary_flag,
    #            attrs=replaced_item.attrs
    #        )
    #        for replaced_item in replaced_items
    #    ]
    #    return cls._replace_string(
    #        original_pieces=original_pieces,
    #        replaced_pieces=content_replaced_pieces,
    #        start_index=0,
    #        stop_index=len(content_replaced_pieces)
    #    )

    @classmethod
    def _get_indices_by_span(
        cls: type[Self],
        specified_span: Span,
        spans: tuple[Span, ...]
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index, span in enumerate(spans)
            if specified_span.contains(span)
        )

    @classmethod
    def _get_labelled_part_items(
        cls: type[Self],
        string: str,
        spans: tuple[Span, ...],
        label_to_span_dict: dict[int, Span]
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(
            (string[span.as_slice()], cls._get_indices_by_span(span, spans))
            for span in label_to_span_dict.values()
        )

    @classmethod
    def _get_group_part_items(
        cls: type[Self],
        labels: tuple[int, ...],
        label_to_span_dict: dict[int, Span],
        original_pieces: tuple[str, ...],
        replacement_records: tuple[ReplacementRecord, ...]
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:

        def iter_boundary_item_pairs(
            label_iterator: Iterator[int],
            label_to_span_dict: dict[int, Span]
        ) -> Iterator[tuple[tuple[int, BoundaryFlag], tuple[int, BoundaryFlag]]]:
            prev_label = next(label_iterator)
            prev_span = label_to_span_dict[prev_label]
            start_item = (prev_label, BoundaryFlag.START)
            for next_label in label_iterator:
                next_span = label_to_span_dict[next_label]
                prev_stop = (prev_label, BoundaryFlag.STOP)
                next_start = (next_label, BoundaryFlag.START)
                stop_item = next_start if prev_span.contains(next_span) else prev_stop
                yield (start_item, stop_item)
                start_item = prev_stop if next_span.contains(prev_span) else next_start
                prev_label = next_label
                prev_span = next_span
            stop_item = (prev_label, BoundaryFlag.STOP)
            yield (start_item, stop_item)

        label_groupers = list(itertools.groupby(
            enumerate(labels),
            key=lambda label_item: label_item[1]
        ))
        label_boundary_to_index_dict = {
            (replacement_record._label, replacement_record._boundary_flag): index
            for index, replacement_record in enumerate(replacement_records)
            if isinstance(replacement_record, InsertionRecord)
        }
        matching_replacement_pieces = tuple(
            replacement_record._matching_replacement
            for replacement_record in replacement_records
        )

        #matching_replaced_pieces = [
        #    cls._replace_for_matching(match_obj=replaced_item.match_obj)
        #    if isinstance(replaced_item, CommandItem)
        #    else ""
        #    for replaced_item in replaced_items
        #]
        return tuple(
            (
                re.sub(r"\s+", "", cls._replace_string(
                    original_pieces=original_pieces,
                    replacement_pieces=matching_replacement_pieces,
                    start_index=start_index,
                    stop_index=stop_index
                )),
                tuple(label_item[0] for label_item in grouper)
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
                (start_index := label_boundary_to_index_dict[start_item])
                < (stop_index := label_boundary_to_index_dict[stop_item])
            )
        )

    @classmethod
    @abstractmethod
    def _iter_command_matches(
        cls: type[Self],
        string: str
    ) -> Iterator[re.Match[str]]:
        pass

    @classmethod
    @abstractmethod
    def _get_command_flag(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> CommandFlag:
        pass

    @classmethod
    @abstractmethod
    def _replace_for_content(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        pass

    @classmethod
    @abstractmethod
    def _replace_for_matching(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        pass

    @classmethod
    @abstractmethod
    def _get_attrs_from_command_pair(
        cls: type[Self],
        open_command: re.Match[str],
        close_command: re.Match[str]
    ) -> dict[str, str] | None:
        pass

    @classmethod
    @abstractmethod
    def _get_command_string(
        cls: type[Self],
        label: int | None,
        boundary_flag: BoundaryFlag,
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
        "_shape_mobjects",
        "_string",
        "_spans",
        "_labelled_part_items",
        "_group_part_items"
    )

    #_input_dataclass: ClassVar[type[StringMobjectInput]] = StringMobjectInput
    #_io_cls: ClassVar[type[StringMobjectIO]] = StringMobjectIO

    def __init__(
        self: Self,
        #string: str,
        #**kwargs: Unpack[StringMobjectKwargs]
        #string: str,
        output_data: StringMobjectOutput
    ) -> None:
        super().__init__()

        #cls = type(self)
        #input_data = StringMobjectInput(
        #    string=string,
        #    **{
        #        key: value
        #        for key in StringMobjectKwargs.__optional_keys__
        #        if (value := kwargs.pop(key, None)) is not None
        #    },
        #    settings=cls._settings_dataclass(**kwargs)
        #)
        #output_data = cls._io_cls.get(input_data)
        shape_mobjects = output_data.shape_mobjects
        self._shape_mobjects: tuple[ShapeMobject, ...] = shape_mobjects
        self._string: str = output_data.string
        self._spans: tuple[Span, ...] = output_data.spans
        self._labelled_part_items: tuple[tuple[str, tuple[int, ...]], ...] = output_data.labelled_part_items
        self._group_part_items: tuple[tuple[str, tuple[int, ...]], ...] = output_data.group_part_items
        self.add(*shape_mobjects)

    #@classmethod
    #@property
    #@abstractmethod
    #def _io_cls(
    #    cls: type[Self]
    #) -> type[StringMobjectIO[StringMobjectInputDataT]]:
    #    pass

    #@classmethod
    ##@abstractmethod
    #def _collect_input(
    #    cls: type[Self],
    #    **kwargs: StringMobjectUnpackedKwargsT
    #) -> StringMobjectInputDataT:
    #    StringMobjectInputData(

    #    )
    #    pass

    def _build_from_indices(
        self: Self,
        indices: tuple[int, ...]
    ) -> ShapeMobject:
        return ShapeMobject().add(*(
            self._shape_mobjects[index]
            for index in indices
        ))

    def _build_from_indices_tuple(
        self: Self,
        indices_tuple: tuple[tuple[int, ...], ...]
    ) -> ShapeMobject:
        return ShapeMobject().add(*(
            self._build_from_indices(indices)
            for indices in indices_tuple
        ))

    def select_part(
        self: Self,
        selector: SelectorT,
        index: int = 0
    ) -> ShapeMobject:
        return self._build_from_indices([
            StringMobjectIO._get_indices_by_span(specified_span, self._spans)
            for specified_span in StringMobjectIO._iter_spans_by_selector(selector, self._string)
        ][index])

    def select_parts(
        self: Self,
        selector: SelectorT
    ) -> ShapeMobject:
        return self._build_from_indices_tuple(tuple(
            StringMobjectIO._get_indices_by_span(specified_span, self._spans)
            for specified_span in StringMobjectIO._iter_spans_by_selector(selector, self._string)
        ))
