from __future__ import annotations


import re
from abc import abstractmethod
from typing import (
    Iterator,
    Self
)

import attrs

from ...toplevel.toplevel import Toplevel
from .string_mobject import (
    BoundaryFlag,
    CommandFlag,
    #StringMobject,
    StringMobjectIO,
    StringMobjectInput,
    StringMobjectKwargs
)


@attrs.frozen(kw_only=True)
class LatexStringMobjectInput(StringMobjectInput):
    font_size: float = attrs.field(factory=lambda: Toplevel.config.latex_font_size)
    #local_spans: list[Span]


class LatexStringMobjectKwargs(StringMobjectKwargs, total=False):
    font_size: float


class LatexStringMobjectIO[LatexStringMobjectInputT: LatexStringMobjectInput](StringMobjectIO[LatexStringMobjectInputT]):
    __slots__ = ()

    #@classmethod
    #def _get_global_attrs(
    #    cls: type[Self],
    #    input_data: LatexStringMobjectInputT,
    #    temp_path: pathlib.Path
    #) -> dict[str, str]:
    #    return {}

    #@classmethod
    #def _get_local_attrs(
    #    cls: type[Self],
    #    input_data: LatexStringMobjectInputT,
    #    temp_path: pathlib.Path
    #) -> dict[Span, dict[str, str]]:
    #    local_spans = input_data.local_spans
    #    return {span: {} for span in local_spans}

    @classmethod
    def _get_svg_frame_scale(
        cls: type[Self],
        input_data: LatexStringMobjectInput
    ) -> float:
        return cls._scale_factor_per_font_point * input_data.font_size

    @classmethod
    @property
    @abstractmethod
    def _scale_factor_per_font_point(
        cls: type[Self]
    ) -> float:
        pass

    @classmethod
    def _iter_command_matches(
        cls: type[Self],
        string: str
    ) -> Iterator[re.Match[str]]:
        # Lump together adjacent brace pairs.
        pattern = re.compile(r"""
            (?P<command>\\(?:[a-zA-Z]+|.))
            |(?P<open>{+)
            |(?P<close>}+)
        """, flags=re.VERBOSE | re.DOTALL)
        open_stack: list[tuple[int, int]] = []
        for match_obj in pattern.finditer(string):
            if not match_obj.group("close"):
                if not match_obj.group("open"):
                    yield match_obj
                    continue
                open_stack.append(match_obj.span())
                continue
            close_start, close_stop = match_obj.span()
            while True:
                if not open_stack:
                    raise ValueError("Missing '{' inserted")
                open_start, open_stop = open_stack.pop()
                n = min(open_stop - open_start, close_stop - close_start)
                assert (open_match_obj := pattern.fullmatch(
                    string, pos=open_stop - n, endpos=open_stop
                )) is not None
                yield open_match_obj
                assert (close_match_obj := pattern.fullmatch(
                    string, pos=close_start, endpos=close_start + n
                )) is not None
                yield close_match_obj
                close_start += n
                if close_start < close_stop:
                    continue
                open_stop -= n
                if open_start < open_stop:
                    open_stack.append((open_start, open_stop))
                break
        if open_stack:
            raise ValueError("Missing '}' inserted")

    @classmethod
    def _get_command_flag(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> CommandFlag:
        if match_obj.group("open"):
            return CommandFlag.OPEN
        if match_obj.group("close"):
            return CommandFlag.CLOSE
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        return match_obj.group()

    @classmethod
    def _replace_for_matching(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("command"):
            return match_obj.group()
        return ""

    @classmethod
    def _get_attrs_from_command_pair(
        cls: type[Self],
        open_command: re.Match[str],
        close_command: re.Match[str]
    ) -> dict[str, str] | None:
        if len(open_command.group()) >= 2:
            return {}
        return None

    @classmethod
    def _get_command_string(
        cls: type[Self],
        label: int | None,
        boundary_flag: BoundaryFlag,
        attrs: dict[str, str]
    ) -> str:
        if label is None:
            return ""
        if boundary_flag == BoundaryFlag.STOP:
            return "}}"
        rg, b = divmod(label, 256)
        r, g = divmod(rg, 256)
        color_command = f"\\color[RGB]{{{r}, {g}, {b}}}"
        return "{{" + color_command


#class LatexStringMobject(StringMobject):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        string: str,
#        **kwargs: Unpack[LatexStringMobjectKwargs]
#    ) -> None:
#        super().__init__(string, **kwargs)


#class LatexStringMobject(StringMobject):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        string: str,
#        *,
#        isolate: Iterable[SelectorT] = (),
#        protect: Iterable[SelectorT] = (),
#        color: ColorT | None = None,
#        font_size: float | None = None,
#        local_colors: dict[SelectorT, ColorT] | None = None,
#        **kwargs
#    ) -> None:
#        config = Toplevel.config
#        if color is None:
#            color = config.latex_color
#        if font_size is None:
#            font_size = config.latex_font_size
#        if local_colors is None:
#            local_colors = {}

#        cls = type(self)
#        super().__init__(
#            string=string,
#            isolate=cls._get_spans_by_selectors(isolate, string),
#            protect=cls._get_spans_by_selectors(protect, string),
#            font_size=font_size,
#            local_spans=cls._get_spans_by_selectors(local_colors, string),
#            **kwargs
#        )

#        self.set(color=color)
#        for selector, local_color in local_colors.items():
#            self.select_parts(selector).set(color=local_color)
