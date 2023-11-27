from __future__ import annotations


import pathlib
import re
from typing import (
    Iterator,
    Self
)

import attrs

from ...animatables.arrays.animatable_color import AnimatableColor
from ...constants.custom_typing import ColorType
from ...toplevel.toplevel import Toplevel
from .string_mobject import (
    BalancedCommandInfo,
    CommandInfo,
    StandaloneCommandInfo,
    StringMobjectIO,
    StringMobjectInput,
    StringMobjectKwargs
)


@attrs.frozen(kw_only=True)
class LatexStringMobjectInput(StringMobjectInput):
    color: ColorType = attrs.field(factory=lambda: Toplevel._get_config().default_color)


class LatexStringMobjectKwargs(StringMobjectKwargs, total=False):
    color: ColorType


class LatexStringMobjectIO[LatexStringMobjectInputT: LatexStringMobjectInput](StringMobjectIO[LatexStringMobjectInputT]):
    __slots__ = ()

    @classmethod
    def _get_global_span_attributes(
        cls: type[Self],
        input_data: LatexStringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        global_span_attributes = {
            "color": AnimatableColor._color_to_hex(input_data.color)
        }
        global_span_attributes.update(super()._get_global_span_attributes(input_data, temp_path))
        return global_span_attributes

    @classmethod
    def _get_command_pair(
        cls: type[Self],
        attributes: dict[str, str]
    ) -> tuple[str, str]:
        if (color_hex := attributes.get("color")) is None:
            return "", ""
        match = re.fullmatch(r"#([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})", color_hex, flags=re.IGNORECASE)
        assert match is not None
        return f"{{\\color[RGB]{{{", ".join(
            f"{int(match.group(index), 16)}"
            for index in range(1, 4)
        )}}}{{", f"}}}}"

    @classmethod
    def _convert_attributes_for_labelling(
        cls: type[Self],
        attributes: dict[str, str],
        label: int | None
    ) -> dict[str, str]:
        result = {
            key: value
            for key, value in attributes.items()
            if key != "color"
        }
        if label is not None:
            result["color"] = f"#{label:06x}"
        return result

    @classmethod
    def _iter_command_infos(
        cls: type[Self],
        string: str
    ) -> Iterator[CommandInfo]:
        pattern = re.compile(r"""
            (?P<command>\\(?:[a-zA-Z]+|.))
            |(?P<open>{+)
            |(?P<close>}+)
        """, flags=re.VERBOSE | re.DOTALL)
        open_stack: list[tuple[int, int]] = []
        for match in pattern.finditer(string):
            if not match.group("close"):
                if not match.group("open"):
                    yield StandaloneCommandInfo(match)
                    continue
                open_stack.append(match.span())
                continue
            close_start, close_stop = match.span()
            while True:
                if not open_stack:
                    raise ValueError("Missing '{' inserted")
                open_start, open_stop = open_stack.pop()
                width = min(open_stop - open_start, close_stop - close_start)
                assert (open_match_obj := pattern.fullmatch(string, pos=open_stop - width, endpos=open_stop)) is not None
                assert (close_match_obj := pattern.fullmatch(string, pos=close_start, endpos=close_start + width)) is not None
                yield BalancedCommandInfo(
                    attributes={},
                    isolated=width >= 2,
                    open_match_obj=open_match_obj,
                    close_match_obj=close_match_obj
                )
                open_stop -= width
                close_start += width
                if close_start < close_stop:
                    continue
                if open_start < open_stop:
                    open_stack.append((open_start, open_stop))
                break
        if open_stack:
            raise ValueError("Missing '}' inserted")
