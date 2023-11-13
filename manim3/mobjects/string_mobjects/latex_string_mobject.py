from __future__ import annotations
import pathlib


import re
from abc import abstractmethod
from typing import (
    Iterator,
    Self
)

import attrs

from ...constants.custom_typing import ColorType
from ...toplevel.toplevel import Toplevel
from ...utils.color_utils import ColorUtils
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
    color: ColorType = attrs.field(factory=lambda: Toplevel.config.default_color)
    font_size: float = attrs.field(factory=lambda: Toplevel.config.latex_font_size)


class LatexStringMobjectKwargs(StringMobjectKwargs, total=False):
    color: ColorType
    font_size: float


class LatexStringMobjectIO[LatexStringMobjectInputT: LatexStringMobjectInput](StringMobjectIO[LatexStringMobjectInputT]):
    __slots__ = ()

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
    def _get_global_span_attribs(
        cls: type[Self],
        input_data: LatexStringMobjectInputT,
        temp_path: pathlib.Path
    ) -> dict[str, str]:
        global_span_attribs = {
            "color": ColorUtils.color_to_hex(input_data.color)
        }
        global_span_attribs.update(super()._get_global_span_attribs(input_data, temp_path))
        return global_span_attribs

    @classmethod
    def _get_command_pair(
        cls: type[Self],
        attribs: dict[str, str]
    ) -> tuple[str, str]:
        if (color_hex := attribs.get("color")) is None:
            return "", ""
        match_obj = re.fullmatch(r"#([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})", color_hex, flags=re.IGNORECASE)
        assert match_obj is not None
        return "{{" + f"\\color[RGB]{{{", ".join(
            str(int(match_obj.group(index), 16))
            for index in range(1, 4)
        )}}}", "}}"

    @classmethod
    def _convert_attribs_for_labelling(
        cls: type[Self],
        attribs: dict[str, str],
        label: int | None
    ) -> dict[str, str]:
        result = {
            key: value
            for key, value in attribs.items()
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
        for match_obj in pattern.finditer(string):
            if not match_obj.group("close"):
                if not match_obj.group("open"):
                    yield StandaloneCommandInfo(match_obj)
                    continue
                open_stack.append(match_obj.span())
                continue
            close_start, close_stop = match_obj.span()
            while True:
                if not open_stack:
                    raise ValueError("Missing '{' inserted")
                open_start, open_stop = open_stack.pop()
                width = min(open_stop - open_start, close_stop - close_start)
                assert (open_match_obj := pattern.fullmatch(string, pos=open_stop - width, endpos=open_stop)) is not None
                assert (close_match_obj := pattern.fullmatch(string, pos=close_start, endpos=close_start + width)) is not None
                yield BalancedCommandInfo(
                    attribs={},
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
