from __future__ import annotations


import re
from typing import (
    Iterator,
    Self,
    Unpack
)

import attrs

from .pango_string_mobject import (
    PangoStringMobjectIO,
    PangoStringMobjectInput,
    PangoStringMobjectKwargs
)
from .string_mobject import (
    BalancedCommandInfo,
    CommandInfo,
    StandaloneCommandInfo,
    StringMobject
)


@attrs.frozen(kw_only=True)
class MarkupInput(PangoStringMobjectInput):
    pass


class MarkupKwargs(PangoStringMobjectKwargs, total=False):
    pass


class MarkupIO[MarkupInputT: MarkupInput](PangoStringMobjectIO[MarkupInputT]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(
        cls: type[Self]
    ) -> str:
        return "markup"

    @classmethod
    def _iter_command_infos(
        cls: type[Self],
        string: str
    ) -> Iterator[CommandInfo]:
        pattern = re.compile(r"""
            (?P<tag>
                <
                (?P<close_slash>/)?
                (?P<tag_name>\w+)\s*
                (?P<attr_list>(?:\w+\s*\=\s*(?P<quot>["']).*?(?P=quot)\s*)*)
                (?P<elision_slash>/)?
                >
            )
            |(?P<passthrough>
                <\?.*?\?>|<!--.*?-->|<!\[CDATA\[.*?\]\]>|<!DOCTYPE.*?>
            )
            |(?P<entity>&(?P<unicode>\#(?P<hex>x)?)?(?P<content>.*?);)
            |(?P<char>[>"'])
        """, flags=re.VERBOSE | re.DOTALL)
        attribs_pattern = re.compile(r"""
            (?P<attr_name>\w+)
            \s*\=\s*
            (?P<quot>["'])(?P<attr_val>.*?)(?P=quot)
        """, flags=re.VERBOSE | re.DOTALL)
        open_stack: list[re.Match[str]] = []
        for match_obj in pattern.finditer(string):
            if match_obj.group("tag"):
                if match_obj.group("elision_slash"):
                    yield StandaloneCommandInfo(match_obj)
                elif not match_obj.group("close_slash"):
                    open_stack.append(match_obj)
                else:
                    open_match_obj = open_stack.pop()
                    attribs = {
                        attribs_match_obj.group("attr_name"): attribs_match_obj.group("attr_val")
                        for attribs_match_obj in attribs_pattern.finditer(open_match_obj.group("attr_list"))
                    } if (tag_name := open_match_obj.group("tag_name")) == "span" else cls._MARKUP_TAGS[tag_name]
                    yield BalancedCommandInfo(
                        attribs=attribs,
                        isolated=False,
                        open_match_obj=open_match_obj,
                        close_match_obj=match_obj,
                        open_replacement="",
                        close_replacement=""
                    )
            elif match_obj.group("char"):
                yield StandaloneCommandInfo(match_obj, replacement=cls._markup_escape(match_obj.group("char")))
            else:
                yield StandaloneCommandInfo(match_obj)
            assert not open_stack


class Markup(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[MarkupKwargs]
    ) -> None:
        super().__init__(MarkupIO.get(MarkupInput(string=string, **kwargs)))
