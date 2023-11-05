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
    CommandFlag,
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
    def _iter_command_matches(
        cls: type[Self],
        string: str
    ) -> Iterator[re.Match[str]]:
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
        yield from pattern.finditer(string)

    @classmethod
    def _get_command_flag(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> CommandFlag:
        if match_obj.group("tag"):
            if match_obj.group("close_slash"):
                return CommandFlag.CLOSE
            if not match_obj.group("elision_slash"):
                return CommandFlag.OPEN
        return CommandFlag.OTHER

    @classmethod
    def _replace_for_content(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("tag"):
            return ""
        if match_obj.group("char"):
            return cls._markup_escape(match_obj.group("char"))
        return match_obj.group()

    @classmethod
    def _replace_for_matching(
        cls: type[Self],
        match_obj: re.Match[str]
    ) -> str:
        if match_obj.group("tag") or match_obj.group("passthrough"):
            return ""
        if match_obj.group("entity"):
            if match_obj.group("unicode"):
                base = 10
                if match_obj.group("hex"):
                    base = 16
                return chr(int(match_obj.group("content"), base))
            return cls._markup_unescape(match_obj.group("entity"))
        return match_obj.group()


class Markup(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[MarkupKwargs]
    ) -> None:
        super().__init__(MarkupIO.get(MarkupInput(string=string, **kwargs)))
