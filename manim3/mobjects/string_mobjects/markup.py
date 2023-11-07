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
    #CommandFlag,
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
            #span = match_obj.span()
            #flag = CommandFlag.OTHER
            #replacement = match_obj.group()
            #attribs: dict[str, str] | None = None
            if match_obj.group("tag"):
                #replacement = ""
                if match_obj.group("elision_slash"):
                    yield StandaloneCommandInfo(match_obj)
                    #yield CommandInfo(
                    #    match_obj=match_obj,
                    #    replacement=""
                    #)
                elif not match_obj.group("close_slash"):
                    open_stack.append(match_obj)
                    #yield CommandInfo(
                    #    match_obj=match_obj,
                    #    command_flag=CommandFlag.OPEN,
                    #    replacement="",
                    #    attribs={
                    #        match_obj.group("attr_name"): match_obj.group("attr_val")
                    #        for match_obj in attribs_pattern.finditer(match_obj.group("attr_list"))
                    #    } if (tag_name := match_obj.group("tag_name")) == "span" else cls._MARKUP_TAGS.get(tag_name, {})
                    #)
                    #flag = CommandFlag.CLOSE
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
                    #if tag_name == "span":
                    #yield CommandInfo(
                    #    match_obj=match_obj,
                    #    command_flag=CommandFlag.CLOSE,
                    #    replacement=""
                    #)
                    #flag = CommandFlag.OPEN
                    #attribs = {
                    #    match_obj.group("attr_name"): match_obj.group("attr_val")
                    #    for match_obj in attribs_pattern.finditer(match_obj.group("attr_list"))
                    #} if (tag_name := match_obj.group("tag_name")) == "span" else cls._MARKUP_TAGS.get(tag_name, {})
                    #return cls._MARKUP_TAGS.get(tag_name, {})
                    #yield CommandInfo(
                    #    span=match_obj.span(),
                    #    flag=CommandFlag.OPEN,
                    #    replacement="",
                    #    attribs=attribs
                    #)
            elif match_obj.group("char"):
                yield StandaloneCommandInfo(match_obj, replacement=cls._markup_escape(match_obj.group("char")))
                #yield CommandInfo(
                #    match_obj=match_obj,
                #    replacement=cls._markup_escape(match_obj.group("char"))
                #)
                #replacement = cls._markup_escape(match_obj.group("char"))
            else:
                yield StandaloneCommandInfo(match_obj)
                #yield CommandInfo(
                #    match_obj=match_obj
                #)
            #yield CommandInfo(
            #    span=span,
            #    flag=flag,
            #    replacement=replacement,
            #    attribs=attribs
            #)
            assert not open_stack

    #@classmethod
    #def _iter_command_matches(
    #    cls: type[Self],
    #    string: str
    #) -> Iterator[re.Match[str]]:
    #    pattern = re.compile(r"""
    #        (?P<tag>
    #            <
    #            (?P<close_slash>/)?
    #            (?P<tag_name>\w+)\s*
    #            (?P<attr_list>(?:\w+\s*\=\s*(?P<quot>["']).*?(?P=quot)\s*)*)
    #            (?P<elision_slash>/)?
    #            >
    #        )
    #        |(?P<passthrough>
    #            <\?.*?\?>|<!--.*?-->|<!\[CDATA\[.*?\]\]>|<!DOCTYPE.*?>
    #        )
    #        |(?P<entity>&(?P<unicode>\#(?P<hex>x)?)?(?P<content>.*?);)
    #        |(?P<char>[>"'])
    #    """, flags=re.VERBOSE | re.DOTALL)
    #    yield from pattern.finditer(string)

    #@classmethod
    #def _get_command_flag(
    #    cls: type[Self],
    #    match_obj: re.Match[str]
    #) -> CommandFlag:
    #    if match_obj.group("tag"):
    #        if match_obj.group("close_slash"):
    #            return CommandFlag.CLOSE
    #        if not match_obj.group("elision_slash"):
    #            return CommandFlag.OPEN
    #    return CommandFlag.OTHER

    #@classmethod
    #def _get_command_replacement(
    #    cls: type[Self],
    #    match_obj: re.Match[str]
    #) -> str:
    #    if match_obj.group("tag"):
    #        return ""
    #    if match_obj.group("char"):
    #        return cls._markup_escape(match_obj.group("char"))
    #    return match_obj.group()

    #@classmethod
    #def _replace_for_matching(
    #    cls: type[Self],
    #    match_obj: re.Match[str]
    #) -> str:
    #    if match_obj.group("tag") or match_obj.group("passthrough"):
    #        return ""
    #    if match_obj.group("entity"):
    #        if match_obj.group("unicode"):
    #            base = 10
    #            if match_obj.group("hex"):
    #                base = 16
    #            return chr(int(match_obj.group("content"), base))
    #        return cls._markup_unescape(match_obj.group("entity"))
    #    return match_obj.group()


class Markup(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[MarkupKwargs]
    ) -> None:
        super().__init__(MarkupIO.get(MarkupInput(string=string, **kwargs)))
