from __future__ import annotations


from dataclasses import dataclass
from typing import (
    Self,
    Unpack
)

from .pango_string_mobject import (
    PangoStringMobjectIO,
    PangoStringMobjectInput,
    PangoStringMobjectKwargs
)
from .string_mobject import StringMobject


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TextInput(PangoStringMobjectInput):
    pass


class TextKwargs(PangoStringMobjectKwargs, total=False):
    pass


class TextIO[TextInputT: TextInput](PangoStringMobjectIO[TextInputT]):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(
        cls: type[Self]
    ) -> str:
        return "text"


class Text(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[TextKwargs]
    ) -> None:
        super().__init__(TextIO.get(TextInput(string=string, **kwargs)))

    #@classmethod
    #@property
    #def _io_cls(
    #    cls: type[Self]
    #) -> type[TextIO]:
    #    return TextIO

    #@classmethod
    #@property
    #def _input_data_cls(
    #    cls: type[Self]
    #) -> type[TextInputData]:
    #    return TextInputData
