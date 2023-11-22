from __future__ import annotations


from typing import (
    ClassVar,
    Self,
    Unpack
)

import attrs

from .pango_string_mobject import (
    PangoStringMobjectIO,
    PangoStringMobjectInput,
    PangoStringMobjectKwargs
)
from .string_mobject import StringMobject


@attrs.frozen(kw_only=True)
class TextInput(PangoStringMobjectInput):
    pass


class TextKwargs(PangoStringMobjectKwargs, total=False):
    pass


class TextIO[TextInputT: TextInput](PangoStringMobjectIO[TextInputT]):
    __slots__ = ()

    _dir_name: ClassVar[str] = "text"


class Text(StringMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        string: str,
        **kwargs: Unpack[TextKwargs]
    ) -> None:
        super().__init__(TextIO.get(TextInput(string=string, **kwargs)))
