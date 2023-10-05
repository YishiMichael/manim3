from __future__ import annotations


from dataclasses import dataclass
from typing import Self

from .pango_string_mobject import (
    PangoStringMobject,
    PangoStringMobjectIO,
    PangoStringMobjectInputData
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class TextInputData(PangoStringMobjectInputData):
    pass


class TextIO(PangoStringMobjectIO):
    __slots__ = ()

    @classmethod
    @property
    def _dir_name(
        cls: type[Self]
    ) -> str:
        return "text"


class Text(PangoStringMobject):
    __slots__ = ()

    @classmethod
    @property
    def _io_cls(
        cls: type[Self]
    ) -> type[TextIO]:
        return TextIO

    @classmethod
    @property
    def _input_data_cls(
        cls: type[Self]
    ) -> type[TextInputData]:
        return TextInputData
