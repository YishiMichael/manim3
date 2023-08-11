from dataclasses import dataclass

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
    def _dir_name(cls) -> str:
        return "text"


class Text(PangoStringMobject):
    __slots__ = ()

    @classmethod
    @property
    def _io_cls(cls) -> type[TextIO]:
        return TextIO

    @classmethod
    @property
    def _input_data_cls(cls) -> type[TextInputData]:
        return TextInputData
