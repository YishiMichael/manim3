import re

from .buffer import Buffer


class TextureIdBuffer(Buffer):  # TODO: make TextureBuffer; bind Texture objs directly
    __slots__ = ()

    def __init__(
        self,
        *,
        field: str,
        array_lens: dict[str, int] | None = None
    ) -> None:
        replaced_field = re.sub(r"^sampler2D\b", "uint", field)
        assert field != replaced_field
        super().__init__(
            field=replaced_field,
            child_structs=None,
            array_lens=array_lens
        )
