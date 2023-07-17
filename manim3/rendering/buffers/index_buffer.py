from ...constants.custom_typing import NP_xu4
from .write_only_buffer import WriteOnlyBuffer


class IndexBuffer(WriteOnlyBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        data: NP_xu4
    ) -> None:
        super().__init__(
            field="uint __index__[__NUM_INDEX__]",
            child_structs={},
            array_lens={
                "__NUM_INDEX__": len(data)
            }
        )
        self.write({
            "": data
        })
