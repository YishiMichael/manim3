from ...constants.custom_typing import NP_xu4
from .write_only_buffer import WriteOnlyBuffer


class IndexBuffer(WriteOnlyBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        data: NP_xu4# | None
    ) -> None:
        #data_len = 0 if data is None else len(data)
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
        #if data is not None:
        #    self.write({
        #        "": data
        #    })
        #    self._omitted_ = False

    #@Lazy.variable_hashable
    #@classmethod
    #def _omitted_(cls) -> bool:
    #    return True
