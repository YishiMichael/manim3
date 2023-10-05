from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.custom_typing import NP_xi4
from .write_only_buffer import WriteOnlyBuffer


class IndexBuffer(WriteOnlyBuffer):
    __slots__ = ("_omitted",)

    def __init__(
        self: Self,
        *,
        data: NP_xi4 | None = None
    ) -> None:
        omitted = data is None
        if data is None:
            data = np.zeros((0,), dtype=np.int32)
        super().__init__(
            field="uint __index__[__NUM_INDEX__]",
            child_structs={},
            array_lens={
                "__NUM_INDEX__": len(data)
            }
        )
        self._omitted: bool = omitted
        self.write({
            "": data.astype(np.uint32)
        })
