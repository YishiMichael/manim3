from .read_only_buffer import ReadOnlyBuffer


class TransformFeedbackBuffer(ReadOnlyBuffer):
    __slots__ = ()

    def __init__(
        self,
        *,
        fields: list[str],
        child_structs: dict[str, list[str]] | None = None,
        num_vertex: int,
        array_lens: dict[str, int] | None = None
    ) -> None:
        # The interface should be similar to `AttributesBuffer`.
        if child_structs is None:
            child_structs = {}
        if array_lens is None:
            array_lens = {}
        super().__init__(
            field="__VertexStruct__ __vertex__[__NUM_VERTEX__]",
            child_structs={
                "__VertexStruct__": fields,
                **child_structs
            },
            array_lens={
                "__NUM_VERTEX__": num_vertex,
                **array_lens
            }
        )
