import numpy as np

from .index_buffer import IndexBuffer


class OmittedIndexBuffer(IndexBuffer):
	__slots__ = ()

	def __init__(self) -> None:
		super().__init__(
			data=np.zeros((0,), dtype=np.uint32)
		)
