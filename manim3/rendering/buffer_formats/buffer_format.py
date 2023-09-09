import numpy as np

from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject


class BufferFormat(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...]
    ) -> None:
        super().__init__()
        self._name_ = name
        self._shape_ = shape

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _name_() -> str:
        return ""

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _shape_() -> tuple[int, ...]:
        return ()

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _itemsize_() -> int:
        # Implemented in subclasses.
        return 0

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _size_(
        shape: tuple[int, ...]
    ) -> int:
        return int(np.prod(shape, dtype=np.int32))

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _nbytes_(
        itemsize: int,
        size: int
    ) -> int:
        return itemsize * size

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_empty_(
        size: int
    ) -> bool:
        return not size

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _dtype_() -> np.dtype:
        # Implemented in subclasses.
        return np.dtype("f4")

    @Lazy.property_collection(hasher=Lazy.naive_hasher)
    @staticmethod
    def _pointers_() -> tuple[tuple[tuple[str, ...], int], ...]:
        # Implemented in subclasses.
        return ()

    def _get_np_buffer_and_pointers(self) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, int]]]:

        def get_np_buffer_pointer(
            np_buffer: np.ndarray,
            name_chain: list[str]
        ) -> np.ndarray:
            if not name_chain:
                return np_buffer["_"]
            name = name_chain.pop(0)
            return get_np_buffer_pointer(np_buffer[name], name_chain)

        np_buffer = np.zeros(self._shape_, dtype=self._dtype_)
        np_buffer_pointers = {
            ".".join(name_chain): (get_np_buffer_pointer(np_buffer, list(name_chain)), base_ndim)
            for name_chain, base_ndim in self._pointers_
        }
        return np_buffer, np_buffer_pointers

    def _write(
        self,
        data_dict: dict[str, np.ndarray]
    ) -> bytes:
        np_buffer, np_buffer_pointers = self._get_np_buffer_and_pointers()
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data = data_dict[key]
            if not np_buffer_pointer.size:
                assert not data.size
                continue
            data_expanded = np.expand_dims(data, axis=tuple(range(-2, -base_ndim)))
            assert np_buffer_pointer.shape == data_expanded.shape
            np_buffer_pointer[...] = data_expanded
        return np_buffer.tobytes()

    def _read(
        self,
        data_bytes: bytes
    ) -> dict[str, np.ndarray]:
        data_dict: dict[str, np.ndarray] = {}
        np_buffer, np_buffer_pointers = self._get_np_buffer_and_pointers()
        np_buffer[...] = np.frombuffer(data_bytes, dtype=np_buffer.dtype).reshape(np_buffer.shape)
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data_expanded = np_buffer_pointer[...]
            data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))
            data_dict[key] = data
        return data_dict
