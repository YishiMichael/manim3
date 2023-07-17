import re
from typing import (
    ClassVar,
    Iterator
)

import moderngl
import numpy as np

from ...lazy.lazy import (
    Lazy,
    LazyObject
)
from ...toplevel.toplevel import Toplevel
from ..buffer_formats.atomic_buffer_format import AtomicBufferFormat
from ..buffer_formats.buffer_format import BufferFormat
from ..buffer_formats.buffer_layout import BufferLayout
from ..buffer_formats.structured_buffer_format import StructuredBufferFormat


class Buffer(LazyObject):
    __slots__ = ()

    _vacant_buffers: ClassVar[list[moderngl.Buffer]] = []

    def __init__(
        self,
        field: str,
        child_structs: dict[str, list[str]] | None,
        array_lens: dict[str, int] | None
    ) -> None:
        super().__init__()
        self._field_ = field
        if child_structs is not None:
            self._child_struct_items_ = tuple(
                (name, tuple(child_struct_fields))
                for name, child_struct_fields in child_structs.items()
            )
        if array_lens is not None:
            self._array_len_items_ = tuple(array_lens.items())

    @Lazy.variable_hashable
    @classmethod
    def _field_(cls) -> str:
        return ""

    @Lazy.variable_hashable
    @classmethod
    def _child_struct_items_(cls) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return ()

    @Lazy.variable_hashable
    @classmethod
    def _array_len_items_(cls) -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.property_hashable
    @classmethod
    def _layout_(cls) -> BufferLayout:
        return BufferLayout.PACKED

    @Lazy.property
    @classmethod
    def _buffer_format_(
        cls,
        field: str,
        child_struct_items: tuple[tuple[str, tuple[str, ...]], ...],
        array_len_items: tuple[tuple[str, int], ...],
        layout: BufferLayout
    ) -> BufferFormat:

        def parse_field_str(
            field_str: str,
            array_lens_dict: dict[str, int]
        ) -> tuple[str, str, tuple[int, ...]]:
            pattern = re.compile(r"""
                (?P<dtype_str>\w+?)
                \s
                (?P<name>\w+?)
                (?P<shape>(\[\w+?\])*)
            """, flags=re.VERBOSE)
            match_obj = pattern.fullmatch(field_str)
            assert match_obj is not None
            dtype_str = match_obj.group("dtype_str")
            name = match_obj.group("name")
            shape = tuple(
                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else array_lens_dict[s]
                for index_match in re.finditer(r"\[(\w+?)\]", match_obj.group("shape"))
            )
            return (dtype_str, name, shape)

        child_structs_dict = dict(child_struct_items)
        array_lens_dict = dict(array_len_items)

        def get_buffer_format(
            field: str
        ) -> BufferFormat:
            dtype_str, name, shape = parse_field_str(field, array_lens_dict)
            if (child_struct_fields := child_structs_dict.get(dtype_str)) is None:
                return AtomicBufferFormat(
                    name=name,
                    shape=shape,
                    gl_dtype_str=dtype_str,
                    layout=layout
                )
            return StructuredBufferFormat(
                name=name,
                shape=shape,
                children=[
                    get_buffer_format(child_struct_field)
                    for child_struct_field in child_struct_fields
                ],
                layout=layout
            )

        return get_buffer_format(field)

    @Lazy.property_external
    @classmethod
    def _np_buffer_(
        cls,
        buffer_format__shape: tuple[int, ...],
        buffer_format__dtype: np.dtype
    ) -> np.ndarray:
        return np.zeros(buffer_format__shape, dtype=buffer_format__dtype)

    @Lazy.property_external
    @classmethod
    def _np_buffer_pointers_(
        cls,
        np_buffer: np.ndarray,
        buffer_format: BufferFormat
    ) -> dict[str, tuple[np.ndarray, int]]:

        def get_pointers(
            np_buffer_pointer: np.ndarray,
            buffer_format: BufferFormat,
            name_chain: tuple[str, ...]
        ) -> Iterator[tuple[str, np.ndarray, int]]:
            if isinstance(buffer_format, AtomicBufferFormat):
                yield ".".join(name_chain), np_buffer_pointer["_"], buffer_format._base_ndim_
            elif isinstance(buffer_format, StructuredBufferFormat):
                for child in buffer_format._children_:
                    name = child._name_
                    yield from get_pointers(
                        np_buffer_pointer[name],
                        child,
                        name_chain + (name,)
                    )

        return {
            key: (np_buffer_pointer, base_ndim)
            for key, np_buffer_pointer, base_ndim in get_pointers(np_buffer, buffer_format, ())
        }

    @Lazy.property_hashable
    @classmethod
    def _np_buffer_pointer_keys_(
        cls,
        np_buffer_pointers: dict[str, tuple[np.ndarray, int]]
    ) -> tuple[str, ...]:
        return tuple(np_buffer_pointers)

    @classmethod
    def _fetch_buffer(cls) -> moderngl.Buffer:
        if cls._vacant_buffers:
            return cls._vacant_buffers.pop()
        return Toplevel.context.buffer()

    @classmethod
    def _finalize_buffer(
        cls,
        buffer: moderngl.Buffer
    ) -> None:
        cls._vacant_buffers.append(buffer)

    @classmethod
    def _write_to_buffer(
        cls,
        buffer: moderngl.Buffer,
        np_buffer: np.ndarray,
        np_buffer_pointers: dict[str, tuple[np.ndarray, int]],
        data_dict: dict[str, np.ndarray]
    ) -> None:
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data = data_dict[key]
            if not np_buffer_pointer.size:
                assert not data.size
                continue
            data_expanded = np.expand_dims(data, axis=tuple(range(-2, -base_ndim)))
            assert np_buffer_pointer.shape == data_expanded.shape
            np_buffer_pointer[...] = data_expanded

        buffer.orphan(np_buffer.nbytes)
        buffer.write(np_buffer.tobytes())

    @classmethod
    def _read_from_buffer(
        cls,
        buffer: moderngl.Buffer,
        np_buffer: np.ndarray,
        np_buffer_pointers: dict[str, tuple[np.ndarray, int]]
    ) -> dict[str, np.ndarray]:
        data_dict: dict[str, np.ndarray] = {}
        np_buffer[...] = np.frombuffer(buffer.read(), dtype=np_buffer.dtype).reshape(np_buffer.shape)
        for key, (np_buffer_pointer, base_ndim) in np_buffer_pointers.items():
            data_expanded = np_buffer_pointer[...]
            data = np.squeeze(data_expanded, axis=tuple(range(-2, -base_ndim)))
            data_dict[key] = data
        return data_dict
