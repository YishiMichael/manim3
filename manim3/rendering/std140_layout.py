#from __future__ import annotations


#import functools
#import itertools
#import operator
#import re
#from typing import (
#    ClassVar,
#    Never,
#    Self
#)

#from ..constants.custom_typing import ShapeType
#from .buffer_format import (
#    AtomicBufferFormat,
#    BufferField,
#    BufferFormat,
#    StructuredBufferFormat
#)


#class STD140Layout:
#    __slots__ = ()

#    _GL_DTYPES: ClassVar[dict[str, tuple[str, int, ShapeType]]] = {
#        "int":     ("i", 4, ()),
#        "ivec2":   ("i", 4, (2,)),
#        "ivec3":   ("i", 4, (3,)),
#        "ivec4":   ("i", 4, (4,)),
#        "uint":    ("u", 4, ()),
#        "uvec2":   ("u", 4, (2,)),
#        "uvec3":   ("u", 4, (3,)),
#        "uvec4":   ("u", 4, (4,)),
#        "float":   ("f", 4, ()),
#        "vec2":    ("f", 4, (2,)),
#        "vec3":    ("f", 4, (3,)),
#        "vec4":    ("f", 4, (4,)),
#        "mat2":    ("f", 4, (2, 2)),
#        "mat2x3":  ("f", 4, (2, 3)),  # TODO: check order
#        "mat2x4":  ("f", 4, (2, 4)),
#        "mat3x2":  ("f", 4, (3, 2)),
#        "mat3":    ("f", 4, (3, 3)),
#        "mat3x4":  ("f", 4, (3, 4)),
#        "mat4x2":  ("f", 4, (4, 2)),
#        "mat4x3":  ("f", 4, (4, 3)),
#        "mat4":    ("f", 4, (4, 4)),
#        "double":  ("f", 8, ()),
#        "dvec2":   ("f", 8, (2,)),
#        "dvec3":   ("f", 8, (3,)),
#        "dvec4":   ("f", 8, (4,)),
#        "dmat2":   ("f", 8, (2, 2)),
#        "dmat2x3": ("f", 8, (2, 3)),
#        "dmat2x4": ("f", 8, (2, 4)),
#        "dmat3x2": ("f", 8, (3, 2)),
#        "dmat3":   ("f", 8, (3, 3)),
#        "dmat3x4": ("f", 8, (3, 4)),
#        "dmat4x2": ("f", 8, (4, 2)),
#        "dmat4x3": ("f", 8, (4, 3)),
#        "dmat4":   ("f", 8, (4, 4))
#    }

#    def __new__(
#        cls: type[Self]
#    ) -> Never:
#        raise TypeError

#    @classmethod
#    def _parse_field_declaration(
#        cls: type[Self],
#        field_declaration: str,
#        array_lens_dict: dict[str, int]
#    ) -> tuple[str, str, ShapeType]:
#        pattern = re.compile(r"""
#            (?P<dtype_str>\w+?)
#            \s
#            (?P<name>\w+?)
#            (?P<shape>(\[\w+?\])*)
#        """, flags=re.VERBOSE)
#        match = pattern.fullmatch(field_declaration)
#        assert match is not None
#        dtype_str = match.group("dtype_str")
#        name = match.group("name")
#        shape = tuple(
#            int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else array_lens_dict[s]
#            for index_match in re.finditer(r"\[(\w+?)\]", match.group("shape"))
#        )
#        return dtype_str, name, shape

#    @classmethod
#    def get_atomic_buffer_format_item(
#        cls: type[Self],
#        field_declaration: str,
#        array_lens_dict: dict[str, int]
#    ) -> tuple[AtomicBufferFormat, str, ShapeType]:
#        dtype_str, name, shape = cls._parse_field_declaration(
#            field_declaration=field_declaration,
#            array_lens_dict=array_lens_dict
#        )
#        buffer_format = cls.get_atomic_buffer_format(
#            gl_dtype_str=dtype_str,
#            is_array=bool(shape)
#        )
#        return buffer_format, name, shape

#    @classmethod
#    def get_buffer_format_item(
#        cls: type[Self],
#        field_declaration: str,
#        struct_dict: dict[str, tuple[str, ...]],
#        array_lens_dict: dict[str, int]
#    ) -> tuple[BufferFormat, str, ShapeType]:
#        dtype_str, name, shape = cls._parse_field_declaration(
#            field_declaration=field_declaration,
#            array_lens_dict=array_lens_dict
#        )
#        if (struct_field_declarations := struct_dict.get(dtype_str)) is None:
#            buffer_format = cls.get_atomic_buffer_format(
#                gl_dtype_str=dtype_str,
#                is_array=bool(shape)
#            )
#        else:
#            buffer_format = cls.get_structured_buffer_format(tuple(
#                cls.get_buffer_format_item(
#                    field_declaration=field_declaration,
#                    struct_dict=struct_dict,
#                    array_lens_dict=array_lens_dict
#                )
#                for field_declaration in struct_field_declarations
#            ))
#        return buffer_format, name, shape

#    @classmethod
#    def get_atomic_buffer_format(
#        cls: type[Self],
#        #name: str,
#        gl_dtype_str: str,
#        is_array: bool
#    ) -> AtomicBufferFormat:
#        base_char, base_itemsize, base_shape = cls._GL_DTYPES[gl_dtype_str]
#        assert len(base_shape) <= 2 and all(2 <= l <= 4 for l in base_shape)
#        shape_dict = dict(enumerate(base_shape))
#        col_len = shape_dict.get(0, 1)
#        row_len = shape_dict.get(1, 1)
#        col_padding = 0 if not is_array and row_len == 1 else 4 - col_len
#        base_alignment = (col_len if not is_array and col_len <= 2 and row_len == 1 else 4) * base_itemsize
#        return AtomicBufferFormat(
#            #name=name,
#            #shape=shape,
#            itemsize=row_len * (col_len + col_padding) * base_itemsize,
#            base_alignment=base_alignment,
#            base_char=base_char,
#            base_itemsize=base_itemsize,
#            base_ndim=len(base_shape),
#            row_len=row_len,
#            col_len=col_len,
#            col_padding=col_padding
#        )

#    @classmethod
#    def get_structured_buffer_format(
#        cls: type[Self],
#        field_buffer_format_items: tuple[tuple[BufferFormat, str, ShapeType], ...]
#        #fields: tuple[str, ...],
#        #struct_dict: dict[str, tuple[str, ...]],
#        #array_lens_dict: dict[str, int]
#        #name: str,
#        #shape: ShapeType,
#        #children: tuple[BufferFormat, ...],
#        #names: tuple[str, ...],
#        #shapes: tuple[ShapeType, ...]
#    ) -> StructuredBufferFormat:
#        #field_buffer_format_items = tuple(
#        #    cls.get_buffer_format_item(
#        #        field=field,
#        #        struct_dict=struct_dict,
#        #        array_lens_dict=array_lens_dict
#        #    )
#        #    for field in fields
#        #)
#        #children: list[BufferFormat] = []
#        #names: list[str] = []
#        #shapes: list[ShapeType] = []
#        #sizes: list[int] = []
#        #offsets: list[int] = []
#        struct_base_alignment = 16
#        #next_base_alignments: list[int] = []
#        #if field_buffer_format_items:
#        #    next_base_alignments.extend(
#        #        buffer_format._base_alignment_
#        #        for buffer_format, _, _ in field_buffer_format_items
#        #    )
#        #    next_base_alignments.append(struct_base_alignment)

#        buffer_fields: list[BufferField] = []
#        offset: int = 0
#        for (buffer_format, name, shape), (next_buffer_format, _, _) in itertools.pairwise((
#            *field_buffer_format_items,
#            (BufferFormat(itemsize=0, base_alignment=struct_base_alignment), "", ())
#        )):
#            size = functools.reduce(operator.mul, shape, initial=1)
#            offset += buffer_format._itemsize_ * size
#            padding = (-offset) % next_buffer_format._base_alignment_
#            buffer_fields.append(BufferField(
#                buffer_format=buffer_format,
#                name=name,
#                shape=shape,
#                size=size,
#                offset=offset,
#                padding=padding
#            ))
#            offset += padding
#            #children.append(child)
#            #names.append(name)
#            #shapes.append(shape)
#            #offsets.append(offset)

#        #base_alignment = 16
#        #offset += (-offset) % base_alignment
#        return StructuredBufferFormat(
#            #name=name,
#            #shape=shape,
#            itemsize=offset,
#            base_alignment=struct_base_alignment,
#            buffer_fields=tuple(buffer_fields)
#            #children=tuple(children),
#            #names=tuple(names),
#            #shapes=tuple(shapes),
#            #offsets=tuple(offsets)
#        )

#    #@classmethod
#    #def _get_atomic_col_padding(
#    #    cls: type[Self],
#    #    col_len: int,
#    #    row_len: int,
#    #    is_array: bool
#    #) -> int:
#    #    return 0 if not is_array and row_len == 1 else 4 - col_len

#    #@classmethod
#    #def _get_atomic_base_alignment(
#    #    cls: type[Self],
#    #    col_len: int,
#    #    row_len: int,
#    #    base_itemsize: int,
#    #    is_array: bool
#    #) -> int:
#    #    return (col_len if not is_array and col_len <= 2 and row_len == 1 else 4) * base_itemsize

#    #@classmethod
#    #def _get_structured_base_alignment(
#    #    cls: type[Self]
#    #) -> int:
#    #    return 16
