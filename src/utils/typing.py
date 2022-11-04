from typing import Any, Literal, Union

import numpy as np


__all__ = [
	"Vector2Type",
	"Vector3Type",
	"Vector4Type",
	"Matrix33Type",
	"Matrix44Type",
	"IntArrayType",
	"FloatArrayType",
	"Vector2ArrayType",
	"Vector3ArrayType",
	"Vector4ArrayType",
	"Matrix33ArrayType",
	"Matrix44ArrayType",
	"ColorArrayType",
	"UniformType",
	"TextureArrayType",
	"AttributesType",
	"VertexIndicesType",
	"Self"
]


Vector2Type = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
Vector3Type = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
Vector4Type = np.ndarray[tuple[Literal[4]], np.dtype[np.float64]]
Matrix33Type = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float64]]
Matrix44Type = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float64]]

IntArrayType = np.ndarray[tuple[int], np.dtype[np.int32]]
FloatArrayType = np.ndarray[tuple[int], np.dtype[np.float64]]
Vector2ArrayType = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]
Vector3ArrayType = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]
Vector4ArrayType = np.ndarray[tuple[int, Literal[4]], np.dtype[np.float64]]
Matrix33ArrayType = np.ndarray[tuple[int, Literal[3], Literal[3]], np.dtype[np.float64]]
Matrix44ArrayType = np.ndarray[tuple[int, Literal[4], Literal[4]], np.dtype[np.float64]]

ColorArrayType = Vector4Type
UniformType = Union[int, float, np.ndarray[tuple[int, ...], np.dtype[np.float64]]]
TextureArrayType = np.ndarray[tuple[int, int, Literal[4]], np.dtype[np.uint8]]
AttributesType = np.ndarray[tuple[int], np.dtype[Any]]
VertexIndicesType = np.ndarray[tuple[int], np.dtype[np.uint32]]

Self = Any  # This shall be removed when advanced to py 3.11
