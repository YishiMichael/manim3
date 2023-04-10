__all__ = [
    "BlendEquation",
    "BlendFunc",
    "ContextFlag",
    "PrimitiveMode",
    "TextureFilter"
]


from enum import Enum

import moderngl


class ContextFlag(Enum):
    NOTHING = moderngl.NOTHING
    BLEND = moderngl.BLEND
    DEPTH_TEST = moderngl.DEPTH_TEST
    CULL_FACE = moderngl.CULL_FACE
    RASTERIZER_DISCARD = moderngl.RASTERIZER_DISCARD
    PROGRAM_POINT_SIZE = moderngl.PROGRAM_POINT_SIZE


class PrimitiveMode(Enum):
    POINTS = moderngl.POINTS
    LINES = moderngl.LINES
    LINE_LOOP = moderngl.LINE_LOOP
    LINE_STRIP = moderngl.LINE_STRIP
    TRIANGLES = moderngl.TRIANGLES
    TRIANGLE_STRIP = moderngl.TRIANGLE_STRIP
    TRIANGLE_FAN = moderngl.TRIANGLE_FAN
    LINES_ADJACENCY = moderngl.LINES_ADJACENCY
    LINE_STRIP_ADJACENCY = moderngl.LINE_STRIP_ADJACENCY
    TRIANGLES_ADJACENCY = moderngl.TRIANGLES_ADJACENCY
    TRIANGLE_STRIP_ADJACENCY = moderngl.TRIANGLE_STRIP_ADJACENCY
    PATCHES = moderngl.PATCHES


class TextureFilter(Enum):
    NEAREST = moderngl.NEAREST
    LINEAR = moderngl.LINEAR
    NEAREST_MIPMAP_NEAREST = moderngl.NEAREST_MIPMAP_NEAREST
    LINEAR_MIPMAP_NEAREST = moderngl.LINEAR_MIPMAP_NEAREST
    NEAREST_MIPMAP_LINEAR = moderngl.NEAREST_MIPMAP_LINEAR
    LINEAR_MIPMAP_LINEAR = moderngl.LINEAR_MIPMAP_LINEAR


class BlendFunc(Enum):
    ZERO = moderngl.ZERO
    ONE = moderngl.ONE
    SRC_COLOR = moderngl.SRC_COLOR
    ONE_MINUS_SRC_COLOR = moderngl.ONE_MINUS_SRC_COLOR
    SRC_ALPHA = moderngl.SRC_ALPHA
    ONE_MINUS_SRC_ALPHA = moderngl.ONE_MINUS_SRC_ALPHA
    DST_ALPHA = moderngl.DST_ALPHA
    ONE_MINUS_DST_ALPHA = moderngl.ONE_MINUS_DST_ALPHA
    DST_COLOR = moderngl.DST_COLOR
    ONE_MINUS_DST_COLOR = moderngl.ONE_MINUS_DST_COLOR


class BlendEquation(Enum):
    FUNC_ADD = moderngl.FUNC_ADD
    FUNC_SUBTRACT = moderngl.FUNC_SUBTRACT
    FUNC_REVERSE_SUBTRACT = moderngl.FUNC_REVERSE_SUBTRACT
    MIN = moderngl.MIN
    MAX = moderngl.MAX
