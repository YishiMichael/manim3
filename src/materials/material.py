from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from colour import Color

from constants import (
    AddEquation,
    AlwaysStencilFunc,
    FrontSide,
    KeepStencilOp,
    LessEqualDepth,
    NormalBlending,
    OneMinusSrcAlphaFactor,
    SrcAlphaFactor,
)
from utils.arrays import Mat3, Mat4, Vec2, Vec3, Vec4
from utils.texture import Texture


Uniform = Union[
    float,
    int,
    Mat3,
    Mat4,
    Vec2,
    Vec3,
    Vec4,
    Color,
    Texture
]


@dataclass
class Material:
    blending: int = NormalBlending
    side: int = FrontSide
    vertexColors: bool = False

    opacity: float = 1.0
    transparent: bool = False

    blendSrc: int = SrcAlphaFactor
    blendDst: int = OneMinusSrcAlphaFactor
    blendEquation: int = AddEquation
    blendSrcAlpha: int | None = None
    blendDstAlpha: int | None = None
    blendEquationAlpha: int | None = None

    depthFunc: int = LessEqualDepth
    depthTest: bool = True
    depthWrite: bool = True

    stencilWriteMask: int = 0xff
    stencilFunc: int = AlwaysStencilFunc
    stencilRef: int = 0
    stencilFuncMask: int = 0xff
    stencilFail: int = KeepStencilOp
    stencilZFail: int = KeepStencilOp
    stencilZPass: int = KeepStencilOp
    stencilWrite: bool = False

    clippingPlanes: None = None  # list[Plane]
    clipIntersection: bool = False
    clipShadows: bool = False

    shadowSide: int | None = None

    colorWrite: bool = True

    precision: str | None = None # override the renderer's default precision for this material

    polygonOffset: bool = False
    polygonOffsetFactor: int = 0
    polygonOffsetUnits: int = 0


    dithering: bool = False

    alphaToCoverage: bool = False
    premultipliedAlpha: bool = False

    visible: bool = True

    toneMapped: bool = True

    version: int = 0

    alphaTest: int = 0

    def construct_uniform(self) -> dict[str, Uniform]:
        # https://github.com/mrdoob/three.js/issues/5876


