import numpy as np

from ..mobjects.mesh_mobject import MeshMaterialAttributes
from ..mobjects.plane import Plane
from ..typing import *


__all__ = ["PlanarMobject"]


class PlanarMobject(Plane):
    def __init__(
        self: Self,
        enable_depth_test: bool = False,
        enable_blend: bool = False,
        cull_face: str = "back",
        wireframe: bool = False
    ):
        super().__init__(
            color_map=self.init_color_map(),
            enable_depth_test=enable_depth_test,
            enable_blend=enable_blend,
            cull_face=cull_face,
            wireframe=wireframe
        )

    def init_color_map(self: Self) -> TextureArrayType:
        raise NotImplementedError
