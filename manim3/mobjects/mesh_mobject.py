from abc import abstractmethod
from dataclasses import dataclass

import moderngl
import numpy as np

from ..cameras.camera import Camera
from ..geometries.geometry import Geometry
from ..mobjects.mobject import Mobject
from ..shader_utils import ShaderData
from ..typing import *


__all__ = [
    "MeshMaterialAttributes",
    "MeshMobject"
]


@dataclass
class MeshMaterialAttributes:
    color: ColorArrayType
    color_map: TextureArrayType | None


class MeshMobject(Mobject):
    def __init__(self: Self):
        super().__init__()
        self.geometry: Geometry | None = self.init_geometry()
        self.material: MeshMaterialAttributes = MeshMaterialAttributes(
            color=np.ones(4),
            color_map=None
        )

    def init_geometry(self: Self) -> Geometry | None:
        return None

    def get_local_sample_points(self: Self) -> Vector3ArrayType:
        if self.geometry is None:
            return np.zeros((0, 3))
        return self.geometry.position

    def load_color_map(self: Self) -> TextureArrayType | None:
        return None

    def setup_shader_data(self: Self, camera: Camera) -> ShaderData | None:
        geometry = self.geometry
        if geometry is None:
            return None
        material = self.material
        color_map = self.load_color_map()
        if color_map is not None:
            material.color_map = np.flipud(color_map)  # flip y

        defines = []
        if material.color_map is not None:
            defines.append("USE_UV")
            defines.append("USE_COLOR_MAP")

        textures = {}
        if "USE_COLOR_MAP" in defines:
            textures["uniform_color_map"] = (material.color_map, 0)

        attrs = {}
        attrs["in_projection_matrix"] = (camera.get_projection_matrix(), "16f8 /r")
        attrs["in_view_matrix"] = (camera.get_view_matrix(), "16f8 /r")
        attrs["in_model_matrix"] = (self.matrix, "16f8 /r")
        attrs["in_position"] = (geometry.position, "3f8 /v")
        attrs["in_color"] = (material.color, "4f8 /r")
        if "USE_UV" in defines:
            attrs["in_uv"] = (geometry.uv, "2f8 /v")

        return ShaderData(
            enable_depth_test=self.enable_depth_test,
            enable_blend=self.enable_blend,
            cull_face=self.cull_face,
            wireframe=self.wireframe,
            shader_filename="mesh",
            define_macros=defines,
            textures_dict=textures,
            #uniforms_dict=uniforms,
            attributes_dict=attrs,
            vertex_indices=geometry.index,
            render_primitive=moderngl.TRIANGLES
        )
