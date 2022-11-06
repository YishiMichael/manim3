from dataclasses import dataclass
from typing import Any

import moderngl
import numpy as np

from ..cameras.camera import Camera
from ..mobjects.mobject import Mobject
from ..shader_utils import ShaderData
from ..typing import *


__all__ = [
    "GeometryAttributes",
    "MeshMaterialAttributes",
    "MeshMobject"
]


@dataclass
class GeometryAttributes:
    index: VertexIndicesType
    position: Vector3ArrayType
    uv: Vector2ArrayType


@dataclass
class MeshMaterialAttributes:
    color: ColorArrayType = np.ones(4)
    color_map: TextureArrayType | None = None
    enable_depth_test: bool = True
    enable_blend: bool = True
    cull_face: str = "back"
    wireframe: bool = False


class MeshMobject(Mobject):
    def __init__(self: Self, **kwargs):
        super().__init__()
        self.geometry: GeometryAttributes = self.init_geometry_attributes()
        self.material: MeshMaterialAttributes = self.init_mesh_material_attributes(kwargs)

    def init_geometry_attributes(self: Self) -> GeometryAttributes:
        raise NotImplementedError

    def init_mesh_material_attributes(self: Self, kwargs: dict[str, Any]) -> MeshMaterialAttributes:
        return MeshMaterialAttributes(**kwargs)

    def setup_shader_data(self: Self, camera: Camera) -> ShaderData:
        geometry = self.geometry
        material = self.material

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
        attrs["in_position"] = (geometry.position, "3f8 /v")
        attrs["in_color"] = (material.color, "4f8 /r")
        if "USE_UV" in defines:
            attrs["in_uv"] = (geometry.uv, "2f8 /v")

        return ShaderData(
            enable_depth_test=material.enable_depth_test,
            enable_blend=material.enable_blend,
            cull_face=material.cull_face,
            wireframe=material.wireframe,
            shader_filename="mesh",
            define_macros=defines,
            textures_dict=textures,
            #uniforms_dict=uniforms,
            attributes_dict=attrs,
            vertex_indices=geometry.index,
            render_primitive=moderngl.TRIANGLES
        )
