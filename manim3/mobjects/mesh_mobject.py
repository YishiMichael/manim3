#from dataclasses import dataclass

from abc import abstractmethod
import moderngl
import numpy as np
import skia

#from ..cameras.camera import Camera
from ..geometries.geometry import Geometry
from ..mobjects.mobject import Mobject
#from ..shader_utils import ShaderData
from ..utils.lazy import lazy_property
from ..custom_typing import *


__all__ = [
    #"MeshMaterialAttributes",
    "MeshMobject"
]


#@dataclass
#class MeshMaterialAttributes:
#    color: ColorArrayType
#    color_map: skia.Image | None


class MeshMobject(Mobject):
    def __init__(self: Self):
        super().__init__()
        self.abandon_render = False
        #self.geometry: Geometry | None = self.init_geometry()  # TODO: consider using trimesh
        #self.material: MeshMaterialAttributes = MeshMaterialAttributes(
        #    color=np.ones(4),
        #    color_map=None
        #)

    @lazy_property
    @abstractmethod
    def _geometry(self: Self) -> Geometry:
        raise NotImplementedError

    def get_local_sample_points(self: Self) -> Vector3ArrayType:
        if self._geometry is None:
            return np.zeros((0, 3))
        return self._geometry.position

    @lazy_property
    def _color_map(self: Self) -> skia.Image | None:
        return None

    @lazy_property
    def _shader_filename(self: Self) -> str:
        return "mesh"

    @lazy_property
    def _define_macros(self: Self) -> list[str]:
        defines = []
        if self._color_map is not None:
            defines.append("USE_UV")
            defines.append("USE_COLOR_MAP")
        return defines

    @lazy_property
    def _textures_dict(self: Self) -> dict[str, tuple[skia.Image, int]]:
        textures = {}
        if "USE_COLOR_MAP" in self._define_macros:
            textures["uniform_color_map"] = (self._color_map, 0)
        return textures

    @lazy_property
    def _attributes_dict(self: Self) -> dict[str, tuple[AttributeType, str]]:
        color = np.ones(4)
        attrs = {}
        attrs["in_projection_matrix"] = (self._camera.get_projection_matrix(), "16f8 /r")
        attrs["in_view_matrix"] = (self._camera.get_view_matrix(), "16f8 /r")
        attrs["in_model_matrix"] = (self._matrix, "16f8 /r")
        attrs["in_position"] = (self._geometry.position, "3f8 /v")
        attrs["in_color"] = (color, "4f8 /r")
        if "USE_UV" in self._define_macros:
            attrs["in_uv"] = (self._geometry.uv, "2f8 /v")
        return attrs

    @lazy_property
    def _vertex_indices(self: Self) -> VertexIndicesType:
        return self._geometry.index

    @lazy_property
    def _render_primitive(self: Self) -> int:
        return moderngl.TRIANGLES

    #@lazy_property
    #def shader_data(self: Self) -> ShaderData:
    #    geometry = self.geometry
    #    if geometry is None:
    #        return None
    #    #material = self.material
    #    #color_map = self._color_map
    #    #if color_map is not None:
    #    #    material.color_map = np.flipud(color_map)  # flip y  # TODO 

    #    return ShaderData(
    #        enable_depth_test=self._enable_depth_test,
    #        enable_blend=self._enable_blend,
    #        cull_face=self._cull_face,
    #        wireframe=self._wireframe,
    #        shader_filename="mesh",
    #        define_macros=defines,
    #        textures_dict=textures,
    #        #uniforms_dict=uniforms,
    #        attributes_dict=attrs,
    #        vertex_indices=geometry.index,
    #        render_primitive=moderngl.TRIANGLES
    #    )
