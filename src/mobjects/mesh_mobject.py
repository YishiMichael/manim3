import moderngl

from cameras.camera import Camera
from geometries.geometry import Geometry
from materials.material import Material
from mobjects.mobject import Mobject
from shader_utils import ShaderData
from utils.typing import *


__all__ = ["MeshMobject"]


class MeshMobject(Mobject):
    def __init__(self: Self, geometry: Geometry, material: Material):
        super().__init__()
        self.geometry: Geometry = geometry
        self.material: Material = material

    def setup_shader_data(self: Self, camera: Camera) -> ShaderData:
        geometry = self.geometry
        material = self.material
        return ShaderData(
            enable_depth_test=True,
            enable_blend=True,
            shader_filename="mesh",
            define_macros=material.get_define_macros(),
            texture_dict=material.get_texture_dict(),
            attributes_dict={
                AttributeUsage.V: geometry.get_attributes_v(),
                AttributeUsage.R: material.get_attributes_r(camera)
            },
            vertex_indices=geometry.get_vertex_indices(),
            render_primitive=moderngl.TRIANGLES
        )
