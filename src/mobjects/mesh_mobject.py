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
            shader_filename="mesh",
            define_macros=material.get_define_macros(),
            uniforms=material.get_uniforms(camera),
            texture_arrays=material.get_texture_arrays(),
            vertex_attributes=geometry.get_vertex_attributes(),
            vertex_indices=geometry.get_indices(),
            render_primitive=moderngl.TRIANGLES,
            enable_depth_test=True,
            enable_blend=True
        )
