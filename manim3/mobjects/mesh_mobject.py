import moderngl
import numpy as np
import pyrr

from ..cameras.camera import Camera
from ..geometries.geometry import Geometry
from ..mobjects.mobject import Mobject
from ..utils.lazy import lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..custom_typing import *


__all__ = ["MeshMobject"]


#@dataclass
#class MeshMaterialAttributes:
#    color: ColorArrayType
#    color_map: moderngl.Texture | None


class MeshMobject(Mobject):
    def __init__(self):
        super().__init__()
        #self.geometry: Geometry | None = self.init_geometry()  # TODO: consider using trimesh
        #self.material: MeshMaterialAttributes = MeshMaterialAttributes(
        #    color=np.ones(4),
        #    color_map=None
        #)

    @lazy_property_initializer_writable
    @staticmethod
    def _invisible_() -> bool:
        return False

    @lazy_property_initializer
    @staticmethod
    def _color_map_() -> moderngl.Texture | None:
        return None

    @lazy_property
    @classmethod
    def _buffers_from_camera_(
        cls,
        camera: Camera
    ) -> dict[str, tuple[moderngl.Buffer, str]]:
        return {
            "in_projection_matrix": (cls._make_buffer(camera.get_projection_matrix()), "16f8 /r"),
            "in_view_matrix": (cls._make_buffer(camera.get_view_matrix()), "16f8 /r")
        }

    @lazy_property
    @classmethod
    def _buffers_from_geometry_(
        cls,
        geometry: Geometry,
        define_macros: list[str]
    ) -> dict[str, tuple[moderngl.Buffer, str]]:
        buffers = {
            "in_position": (cls._make_buffer(geometry.positions), "3f8 /v")
        }
        if "USE_UV" in define_macros:
            buffers["in_uv"] = (cls._make_buffer(geometry.uvs), "2f8 /v")
        return buffers

    @lazy_property
    @classmethod
    def _buffers_from_matrix_(
        cls,
        composite_matrix: pyrr.Matrix44
    ) -> dict[str, tuple[moderngl.Buffer, str]]:
        return {
            "in_model_matrix": (cls._make_buffer(composite_matrix), "16f8 /r")
        }

    @lazy_property
    @classmethod
    def _buffers_from_material_(cls) -> dict[str, tuple[moderngl.Buffer, str]]:  # TODO
        color = np.ones(4)
        return {
            "in_color": (cls._make_buffer(color), "4f8 /r")
        }

    @lazy_property
    @classmethod
    def _shader_filename_(cls) -> str:
        return "mesh"

    @lazy_property
    @classmethod
    def _define_macros_(cls, color_map: moderngl.Texture | None) -> list[str]:
        defines = []
        if color_map is not None:
            defines.append("USE_UV")
            defines.append("USE_COLOR_MAP")
        return defines

    @lazy_property
    @classmethod
    def _textures_dict_(
        cls,
        define_macros: list[str],
        color_map: moderngl.Texture | None
    ) -> dict[str, tuple[moderngl.Texture, int]]:
        textures = {}
        if "USE_COLOR_MAP" in define_macros:
            textures["uniform_color_map"] = (color_map, 0)
        return textures

    @lazy_property
    @classmethod
    def _buffers_dict_(
        cls,
        buffers_from_camera: dict[str, tuple[moderngl.Buffer, str]],
        buffers_from_geometry: dict[str, tuple[moderngl.Buffer, str]],
        buffers_from_matrix: dict[str, tuple[moderngl.Buffer, str]],
        buffers_from_material: dict[str, tuple[moderngl.Buffer, str]],
    ) -> dict[str, tuple[moderngl.Buffer, str]]:
        # Update distributively, as making buffers is expensive
        return {
            **buffers_from_camera,
            **buffers_from_geometry,
            **buffers_from_matrix,
            **buffers_from_material,
        }

    @lazy_property
    @classmethod
    def _vertex_indices_(cls, geometry: Geometry) -> VertexIndicesType:
        return geometry.indices

    @lazy_property
    @classmethod
    def _render_primitive_(cls) -> int:
        return moderngl.TRIANGLES
