__all__ = ["MeshMobject"]


import moderngl
import numpy as np

from ..geometries.geometry import Geometry
from ..custom_typing import Mat4T
from ..mobjects.mobject import Mobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    Framebuffer,
    RenderStep,
    Renderable,
    TextureStorage,
    UniformBlockBuffer
)
from ..utils.scene_config import SceneConfig


#class MeshMaterial(ABC):
#    @abstractmethod
#    def _get_render_step(
#        self,
#        scene: Scene,
#        geometry: Trimesh,
#        target_framebuffer: moderngl.Framebuffer
#    ) -> RenderStep:
#        pass


#class SimpleMeshMaterial(MeshMaterial):
#    def __init__(self, color: ColorArrayType):
#        self.color: ColorArrayType = color


#class TexturedMeshMaterial(MeshMaterial):
#    def __init__(self, color_map: ColorArrayType):
#        self.color: ColorArrayType = color
#    color: ColorArrayType
#    color_map: moderngl.Texture | None


class MeshMobject(Mobject):
    #@lazy_property
    #@staticmethod
    #def _geometry_matrix_() -> Mat4T:
    #    return np.identity(4)

    @lazy_property_initializer
    @staticmethod
    def _ub_model_matrices_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_model_matrices", [
            "mat4 u_model_matrix",
            #"mat4 u_geometry_matrix"
        ])

    @lazy_property
    @staticmethod
    def _ub_model_matrices_(
        ub_model_matrices_o: UniformBlockBuffer,
        model_matrix: Mat4T,
        #geometry_matrix: Mat4T
    ) -> UniformBlockBuffer:
        ub_model_matrices_o.write({
            "u_model_matrix": model_matrix,
            #"u_geometry_matrix": geometry_matrix
        })
        return ub_model_matrices_o

    @lazy_property_initializer_writable
    @staticmethod
    def _geometry_() -> Geometry:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _color_map_texture_() -> moderngl.Texture | None:
        return None

    @lazy_property_initializer
    @staticmethod
    def _u_color_maps_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_maps[NUM_U_COLOR_MAPS]")

    @lazy_property
    @staticmethod
    def _u_color_maps_(
        u_color_maps_o: TextureStorage,
        color_map_texture: moderngl.Texture | None
    ) -> TextureStorage:
        textures = [color_map_texture] if color_map_texture is not None else []
        u_color_maps_o.write(np.array(textures))
        return u_color_maps_o

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_only_() -> int:
        return moderngl.BLEND | moderngl.DEPTH_TEST

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        self._render_by_step(RenderStep(
            shader_str=Renderable._read_shader("mesh"),
            texture_storages=[
                self._u_color_maps_
            ],
            uniform_blocks=[
                scene_config._camera_._ub_camera_matrices_,
                self._ub_model_matrices_,
                scene_config._ub_lights_
            ],
            attributes=self._geometry_._attributes_,
            subroutines={},
            index_buffer=self._geometry_._index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_,
            mode=moderngl.TRIANGLES
        ))
