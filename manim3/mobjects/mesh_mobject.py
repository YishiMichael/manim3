__all__ = ["MeshMobject"]


import inspect
from typing import Callable

from colour import Color, re
import moderngl
import numpy as np

from ..geometries.geometry import Geometry
from ..custom_typing import (
    ColorType,
    Mat4T,
    Vec4T,
    Vec4sT
)
from ..mobjects.mobject import Mobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    AttributesBuffer,
    Framebuffer,
    IndexBuffer,
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
    def _geometry_() -> Geometry:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _color_() -> ColorType | Callable[..., Vec4T]:
        return Color("white")

    @lazy_property_initializer
    @staticmethod
    def _index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _index_buffer_(
        index_buffer_o: IndexBuffer,
        geometry: Geometry
    ) -> IndexBuffer:
        index_buffer_o.write(geometry._index_)
        return index_buffer_o

    @lazy_property_initializer
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 a_position",
            "vec3 a_normal",
            "vec2 a_uv",
            "vec4 a_color"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        geometry: Geometry,
        #position: Vec3sT,
        #normal: Vec3sT,
        #uv: Vec2sT,
        color: ColorType | Callable[..., Vec4T]
    ) -> AttributesBuffer:
        position = geometry._position_
        normal = geometry._normal_
        uv = geometry._uv_

        if isinstance(color, Callable) and (color_func_params := inspect.signature(color).parameters):
            color_array = np.array([
                color(*args)
                for args in zip(*(
                    {
                        "position": position,
                        "normal": normal,
                        "uv": uv,
                    }[name]
                    for name in color_func_params
                ))
            ])
        else:
            if isinstance(color, Callable):
                pure_color = color()
            else:
                pure_color = MeshMobject._color_to_vector(color)
            color_array = pure_color[None].repeat(len(position), axis=0)

        attributes_o.write({
            "a_position": position,
            "a_normal": normal,
            "a_uv": uv,
            "a_color": color_array
        })
        return attributes_o

    @classmethod
    def _color_to_vector(cls, color: ColorType) -> Vec4T:
        if isinstance(color, Color):
            return np.array([*color.rgb, 1.0])
        if isinstance(color, str):
            if re.fullmatch(r"\w+", color):
                return np.array([*Color(color).rgb, 1.0])
            assert re.fullmatch(r"#[0-9A-F]+", color, flags=re.IGNORECASE)
            hex_len = len(color) - 1
            assert hex_len in (3, 4, 6, 8)
            num_components = 4 if hex_len % 3 else 3
            component_size = hex_len // num_components
            result = np.array([
                int(match_obj.group(), 16)
                for match_obj in re.finditer(rf"[0-9A-F]{{{component_size}}}", color, flags=re.IGNORECASE)
            ]) * (1.0 / (16 ** component_size - 1))
            if num_components == 3:
                result = np.append(result, 1.0)
            return result
        if isinstance(color, np.ndarray):
            if color.shape == (3,):
                return np.array([*color, 1.0], dtype=float)
            if color.shape == (4,):
                return np.array(color, dtype=float)
        raise TypeError

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
            attributes=self._attributes_,
            subroutines={},
            index_buffer=self._index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_,
            mode=moderngl.TRIANGLES
        ))
