__all__ = ["MeshMobject"]


import inspect
import re
from typing import Callable

from colour import Color
import moderngl
import numpy as np

from ..geometries.geometry import Geometry
from ..custom_typing import (
    ColorType,
    Vec2sT,
    Vec3sT,
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
    ContextState,
    Framebuffer,
    IndexBuffer,
    RenderStep,
    Renderable,
    TextureStorage
)
from ..utils.scene_config import SceneConfig


class MeshMobject(Mobject):
    @lazy_property_initializer
    @staticmethod
    def _color_map_texture_() -> moderngl.Texture | None:
        return None

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

    @lazy_property_initializer
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position",
            "vec3 in_normal",
            "vec2 in_uv",
            "vec4 in_color"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        geometry: Geometry,
        color: ColorType | Callable[..., Vec4T]
    ) -> AttributesBuffer:
        position = geometry._position_
        normal = geometry._normal_
        uv = geometry._uv_
        color_array = MeshMobject._calculate_color_array(color, position, normal, uv)
        attributes_o.write({
            "in_position": position,
            "in_normal": normal,
            "in_uv": uv,
            "in_color": color_array
        })
        return attributes_o

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
                scene_config._camera_._ub_camera_,
                self._ub_model_,
                scene_config._ub_lights_
            ],
            subroutines={},
            attributes=self._attributes_,
            index_buffer=self._index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_,
            context_state=ContextState(),
            mode=moderngl.TRIANGLES
        ))

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

    @classmethod
    def _calculate_color_array(
        cls,
        color: ColorType | Callable[..., Vec4T],
        position: Vec3sT,
        normal: Vec3sT,
        uv: Vec2sT
    ) -> Vec4sT:
        if isinstance(color, Callable) and (color_func_params := inspect.signature(color).parameters):
            supported_parameters = {
                "position": position,
                "normal": normal,
                "uv": uv
            }
            return np.array([
                color(*args)
                for args in zip(*(
                    supported_parameters[name]
                    for name in color_func_params
                ), strict=True)
            ])
        if isinstance(color, Callable):
            pure_color = color()
        else:
            pure_color = MeshMobject._color_to_vector(color)
        return pure_color[None].repeat(len(position), axis=0)
