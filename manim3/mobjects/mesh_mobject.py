__all__ = ["MeshMobject"]


from typing import Iterable
import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Vec3T,
    Vec3sT
)
from ..geometries.geometry import Geometry
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mobject import Mobject
from ..rendering.glsl_buffers import (
    TextureStorage,
    UniformBlockBuffer
)
from ..rendering.vertex_array import (
    ContextState,
    IndexedAttributesBuffer,
    VertexArray
)
from ..utils.color import ColorUtils
from ..utils.scene_config import SceneConfig


class MeshMobject(Mobject):
    #__slots__ = (
    #    #"_render_samples",
    #    "_apply_phong_lighting",
    #)

    #def __init__(self):
    #    super().__init__()
    #    #self._render_samples: int = 4
    #    self._apply_phong_lighting: bool = True

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _geometry_(cls) -> Geometry:
        return Geometry()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_map_(cls) -> moderngl.Texture | None:
        return None

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _ambient_strength_(cls) -> float:
        return 1.0

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _specular_strength_(cls) -> float:
        return 0.5

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _shininess_(cls) -> float:
        return 32.0

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _apply_phong_lighting_(cls) -> bool:
        return True

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _local_sample_points_(
        cls,
        _geometry_: Geometry
    ) -> Vec3sT:
        return _geometry_._geometry_data_.value.position

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_color_maps_(
        cls,
        color_map: moderngl.Texture | None
    ) -> TextureStorage:
        len_maps = int(color_map is not None)
        #textures = [color_map] if color_map is not None else []
        return TextureStorage(
            field="sampler2D u_color_maps[NUM_U_COLOR_MAPS]",
            shape=(len_maps,),
            dynamic_array_lens={
                "NUM_U_COLOR_MAPS": len_maps
            }
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_material_(
        cls,
        color: Vec3T,
        opacity: float,
        ambient_strength: float,
        specular_strength: float,
        shininess: float
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_material",
            fields=[
                "vec4 u_color",
                "float u_ambient_strength",
                "float u_specular_strength",
                "float u_shininess"
            ],
            data={
                "u_color": np.append(color, opacity),
                "u_ambient_strength": np.array(ambient_strength),
                "u_specular_strength": np.array(specular_strength),
                "u_shininess": np.array(shininess)
            }
        )

    #@lazy_slot
    #@staticmethod
    #def _render_samples() -> int:
    #    return 4

    #@lazy_slot
    #@staticmethod
    #def _apply_phong_lighting() -> bool:
    #    return True

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _vertex_array_(
        cls,
        apply_phong_lighting: bool,
        _u_color_maps_: TextureStorage,
        _scene_config__camera__ub_camera_: UniformBlockBuffer,
        _ub_model_: UniformBlockBuffer,
        _scene_config__ub_lights_: UniformBlockBuffer,
        _ub_material_: UniformBlockBuffer,
        _geometry__indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        custom_macros = []
        if apply_phong_lighting:
            custom_macros.append("#define APPLY_PHONG_LIGHTING")
        return VertexArray(
            shader_filename="mesh",
            custom_macros=custom_macros,
            texture_storages=[
                _u_color_maps_
            ],
            uniform_blocks=[
                _scene_config__camera__ub_camera_,
                _ub_model_,
                _scene_config__ub_lights_,
                _ub_material_
            ],
            indexed_attributes=_geometry__indexed_attributes_buffer_
            #uniform_blocks=[
            #    scene_config._camera_._ub_camera_,
            #    self._ub_model_,
            #    scene_config._ub_lights_,
            #    self._ub_material_
            #],
            #attributes=_geometry_._attributes_,
            #index_buffer=_geometry_._index_buffer_,
            #mode=moderngl.TRIANGLES
        )

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _scene_config_(cls) -> SceneConfig:
        return NotImplemented

    def _render(
        self,
        scene_config: SceneConfig,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        #print(np.frombuffer(self._ub_model_._buffer_.value.read(), dtype=np.float32).reshape((4,4))[3,0])
        #self._u_color_maps_.write(
        #    dynamic_array_lens={
        #        "NUM_U_COLOR_MAPS": len(textures)
        #    },
        #    texture_array=np.array(textures)
        #)
        self._scene_config_ = scene_config
        self._vertex_array_.render(
            #shader_filename="mesh",
            #custom_macros=custom_macros,
            #texture_storages=[
            #    self._u_color_maps_
            #],
            texture_array_dict={
                "u_color_maps": np.array(
                    [color_map]
                    if (color_map := self._color_map_.value) is not None
                    else [],
                    dtype=moderngl.Texture
                )
            },
            #uniform_blocks=[
            #    scene_config._camera_._ub_camera_,
            #    self._ub_model_,
            #    scene_config._ub_lights_,
            #    self._ub_material_
            #],
            framebuffer=target_framebuffer,
            context_state=ContextState(
                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
            )
        )
        #print(self._model_matrix_.value)
        #from PIL import Image
        #Image.frombytes("RGB", target_framebuffer.size, target_framebuffer.read(), "raw").show()

    @classmethod
    def class_set_style(
        cls,
        mobjects: "Iterable[MeshMobject]",
        *,
        color: ColorType | None = None,
        opacity: float | None = None,
        apply_oit: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,
        apply_phong_lighting: bool | None = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = color_component if color_component is not None else None
        opacity_value = opacity_component if opacity_component is not None else None
        apply_oit_value = apply_oit if apply_oit is not None else \
            True if opacity_component is not None else None
        ambient_strength_value = ambient_strength if ambient_strength is not None else None
        specular_strength_value = specular_strength if specular_strength is not None else None
        shininess_value = shininess if shininess is not None else None
        apply_phong_lighting_value = apply_phong_lighting if apply_phong_lighting is not None else \
            True if any(param is not None for param in (
                ambient_strength,
                specular_strength,
                shininess
            )) else None
        for mobject in mobjects:
            if color_value is not None:
                mobject._color_ = color_value
            if opacity_value is not None:
                mobject._opacity_ = opacity_value
            if apply_oit_value is not None:
                mobject._apply_oit_ = apply_oit_value
            if ambient_strength_value is not None:
                mobject._ambient_strength_ = ambient_strength_value
            if specular_strength_value is not None:
                mobject._specular_strength_ = specular_strength_value
            if shininess_value is not None:
                mobject._shininess_ = shininess_value
            if apply_phong_lighting_value is not None:
                mobject._apply_phong_lighting_ = apply_phong_lighting_value

    def set_style(
        self,
        *,
        color: ColorType | None = None,
        opacity: float | None = None,
        apply_oit: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,
        apply_phong_lighting: bool | None = None,
        broadcast: bool = True
    ):
        self.class_set_style(
            mobjects=(
                mobject
                for mobject in self.iter_descendants(broadcast=broadcast)
                if isinstance(mobject, MeshMobject)
            ),
            color=color,
            opacity=opacity,
            apply_oit=apply_oit,
            ambient_strength=ambient_strength,
            specular_strength=specular_strength,
            shininess=shininess,
            apply_phong_lighting=apply_phong_lighting
        )
        return self
