__all__ = ["MeshMobject"]


import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Real,
    Vec3T,
    Vec3sT
)
from ..geometries.geometry import Geometry
from ..mobjects.mobject import Mobject
from ..rendering.glsl_buffers import (
    TextureStorage,
    UniformBlockBuffer
)
from ..rendering.vertex_array import ContextState
from ..scenes.scene_config import SceneConfig
from ..utils.color import ColorUtils
from ..utils.lazy import (
    LazyWrapper,
    lazy_object,
    lazy_object_raw,
    lazy_property
)


class MeshMobject(Mobject):
    #__slots__ = (
    #    #"_render_samples",
    #    "_apply_phong_lighting",
    #)

    #def __init__(self):
    #    super().__init__()
    #    #self._render_samples: int = 4
    #    self._apply_phong_lighting: bool = True

    @lazy_object
    @staticmethod
    def _geometry_() -> Geometry:
        return Geometry()

    @lazy_object_raw
    @staticmethod
    def _color_map_() -> moderngl.Texture | None:
        return None

    @lazy_object_raw
    @staticmethod
    def _color_() -> Vec3T:
        return np.ones(3)

    @lazy_object_raw
    @staticmethod
    def _opacity_() -> Real:
        return 1.0

    @lazy_object_raw
    @staticmethod
    def _ambient_strength_() -> Real:
        return 1.0

    @lazy_object_raw
    @staticmethod
    def _specular_strength_() -> Real:
        return 0.5

    @lazy_object_raw
    @staticmethod
    def _shininess_() -> Real:
        return 32.0

    @lazy_object_raw
    @staticmethod
    def _apply_phong_lighting_() -> bool:
        return True

    @lazy_property
    @staticmethod
    def _local_sample_points_(
        _geometry_: Geometry
    ) -> LazyWrapper[Vec3sT]:
        return LazyWrapper(_geometry_._geometry_data_.value.position)

    @lazy_property
    @staticmethod
    def _u_color_maps_(
        color_map: moderngl.Texture | None
    ) -> TextureStorage:
        textures = [color_map] if color_map is not None else []
        return TextureStorage(
            field="sampler2D u_color_maps[NUM_U_COLOR_MAPS]",
            dynamic_array_lens={
                "NUM_U_COLOR_MAPS": len(textures)
            },
            texture_array=np.array(textures)
        )

    @lazy_property
    @staticmethod
    def _ub_material_(
        color: Vec3T,
        opacity: Real,
        ambient_strength: Real,
        specular_strength: Real,
        shininess: Real
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

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        custom_macros = []
        if self._apply_phong_lighting_.value:
            custom_macros.append("#define APPLY_PHONG_LIGHTING")
        self._geometry_._vertex_array_.render(
            shader_filename="mesh",
            custom_macros=custom_macros,
            texture_storages=[
                self._u_color_maps_
            ],
            uniform_blocks=[
                scene_config._camera_._ub_camera_,
                self._ub_model_,
                scene_config._ub_lights_,
                self._ub_material_
            ],
            framebuffer=target_framebuffer,
            context_state=ContextState(
                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
            )
        )

    def set_style(
        self,
        *,
        color: ColorType | None = None,
        opacity: Real | None = None,
        apply_oit: bool | None = None,
        ambient_strength: Real | None = None,
        specular_strength: Real | None = None,
        shininess: Real | None = None,
        apply_phong_lighting: bool | None = None,
        broadcast: bool = True
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = LazyWrapper(color_component) if color_component is not None else None
        opacity_value = LazyWrapper(opacity_component) if opacity_component is not None else None
        apply_oit_value = LazyWrapper(apply_oit) if apply_oit is not None else \
            LazyWrapper(True) if opacity_component is not None else None
        ambient_strength_value = LazyWrapper(ambient_strength) if ambient_strength is not None else None
        specular_strength_value = LazyWrapper(specular_strength) if specular_strength is not None else None
        shininess_value = LazyWrapper(shininess) if shininess is not None else None
        apply_phong_lighting_value = LazyWrapper(apply_phong_lighting) if apply_phong_lighting is not None else \
            LazyWrapper(True) if any(param is not None for param in (
                ambient_strength,
                specular_strength,
                shininess
            )) else None
        for mobject in self.iter_descendants(broadcast=broadcast):
            if not isinstance(mobject, MeshMobject):
                continue
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
        return self
