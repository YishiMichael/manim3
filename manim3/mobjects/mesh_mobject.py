import moderngl
import numpy as np

from ..custom_typing import (
    ColorT,
    Vec3T,
    Vec3sT
)
from ..geometries.geometry import Geometry
from ..lazy.lazy import (
    Lazy,
    LazyWrapper
)
from ..lighting.lighting import Lighting
from ..mobjects.mobject import Mobject
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import (
    TextureIdBuffer,
    UniformBlockBuffer
)
from ..rendering.vertex_array import (
    IndexedAttributesBuffer,
    VertexArray
)
from ..utils.color import ColorUtils


class MeshMobject(Mobject):
    __slots__ = ()

    @Lazy.variable
    @classmethod
    def _geometry_(cls) -> Geometry:
        return Geometry()

    @Lazy.variable_external
    @classmethod
    def _color_map_(cls) -> moderngl.Texture | None:
        return None

    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @Lazy.variable
    @classmethod
    def _lighting_(cls) -> Lighting:
        return Lighting()

    @Lazy.variable_shared
    @classmethod
    def _enable_phong_lighting_(cls) -> bool:
        return True

    @Lazy.property_external
    @classmethod
    def _local_sample_points_(
        cls,
        _geometry_: Geometry
    ) -> Vec3sT:
        return _geometry_._geometry_data_.value.position

    @Lazy.property
    @classmethod
    def _color_uniform_block_buffer_(
        cls,
        color: Vec3T,
        opacity: float
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_color",
            fields=[
                "vec4 u_color"
            ],
            data={
                "u_color": np.append(color, opacity)
            }
        )

    @Lazy.property
    @classmethod
    def _mesh_vertex_array_(
        cls,
        is_transparent: bool,
        enable_phong_lighting: bool,
        color_map: moderngl.Texture | None,
        _camera__camera_uniform_block_buffer_: UniformBlockBuffer,
        _model_uniform_block_buffer_: UniformBlockBuffer,
        _lighting__lighting_uniform_block_buffer_: UniformBlockBuffer,
        _color_uniform_block_buffer_: UniformBlockBuffer,
        _geometry__indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        custom_macros: list[str] = []
        if is_transparent:
            custom_macros.append("#define IS_TRANSPARENT")
        phong_lighting_subroutine = "enable_phong_lighting" if enable_phong_lighting else "disable_phong_lighting"
        custom_macros.append(f"#define phong_lighting_subroutine {phong_lighting_subroutine}")
        return VertexArray(
            shader_filename="mesh",
            custom_macros=custom_macros,
            texture_id_buffers=[
                TextureIdBuffer(
                    field="sampler2D t_color_maps[NUM_COLOR_MAPS]",
                    array_lens={
                        "NUM_COLOR_MAPS": int(color_map is not None)
                    }
                )
            ],
            uniform_block_buffers=[
                _camera__camera_uniform_block_buffer_,
                _model_uniform_block_buffer_,
                _lighting__lighting_uniform_block_buffer_,
                _color_uniform_block_buffer_
            ],
            indexed_attributes_buffer=_geometry__indexed_attributes_buffer_
        )

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        textures: list[moderngl.Texture] = []
        if (color_map := self._color_map_.value) is not None:
            textures.append(color_map)
        self._mesh_vertex_array_.render(
            framebuffer=target_framebuffer,
            texture_array_dict={
                "t_color_maps": np.array(textures, dtype=moderngl.Texture)
            }
        )

    def get_geometry(self) -> Geometry:
        return self._geometry_

    def set_geometry(
        self,
        geometry: Geometry
    ):
        self._geometry_ = geometry
        return self

    @property
    def lighting(self) -> Lighting:
        return self._lighting_

    def set_style(
        self,
        *,
        color: ColorT | None = None,
        opacity: float | None = None,
        is_transparent: bool | None = None,
        lighting: Lighting | None = None,
        enable_phong_lighting: bool | None = None,
        broadcast: bool = True
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = LazyWrapper(color_component) if color_component is not None else None
        opacity_value = LazyWrapper(opacity_component) if opacity_component is not None else None
        is_transparent_value = is_transparent if is_transparent is not None else \
            True if opacity_component is not None else None
        enable_phong_lighting_value = enable_phong_lighting if enable_phong_lighting is not None else \
            True if lighting is not None else None
        for mobject in self.iter_descendants_by_type(mobject_type=MeshMobject, broadcast=broadcast):
            if color_value is not None:
                mobject._color_ = color_value
            if opacity_value is not None:
                mobject._opacity_ = opacity_value
            if is_transparent_value is not None:
                mobject._is_transparent_ = is_transparent_value
            if lighting is not None:
                mobject._lighting_ = lighting
            if enable_phong_lighting_value is not None:
                mobject._enable_phong_lighting_ = enable_phong_lighting_value
        return self
