__all__ = ["GaussianBlurPass"]


import moderngl
import numpy as np

from ..custom_typing import (
    FloatsT,
    Real
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
from ..rendering.framebuffer_batches import ColorFramebufferBatch
from ..rendering.glsl_buffers import (
    UniformBlockBuffer,
    TextureStorage
)
from ..rendering.vertex_array import (
    ContextState,
    IndexedAttributesBuffer,
    VertexArray
)


class GaussianBlurPass(RenderPass):
    __slots__ = ()

    def __init__(
        self,
        sigma_width: Real | None = None
    ) -> None:
        super().__init__()
        if sigma_width is not None:
            self._sigma_width_ = sigma_width

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _sigma_width_(cls) -> Real:
        return 0.1

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_map_(cls) -> moderngl.Texture:
        return NotImplemented

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _convolution_core_(
        cls,
        sigma_width: Real
    ) -> FloatsT:
        sigma = sigma_width * ConfigSingleton().pixel_per_unit
        n = int(np.ceil(3.0 * sigma))
        convolution_core = np.exp(-np.arange(n + 1) ** 2 / (2.0 * sigma ** 2))
        return convolution_core / (2.0 * convolution_core.sum() - convolution_core[0])

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_color_map_(
        cls,
        color_map: moderngl.Texture
    ) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_color_map",
            texture_array=np.array(color_map)
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_gaussian_blur_(
        cls,
        convolution_core: FloatsT
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_gaussian_blur",
            fields=[
                "vec2 u_uv_offset",
                "float u_convolution_core[CONVOLUTION_CORE_SIZE]"
            ],
            dynamic_array_lens={
                "CONVOLUTION_CORE_SIZE": len(convolution_core)
            },
            data={
                "u_uv_offset": 1.0 / np.array(ConfigSingleton().pixel_size),
                "u_convolution_core": convolution_core
            }
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _horizontal_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _ub_gaussian_blur_: UniformBlockBuffer,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="gaussian_blur",
            custom_macros=[
                f"#define blur_subroutine horizontal_dilate"
            ],
            texture_storages=[
                _u_color_map_
            ],
            uniform_blocks=[
                _ub_gaussian_blur_
            ],
            indexed_attributes=_indexed_attributes_buffer_
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _vertical_vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _ub_gaussian_blur_: UniformBlockBuffer,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="gaussian_blur",
            custom_macros=[
                f"#define blur_subroutine vertical_dilate"
            ],
            texture_storages=[
                _u_color_map_
            ],
            uniform_blocks=[
                _ub_gaussian_blur_
            ],
            indexed_attributes=_indexed_attributes_buffer_
            #attributes=_attributes_buffer_,
            #index_buffer=_index_buffer_,
            #mode=moderngl.TRIANGLE_FAN
        )

    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        with ColorFramebufferBatch() as batch:
            self._color_map_ = texture
            self._horizontal_vertex_array_.render(
                #shader_filename="gaussian_blur",
                #custom_macros=[
                #    f"#define blur_subroutine horizontal_dilate"
                #],
                #texture_storages=[
                #    self._u_color_map_
                #],
                #uniform_blocks=[
                #    self._ub_gaussian_blur_
                #],
                framebuffer=batch.framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
            self._color_map_ = batch.color_texture
            self._vertical_vertex_array_.render(
                #shader_filename="gaussian_blur",
                #custom_macros=[
                #    f"#define blur_subroutine vertical_dilate"
                #],
                #texture_storages=[
                #    self._u_color_map_
                #],
                #uniform_blocks=[
                #    self._ub_gaussian_blur_
                #],
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
