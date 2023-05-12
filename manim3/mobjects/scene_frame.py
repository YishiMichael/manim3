import moderngl
import numpy as np

from ..lazy.lazy import Lazy
from ..mobjects.mobject import Mobject
from ..passes.render_pass import RenderPass
from ..rendering.context import (
    Context,
    ContextState
)
from ..rendering.framebuffer import (
    ColorFramebuffer,
    Framebuffer,
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import TextureIdBuffer
from ..rendering.mgl_enums import ContextFlag
from ..rendering.texture import TextureFactory
from ..rendering.vertex_array import VertexArray
#from ..utils.color import ColorUtils


class SceneFrame(Mobject):
    __slots__ = ()

    #@Lazy.variable
    #@classmethod
    #def _camera_(cls) -> Camera:
    #    return PerspectiveCamera()

    #@Lazy.variable_external
    #@classmethod
    #def _background_color_(cls) -> Vec3T:
    #    return np.zeros(3)

    #@Lazy.variable_external
    #@classmethod
    #def _background_opacity_(cls) -> float:
    #    return 0.0

    @Lazy.variable_collection
    @classmethod
    def _render_passes_(cls) -> list[RenderPass]:
        return []

    @Lazy.property
    @classmethod
    def _pixelated_vertex_array_(cls) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            texture_id_buffers=[
                TextureIdBuffer(
                    field="sampler2D t_color_map"
                )
            ]
        )

    @Lazy.property
    @classmethod
    def _oit_compose_vertex_array_(cls) -> VertexArray:
        return VertexArray(
            shader_filename="oit_compose",
            texture_id_buffers=[
                TextureIdBuffer(
                    field="sampler2D t_accum_map"
                ),
                TextureIdBuffer(
                    field="sampler2D t_revealage_map"
                )
            ]
        )

    def _render_scene(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        # Inspired from `https://github.com/ambrosiogabe/MathAnimation`
        # `./Animations/src/renderer/Renderer.cpp`.
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.iter_descendants():
            if not mobject._has_local_sample_points_.value:
                continue
            if mobject._is_transparent_.value:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        with TextureFactory.depth_texture() as depth_texture:
            opaque_framebuffer = OpaqueFramebuffer(
                color_texture=target_framebuffer.color_texture,
                depth_texture=depth_texture
            )

            red, green, blue = self._color_.value
            alpha = self._opacity_.value
            opaque_framebuffer.framebuffer.clear(red=red, green=green, blue=blue, alpha=alpha)
            for mobject in opaque_mobjects:
                mobject._render(opaque_framebuffer)

            with TextureFactory.texture(dtype="f2") as accum_texture, \
                    TextureFactory.texture(components=1) as revealage_texture:
                transparent_framebuffer = TransparentFramebuffer(
                    accum_texture=accum_texture,
                    revealage_texture=revealage_texture,
                    depth_texture=depth_texture
                )
                # Test against each fragment by the depth buffer, but never write to it.
                transparent_framebuffer.framebuffer.depth_mask = False
                transparent_framebuffer.framebuffer.clear()
                # Initialize `revealage` with 1.0.
                # TODO: There should be a more elegant way using `clear`.
                revealage_texture.write(b"\xff" * (revealage_texture.width * revealage_texture.height))
                for mobject in transparent_mobjects:
                    mobject._render(transparent_framebuffer)

            self._oit_compose_vertex_array_.render(
                framebuffer=target_framebuffer,
                texture_array_dict={
                    "t_accum_map": np.array(accum_texture),
                    "t_revealage_map": np.array(revealage_texture)
                },
                context_state=ContextState(
                    flags=(ContextFlag.BLEND,)
                )
            )

    def _render_scene_with_passes(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render_scene(target_framebuffer)
            return

        with TextureFactory.texture() as texture_0, TextureFactory.texture() as texture_1:
            framebuffers = (
                ColorFramebuffer(
                    color_texture=texture_0
                ),
                ColorFramebuffer(
                    color_texture=texture_1
                )
            )
            target_id = 0
            self._render_scene(framebuffers[0])
            for render_pass in render_passes[:-1]:
                target_id = 1 - target_id
                render_pass._render(
                    texture=framebuffers[1 - target_id].color_texture,
                    target_framebuffer=framebuffers[target_id]
                )
            render_passes[-1]._render(
                texture=framebuffers[target_id].color_texture,
                target_framebuffer=target_framebuffer
            )

    #def _render_to_texture(
    #    self,
    #    color_texture: moderngl.Texture,
    #    camera: Camera,
    #    lighting: Lighting
    #) -> None:
    #    for mobject in self.iter_descendants_by_type(mobject_type=Mobject):
    #        mobject._camera_ = camera
    #    for mobject in self.iter_descendants_by_type(mobject_type=MeshMobject):
    #        mobject._lighting_ = lighting
    #    framebuffer = ColorFramebuffer(
    #        color_texture=color_texture
    #    )
    #    self._render_scene_with_passes(framebuffer)

    def _render_to_window(
        self,
        color_texture: moderngl.Texture
    ) -> None:
        window = Context.window
        if window.is_closing:
            raise KeyboardInterrupt
        window.clear()
        self._pixelated_vertex_array_.render(
            framebuffer=Framebuffer(
                framebuffer=Context.window_framebuffer,
                default_context_state=ContextState(
                    flags=()
                )
            ),
            texture_array_dict={
                "t_color_map": np.array(color_texture)
            }
        )
        window.swap_buffers()

    #def set_background(
    #    self,
    #    color: ColorT | None = None,
    #    *,
    #    opacity: float | None = None
    #):
    #    color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
    #    if color_component is not None:
    #        self._background_color_ = color_component
    #    if opacity_component is not None:
    #        self._background_opacity_ = opacity_component
    #    return self

    #@property
    #def render_passes(self) -> LazyDynamicContainer[RenderPass]:
    #    return self._render_passes_
