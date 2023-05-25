import moderngl
import numpy as np

from ..cameras.camera import Camera
from ..cameras.perspective_camera import PerspectiveCamera
from ..custom_typing import Vec3T
from ..lazy.lazy import Lazy
from ..lighting.ambient_light import AmbientLight
from ..lighting.lighting import Lighting
from ..lighting.point_light import PointLight
from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.mobject import (
    Mobject,
    MobjectStyleMeta
)
from ..mobjects.renderable_mobject import RenderableMobject
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
from ..utils.space import SpaceUtils


class SceneFrame(Mobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        # Should be standalone for each `SceneFrame` object.
        self._camera_ = PerspectiveCamera()
        self._lighting_ = Lighting()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_vec3
    )
    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.zeros(3)

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_float
    )
    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 0.0

    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:
        return NotImplemented

    @Lazy.variable
    @classmethod
    def _lighting_(cls) -> Lighting:
        return NotImplemented

    @Lazy.variable_collection
    @classmethod
    def _render_passes_(cls) -> list[RenderPass]:
        return []

    @Lazy.property
    @classmethod
    def _copy_vertex_array_(cls) -> VertexArray:
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

    def _render_scene_content(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        camera = self._camera_
        for mobject in self.iter_descendants_by_type(mobject_type=RenderableMobject):
            mobject._camera_uniform_block_buffer_ = camera._camera_uniform_block_buffer_

        lighting = self._lighting_
        lighting._ambient_lights_.reset(self.iter_descendants_by_type(mobject_type=AmbientLight))
        lighting._point_lights_.reset(self.iter_descendants_by_type(mobject_type=PointLight))
        for mobject in self.iter_descendants_by_type(mobject_type=MeshMobject):
            mobject._lighting_uniform_block_buffer_ = lighting._lighting_uniform_block_buffer_

        # Inspired from `https://github.com/ambrosiogabe/MathAnimation`
        # `./Animations/src/renderer/Renderer.cpp`.
        opaque_mobjects: list[RenderableMobject] = []
        transparent_mobjects: list[RenderableMobject] = []
        for mobject in self.iter_descendants_by_type(RenderableMobject):
            if not mobject._has_local_sample_points_:
                continue
            if mobject._is_transparent_:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        with TextureFactory.depth_texture() as depth_texture:
            opaque_framebuffer = OpaqueFramebuffer(
                color_texture=target_framebuffer.color_texture,
                depth_texture=depth_texture
            )
            opaque_framebuffer.framebuffer.clear(
                color=(*self._color_, self._opacity_)
            )
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

    def _render_scene(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render_scene_content(target_framebuffer)
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
            self._render_scene_content(framebuffers[0])
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

    def _render_to_window(
        self,
        color_texture: moderngl.Texture
    ) -> None:
        window = Context.window
        if window.is_closing:
            raise KeyboardInterrupt
        window.clear()
        self._copy_vertex_array_.render(
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

    #@property
    #def color(self) -> Vec3T:
    #    return self._color_

    #@property
    #def opacity(self) -> float:
    #    return self._opacity_
