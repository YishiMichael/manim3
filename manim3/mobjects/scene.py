__all__ = [
    "ChildScene",
    "Scene"
]


import time

import moderngl
#from moderngl_window.context.pyglet.window import Window as PygletWindow
import numpy as np

#from ..constants import (
#    PIXEL_HEIGHT,
#    PIXEL_WIDTH
#)
from ..custom_typing import Real
from ..mobjects.mobject import Mobject
#from ..render_passes.copy_pass import CopyPass
#from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import lazy_property
from ..utils.render_procedure import (
    AttributesBuffer,
    #ContextState,
    #Framebuffer,
    IndexBuffer,
    #IntermediateDepthTextures,
    #IntermediateFramebuffer,
    #IntermediateTextures,
    RenderProcedure,
    #RenderStep,
    TextureStorage
)
from ..utils.scene_config import SceneConfig


class ChildScene(Mobject):
    @lazy_property
    @staticmethod
    def _scene_config_() -> SceneConfig:
        return SceneConfig()

    def _update_dt(self, dt: Real):
        for mobject in self.get_descendants_excluding_self():
            mobject._update_dt(dt)
        return self

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    @lazy_property
    @staticmethod
    def _u_accum_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_accum_map")

    @lazy_property
    @staticmethod
    def _u_revealage_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_revealage_map")

    @lazy_property
    @staticmethod
    def _u_depth_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_depth_map")

    @lazy_property
    @staticmethod
    def _attributes_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position",
            "vec2 in_uv"
        ]).write({
            "in_position": np.array([
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]),
            "in_uv": np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ])
        })

    @lazy_property
    @staticmethod
    def _index_buffer_() -> IndexBuffer:
        return IndexBuffer().write(np.array((
            0, 1, 2, 3
        )))

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        #target_framebuffer.clear()
        # Inspired from https://github.com/ambrosiogabe/MathAnimation
        # ./Animations/src/renderer/Renderer.cpp
        opaque_mobjects: list[Mobject] = []
        transparent_mobjects: list[Mobject] = []
        for mobject in self.get_descendants_excluding_self():
            if mobject._apply_oit_:
                transparent_mobjects.append(mobject)
            else:
                opaque_mobjects.append(mobject)

        with RenderProcedure.texture() as opaque_texture, \
                RenderProcedure.texture(dtype="f2") as accum_texture, \
                RenderProcedure.texture(components=1) as revealage_texture, \
                RenderProcedure.depth_texture() as depth_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[opaque_texture],
                    depth_attachment=depth_texture
                ) as opaque_framebuffer, \
                RenderProcedure.framebuffer(
                    color_attachments=[accum_texture],
                    depth_attachment=depth_texture
                ) as accum_framebuffer, \
                RenderProcedure.framebuffer(
                    color_attachments=[revealage_texture],
                    depth_attachment=depth_texture
                ) as revealage_framebuffer:
            #component_texture = IntermediateTextures.fetch()
            #opaque_texture = IntermediateTextures.fetch()
            #accum_texture = IntermediateTextures.fetch(dtype="f2")
            #revealage_texture = IntermediateTextures.fetch(components=1)
            #component_depth_texture = IntermediateDepthTextures.fetch()
            #depth_texture = IntermediateDepthTextures.fetch()
            #component_framebuffer = IntermediateFramebuffer([component_texture], component_depth_texture)

            #opaque_framebuffer = IntermediateFramebuffer([opaque_texture], depth_texture)

            #opaque_framebuffer.depth_mask = True
            #opaque_framebuffer.clear()
            for mobject in opaque_mobjects:
                with RenderProcedure.texture() as component_texture, \
                        RenderProcedure.depth_texture() as component_depth_texture, \
                        RenderProcedure.framebuffer(
                            color_attachments=[component_texture],
                            depth_attachment=component_depth_texture
                        ) as component_framebuffer:
                    #component_framebuffer.depth_mask = True
                    #component_framebuffer.clear()
                    #from PIL import Image
                    #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
                    mobject._render_with_passes(scene_config, component_framebuffer)
                    #import time
                    #time.sleep(0.1)
                    #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
                    #if isinstance(self, Scene):
                    #    from PIL import Image
                    #    #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
                    #    Image.frombytes('RGBA', opaque_texture.size, opaque_texture.read(), 'raw').show()
                    RenderProcedure.render_step(
                        shader_str=RenderProcedure.read_shader("copy"),
                        custom_macros=[],
                        texture_storages=[
                            self._u_color_map_o_.write(
                                np.array(component_texture)
                            ),
                            self._u_depth_map_o_.write(
                                np.array(component_depth_texture)
                            )
                        ],
                        uniform_blocks=[],
                        attributes=self._attributes_,
                        index_buffer=self._index_buffer_,
                        framebuffer=opaque_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ONE, moderngl.ZERO)
                        ),
                        mode=moderngl.TRIANGLE_FAN
                    )
                    #if isinstance(self, Scene):
                    #    from PIL import Image
                    #    #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
                    #    Image.frombytes('RGBA', opaque_texture.size, opaque_texture.read(), 'raw').show()

            # Test against each fragment by the depth buffer, but never write to it.
            # We should prevent from clearing buffer bits.
            #accum_framebuffer = IntermediateFramebuffer([accum_texture], depth_texture)
            accum_framebuffer.depth_mask = False
            #accum_framebuffer.clear()
            #revealage_framebuffer = IntermediateFramebuffer([revealage_texture], depth_texture)
            revealage_framebuffer.depth_mask = False
            revealage_framebuffer.clear(red=1.0)  # initialize `revealage` with 1.0
            for mobject in transparent_mobjects:
                with RenderProcedure.texture() as component_texture, \
                        RenderProcedure.depth_texture() as component_depth_texture, \
                        RenderProcedure.framebuffer(
                            color_attachments=[component_texture],
                            depth_attachment=component_depth_texture
                        ) as component_framebuffer:
                    #component_framebuffer.depth_mask = True
                    #component_framebuffer.clear()
                    mobject._render_with_passes(scene_config, component_framebuffer)
                    u_color_map = self._u_color_map_o_.write(
                        np.array(component_texture)
                    )
                    u_depth_map = self._u_depth_map_o_.write(
                        np.array(component_depth_texture)
                    )
                    RenderProcedure.render_step(
                        shader_str=RenderProcedure.read_shader("oit_accum"),
                        custom_macros=[],
                        texture_storages=[
                            u_color_map,
                            u_depth_map
                        ],
                        uniform_blocks=[],
                        attributes=self._attributes_,
                        index_buffer=self._index_buffer_,
                        framebuffer=accum_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=moderngl.ADDITIVE_BLENDING
                        ),
                        mode=moderngl.TRIANGLE_FAN
                    )
                    RenderProcedure.render_step(
                        shader_str=RenderProcedure.read_shader("oit_revealage"),
                        custom_macros=[],
                        texture_storages=[
                            u_color_map,
                            u_depth_map
                        ],
                        uniform_blocks=[],
                        attributes=self._attributes_,
                        index_buffer=self._index_buffer_,
                        framebuffer=revealage_framebuffer,
                        context_state=RenderProcedure.context_state(
                            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                            blend_func=(moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR)
                        ),
                        mode=moderngl.TRIANGLE_FAN
                    )

            RenderProcedure.render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(opaque_texture)
                    ),
                    self._u_depth_map_o_.write(
                        np.array(depth_texture)
                    )
                ],
                uniform_blocks=[],
                attributes=self._attributes_,
                index_buffer=self._index_buffer_,
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
                    blend_func=(moderngl.ONE, moderngl.ZERO)
                ),
                mode=moderngl.TRIANGLE_FAN
            )
            RenderProcedure.render_step(
                shader_str=RenderProcedure.read_shader("oit_compose"),
                custom_macros=[],
                texture_storages=[
                    self._u_accum_map_o_.write(
                        np.array(accum_texture)
                    ),
                    self._u_revealage_map_o_.write(
                        np.array(revealage_texture)
                    )
                ],
                uniform_blocks=[],
                attributes=self._attributes_,
                index_buffer=self._index_buffer_,
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
                ),
                mode=moderngl.TRIANGLE_FAN
            )

            #if isinstance(self, Scene):
            #from PIL import Image
            #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
            #Image.frombytes('RGB', target_framebuffer.size, target_framebuffer.read(), 'raw').show()

            #component_framebuffer.release()
            #opaque_framebuffer.release()
            #accum_framebuffer.release()
            #revealage_framebuffer.release()


class Scene(ChildScene):
    #def __init__(self):
    #    self._window: PygletWindow = ContextSingleton.get_window()
    #    self._window_framebuffer: moderngl.Framebuffer = ContextSingleton().detect_framebuffer()
    #    self._framebuffer: Framebuffer = IntermediateFramebuffer(
    #        color_attachments=[
    #            ContextSingleton().texture(
    #                size=(PIXEL_WIDTH, PIXEL_HEIGHT),
    #                components=4
    #            )
    #        ],
    #        depth_attachment=ContextSingleton().depth_texture(
    #            size=(PIXEL_WIDTH, PIXEL_HEIGHT)
    #        )
    #    )
    #    super().__init__()

    def _render_scene(self) -> None:
        framebuffer = RenderProcedure._WINDOW_FRAMEBUFFER
        framebuffer.clear()
        self._render_with_passes(self._scene_config_, framebuffer)
        #framebuffer = self._framebuffer
        #framebuffer.clear()
        #self._render_with_passes(self._scene_config_, framebuffer)
        #ContextSingleton().copy_framebuffer(self._window_framebuffer, framebuffer._framebuffer)

    def wait(self, t: Real):
        window = RenderProcedure._WINDOW
        #if window is None:
        #    return self  # TODO
        FPS = 30.0
        dt = 1.0 / FPS
        elapsed_time = 0.0
        timestamp = time.time()
        while not window.is_closing and elapsed_time < t:
            elapsed_time += dt
            delta_t = time.time() - timestamp
            if dt > delta_t:
                time.sleep(dt - delta_t)
            timestamp = time.time()
            window.clear()
            self._update_dt(dt)
            self._render_scene()
            window.swap_buffers()
        return self


#class ChildSceneRenderProcedure(RenderProcedure):

#    @lazy_property
#    @staticmethod
#    def _component_texture_() -> moderngl.Texture:
#        return RenderProcedure.construct_texture()

#    @lazy_property
#    @staticmethod
#    def _opaque_texture_() -> moderngl.Texture:
#        return RenderProcedure.construct_texture()

#    @lazy_property
#    @staticmethod
#    def _accum_texture_() -> moderngl.Texture:
#        return RenderProcedure.construct_texture(dtype="f2")

#    @lazy_property
#    @staticmethod
#    def _revealage_texture_() -> moderngl.Texture:
#        return RenderProcedure.construct_texture(components=1)

#    @lazy_property
#    @staticmethod
#    def _component_depth_texture_() -> moderngl.Texture:
#        return RenderProcedure.construct_depth_texture()

#    @lazy_property
#    @staticmethod
#    def _depth_texture_() -> moderngl.Texture:
#        return RenderProcedure.construct_depth_texture()

#    # TODO: shall these framebuffers be kept alive always?
#    @lazy_property
#    @staticmethod
#    def _component_framebuffer_(
#        component_texture: moderngl.Texture,
#        component_depth_texture: moderngl.Texture
#    ) -> moderngl.Framebuffer:
#        return RenderProcedure.construct_framebuffer(
#            color_attachments=[component_texture],
#            depth_attachment=component_depth_texture
#        )

#    @lazy_property
#    @staticmethod
#    def _opaque_framebuffer_(
#        opaque_texture: moderngl.Texture,
#        depth_texture: moderngl.Texture
#    ) -> moderngl.Framebuffer:
#        return RenderProcedure.construct_framebuffer(
#            color_attachments=[opaque_texture],
#            depth_attachment=depth_texture
#        )

#    @lazy_property
#    @staticmethod
#    def _accum_framebuffer_(
#        accum_texture: moderngl.Texture,
#        depth_texture: moderngl.Texture
#    ) -> moderngl.Framebuffer:
#        return RenderProcedure.construct_framebuffer(
#            color_attachments=[accum_texture],
#            depth_attachment=depth_texture
#        )

#    @lazy_property
#    @staticmethod
#    def _revealage_framebuffer_(
#        revealage_texture: moderngl.Texture,
#        depth_texture: moderngl.Texture
#    ) -> moderngl.Framebuffer:
#        return RenderProcedure.construct_framebuffer(
#            color_attachments=[revealage_texture],
#            depth_attachment=depth_texture
#        )

#    def render(
#        self,
#        child_scene: ChildScene,
#        scene_config: SceneConfig,
#        target_framebuffer: moderngl.Framebuffer
#    ) -> None:
#        #target_framebuffer.clear()
#        # Inspired from https://github.com/ambrosiogabe/MathAnimation
#        # ./Animations/src/renderer/Renderer.cpp
#        opaque_mobjects: list[Mobject] = []
#        transparent_mobjects: list[Mobject] = []
#        for mobject in child_scene.get_descendants_excluding_self():
#            if mobject._apply_oit_:
#                transparent_mobjects.append(mobject)
#            else:
#                opaque_mobjects.append(mobject)

#        #component_texture = IntermediateTextures.fetch()
#        #opaque_texture = IntermediateTextures.fetch()
#        #accum_texture = IntermediateTextures.fetch(dtype="f2")
#        #revealage_texture = IntermediateTextures.fetch(components=1)
#        #component_depth_texture = IntermediateDepthTextures.fetch()
#        #depth_texture = IntermediateDepthTextures.fetch()
#        #component_framebuffer = IntermediateFramebuffer([component_texture], component_depth_texture)

#        #opaque_framebuffer = IntermediateFramebuffer([opaque_texture], depth_texture)
#        component_framebuffer = self._component_framebuffer_
#        opaque_framebuffer = self._opaque_framebuffer_
#        accum_framebuffer = self._accum_framebuffer_
#        revealage_framebuffer = self._revealage_framebuffer_
#        opaque_framebuffer.depth_mask = True
#        opaque_framebuffer.clear()
#        for mobject in opaque_mobjects:
#            component_framebuffer.depth_mask = True
#            component_framebuffer.clear()
#            #from PIL import Image
#            #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
#            mobject._render_with_passes(scene_config, component_framebuffer)
#            #import time
#            #time.sleep(0.1)
#            #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
#            if isinstance(child_scene, Scene):
#                from PIL import Image
#                #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
#                Image.frombytes('RGBA', self._opaque_texture_.size, self._opaque_texture_.read(), 'raw').show()
#            self.render_step(
#                shader_str=self.read_shader("copy"),
#                custom_macros=[],
#                texture_storages=[
#                    self._u_color_map_o_.write(
#                        np.array(self._component_texture_)
#                    ),
#                    self._u_depth_map_o_.write(
#                        np.array(self._component_depth_texture_)
#                    )
#                ],
#                uniform_blocks=[],
#                attributes=self._attributes_,
#                index_buffer=self._index_buffer_,
#                framebuffer=opaque_framebuffer,
#                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
#                context_state=self.context_state(
#                    blend_func=(moderngl.ONE, moderngl.ZERO)
#                ),
#                mode=moderngl.TRIANGLE_FAN
#            )
#            if isinstance(child_scene, Scene):
#                from PIL import Image
#                #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
#                Image.frombytes('RGBA', self._opaque_texture_.size, self._opaque_texture_.read(), 'raw').show()

#        # Test against each fragment by the depth buffer, but never write to it.
#        # We should prevent from clearing buffer bits.
#        #accum_framebuffer = IntermediateFramebuffer([accum_texture], depth_texture)
#        accum_framebuffer.depth_mask = False
#        accum_framebuffer.clear()
#        #revealage_framebuffer = IntermediateFramebuffer([revealage_texture], depth_texture)
#        revealage_framebuffer.depth_mask = False
#        revealage_framebuffer.clear(red=1.0)  # initialize `revealage` with 1.0
#        for mobject in transparent_mobjects:
#            component_framebuffer.depth_mask = True
#            component_framebuffer.clear()
#            mobject._render_with_passes(scene_config, component_framebuffer)
#            u_color_map = self._u_color_map_o_.write(
#                np.array(self._component_texture_)
#            )
#            u_depth_map = self._u_depth_map_o_.write(
#                np.array(self._component_depth_texture_)
#            )
#            self.render_step(
#                shader_str=self.read_shader("oit_accum"),
#                custom_macros=[],
#                texture_storages=[
#                    u_color_map,
#                    u_depth_map
#                ],
#                uniform_blocks=[],
#                attributes=self._attributes_,
#                index_buffer=self._index_buffer_,
#                framebuffer=accum_framebuffer,
#                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
#                context_state=self.context_state(
#                    blend_func=moderngl.ADDITIVE_BLENDING
#                ),
#                mode=moderngl.TRIANGLE_FAN
#            )
#            self.render_step(
#                shader_str=self.read_shader("oit_revealage"),
#                custom_macros=[],
#                texture_storages=[
#                    u_color_map,
#                    u_depth_map
#                ],
#                uniform_blocks=[],
#                attributes=self._attributes_,
#                index_buffer=self._index_buffer_,
#                framebuffer=revealage_framebuffer,
#                enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
#                context_state=self.context_state(
#                    blend_func=(moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR)
#                ),
#                mode=moderngl.TRIANGLE_FAN
#            )

#        self.render_step(
#            shader_str=self.read_shader("copy"),
#            custom_macros=[],
#            texture_storages=[
#                self._u_color_map_o_.write(
#                    np.array(self._opaque_texture_)
#                ),
#                self._u_depth_map_o_.write(
#                    np.array(self._depth_texture_)
#                )
#            ],
#            uniform_blocks=[],
#            attributes=self._attributes_,
#            index_buffer=self._index_buffer_,
#            framebuffer=target_framebuffer,
#            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
#            context_state=self.context_state(
#                blend_func=(moderngl.ONE, moderngl.ZERO)
#            ),
#            mode=moderngl.TRIANGLE_FAN
#        )
#        self.render_step(
#            shader_str=self.read_shader("oit_compose"),
#            custom_macros=[],
#            texture_storages=[
#                self._u_accum_map_o_.write(
#                    np.array(self._accum_texture_)
#                ),
#                self._u_revealage_map_o_.write(
#                    np.array(self._revealage_texture_)
#                )
#            ],
#            uniform_blocks=[],
#            attributes=self._attributes_,
#            index_buffer=self._index_buffer_,
#            framebuffer=target_framebuffer,
#            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
#            context_state=self.context_state(),
#            mode=moderngl.TRIANGLE_FAN
#        )

#        if isinstance(child_scene, Scene):
#            from PIL import Image
#            #Image.frombytes('RGB', component_framebuffer.size, component_framebuffer.read(), 'raw').show()
#            Image.frombytes('RGB', target_framebuffer.size, target_framebuffer.read(), 'raw').show()

#        #IntermediateTextures.restore(component_texture)
#        #IntermediateTextures.restore(opaque_texture)
#        #IntermediateTextures.restore(accum_texture)
#        #IntermediateTextures.restore(revealage_texture)
#        #IntermediateDepthTextures.restore(component_depth_texture)
#        #IntermediateDepthTextures.restore(depth_texture)
#        #component_framebuffer.release()
#        #opaque_framebuffer.release()
#        #accum_framebuffer.release()
#        #revealage_framebuffer.release()


##class FinalSceneRenderProcedure(RenderProcedure):
#    #@lazy_property
#    #@staticmethod
#    #def _color_texture_() -> moderngl.Texture:
#    #    return RenderProcedure.construct_texture()

#    #@lazy_property
#    #@staticmethod
#    #def _depth_texture_() -> moderngl.Texture:
#    #    return RenderProcedure.construct_depth_texture()

#    #@lazy_property
#    #@staticmethod
#    #def _framebuffer_(
#    #    color_texture: moderngl.Texture,
#    #    depth_texture: moderngl.Texture
#    #) -> moderngl.Framebuffer:
#    #    return RenderProcedure.construct_framebuffer(
#    #        color_attachments=[color_texture],
#    #        depth_attachment=depth_texture
#    #    )

#    #def render_scene(self, scene: Scene) -> None:
#    #    framebuffer = self._window_framebuffer_
#    #    framebuffer.clear()
#    #    scene._render_with_passes(scene._scene_config_, framebuffer)
#    #    #ContextSingleton().copy_framebuffer(self._window_framebuffer, framebuffer._framebuffer)
