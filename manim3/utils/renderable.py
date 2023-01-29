__all__ = ["Renderable"]


import moderngl

from ..render_passes.render_pass import RenderPass
from ..utils.lazy import (
    LazyBase,
    lazy_property_updatable,
    lazy_property_writable
)
from ..utils.render_procedure import RenderProcedure
from ..utils.scene_config import SceneConfig


class Renderable(LazyBase):
    @lazy_property_writable
    @staticmethod
    def _render_samples_() -> int:
        return 0

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        # Implemented in subclasses
        # This function is not responsible for clearing the `target_framebuffer`.
        # On the other hand, one shall clear the framebuffer before calling this function.
        pass

    def _render_with_samples(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        samples = self._render_samples_
        if not samples:
            self._render(scene_config, target_framebuffer)
            return

        with RenderProcedure.texture(samples=4) as msaa_color_texture, \
                RenderProcedure.depth_texture(samples=4) as msaa_depth_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[msaa_color_texture],
                    depth_attachment=msaa_depth_texture
                ) as msaa_framebuffer:
            self._render(scene_config, msaa_framebuffer)
            RenderProcedure.downsample_framebuffer(msaa_framebuffer, target_framebuffer)

    def _render_with_passes(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render_with_samples(scene_config, target_framebuffer)
            return

        with RenderProcedure.texture() as intermediate_texture_0, \
                RenderProcedure.texture() as intermediate_texture_1:
            textures = (intermediate_texture_0, intermediate_texture_1)
            target_texture_id = 0
            with RenderProcedure.framebuffer(
                        color_attachments=[intermediate_texture_0],
                        depth_attachment=target_framebuffer.depth_attachment
                    ) as initial_framebuffer:
                self._render_with_samples(scene_config, initial_framebuffer)
            for render_pass in render_passes[:-1]:
                target_texture_id = 1 - target_texture_id
                with RenderProcedure.framebuffer(
                            color_attachments=[textures[target_texture_id]],
                            depth_attachment=None
                        ) as intermediate_framebuffer:
                    render_pass._render(
                        texture=textures[1 - target_texture_id],
                        target_framebuffer=intermediate_framebuffer
                    )
            target_framebuffer.depth_mask = False  # TODO: shall we disable writing to depth?
            render_passes[-1]._render(
                texture=textures[target_texture_id],
                target_framebuffer=target_framebuffer
            )
            target_framebuffer.depth_mask = True

    @lazy_property_updatable
    @staticmethod
    def _render_passes_() -> list[RenderPass]:
        return []

    @_render_passes_.updater
    def add_pass(self, *render_passes: RenderPass):
        for render_pass in render_passes:
            self._render_passes_.append(render_pass)
        return self

    @_render_passes_.updater
    def remove_pass(self, *render_passes: RenderPass):
        for render_pass in render_passes:
            self._render_passes_.remove(render_pass)
        return self
