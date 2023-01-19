__all__ = ["StrokeMobject"]


import itertools as it

from colour import Color
import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Real
)
from ..mobjects.mobject import Mobject
from ..mobjects.mesh_mobject import MeshMobject
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
    UniformBlockBuffer
)
from ..utils.scene_config import SceneConfig
from ..utils.shape import MultiLineString3D


class StrokeMobject(Mobject):
    def __init__(self, multi_line_string: MultiLineString3D):
        super().__init__()
        self._multi_line_string_ = multi_line_string

    @lazy_property_initializer_writable
    @staticmethod
    def _multi_line_string_() -> MultiLineString3D:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _width_() -> Real:
        return 0.04

    @lazy_property_initializer_writable
    @staticmethod
    def _color_() -> ColorType:
        return Color("white")

    @lazy_property_initializer_writable
    @staticmethod
    def _dilate_() -> Real:
        return 0.0

    @lazy_property_initializer_writable
    @staticmethod
    def _single_sided_() -> bool:
        return False

    @lazy_property_initializer
    @staticmethod
    def _ub_stroke_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_stroke", [
            "float u_stroke_width",
            "vec4 u_stroke_color",
            "float u_stroke_dilate"
        ])

    @lazy_property
    @staticmethod
    def _ub_stroke_(
        ub_stroke_o: UniformBlockBuffer,
        width: Real,
        color: ColorType,
        dilate: Real
    ) -> UniformBlockBuffer:
        ub_stroke_o.write({
            "u_stroke_width": np.array(width),
            "u_stroke_color": MeshMobject._color_to_vector(color),
            "u_stroke_dilate": np.array(dilate)
        })
        return ub_stroke_o

    @lazy_property_initializer
    @staticmethod
    def _attributes_o_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position"
        ])

    @lazy_property
    @staticmethod
    def _attributes_(
        attributes_o: AttributesBuffer,
        multi_line_string: MultiLineString3D
    ) -> AttributesBuffer:
        if not multi_line_string._children_:
            position = np.zeros((0, 3))
        else:
            position = np.concatenate([
                line_string._coords_[:-1]
                if line_string._kind_ == "linear_ring"
                else line_string._coords_
                for line_string in multi_line_string._children_
                if line_string._kind_ != "point"
            ])
        attributes_o.write({
            "in_position": position
        })
        return attributes_o

    @lazy_property_initializer
    @staticmethod
    def _line_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _line_index_buffer_(
        line_index_buffer_o: IndexBuffer,
        multi_line_string: MultiLineString3D
    ) -> IndexBuffer:
        index_list: list[int] = []
        offset = 0
        for line_string in multi_line_string._children_:
            if line_string._kind_ == "point":
                continue
            n_points = len(line_string._coords_)
            if line_string._kind_ == "linear_ring":
                n_points -= 1
            index_list.extend(range(offset, offset + n_points))
            if line_string._kind_ == "linear_ring":
                index_list.append(offset)
            offset += n_points
        line_index_buffer_o.write(np.array(index_list))
        return line_index_buffer_o

    @lazy_property_initializer
    @staticmethod
    def _join_index_buffer_o_() -> IndexBuffer:
        return IndexBuffer()

    @lazy_property
    @staticmethod
    def _join_index_buffer_(
        join_index_buffer_o: IndexBuffer,
        multi_line_string: MultiLineString3D
    ) -> IndexBuffer:
        index_list: list[int] = []
        offset = 0
        for line_string in multi_line_string._children_:
            if line_string._kind_ == "point":
                continue
            n_points = len(line_string._coords_)
            if line_string._kind_ == "linear_ring":
                n_points -= 1
            index_list.extend(it.chain(*(
                range(offset + i, offset + i + 3)
                for i in range(0, n_points - 2)
            )))
            if line_string._kind_ == "linear_ring":
                index_list.extend(
                    offset + i
                    for i in (n_points - 2, n_points - 1, 0, n_points - 1, 0, 1)
                )
            offset += n_points
        join_index_buffer_o.write(np.array(index_list))
        return join_index_buffer_o

    #@lazy_property
    #@staticmethod
    #def _copy_pass_() -> CopyPass:
    #    return CopyPass()

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
        #with IntermediateTextures.register_n(1) as textures:
        #    with IntermediateDepthTextures.register_n(1) as depth_textures:
        #        intermediate_framebuffer = IntermediateFramebuffer(textures, depth_textures[0])
        #        #from PIL import Image
        #        #print("*"*30)
        #        #print(intermediate_framebuffer._framebuffer)
        #        #print(intermediate_framebuffer._color_attachments, intermediate_framebuffer._depth_attachment)
        #        #print(target_framebuffer._framebuffer)
        #        #print()
        #        #Image.frombytes('RGB', intermediate_framebuffer._framebuffer.size, intermediate_framebuffer._framebuffer.read(), 'raw').show()
        #        intermediate_framebuffer._framebuffer.clear()  # ???
        #        ##Image.frombytes('RGB', intermediate_framebuffer._framebuffer.size, intermediate_framebuffer._framebuffer.read(), 'raw').show()
        self._render_by_step(RenderStep(
            shader_str=Renderable._read_shader("stroke_line"),
            texture_storages=[],
            uniform_blocks=[
                scene_config._camera_._ub_camera_,
                self._ub_model_,
                self._ub_stroke_
            ],
            subroutines={
                "line_dilate_func": "single_sided_dilate" if self._single_sided_ else "both_sided_dilate"
            },
            attributes=self._attributes_,
            index_buffer=self._line_index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
            context_state=ContextState(
                blend_func=moderngl.ADDITIVE_BLENDING,
                blend_equation=moderngl.MAX
            ),
            mode=moderngl.LINE_STRIP
        ), RenderStep(
            shader_str=Renderable._read_shader("stroke_join"),
            texture_storages=[],
            uniform_blocks=[
                scene_config._camera_._ub_camera_,
                self._ub_model_,
                self._ub_stroke_
            ],
            subroutines={
                "join_dilate_func": "single_sided_dilate" if self._single_sided_ else "both_sided_dilate"
            },
            attributes=self._attributes_,
            index_buffer=self._join_index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST,
            context_state=ContextState(
                blend_func=moderngl.ADDITIVE_BLENDING,
                blend_equation=moderngl.MAX
            ),
            mode=moderngl.TRIANGLES
        ))
        #from PIL import Image
        #Image.frombytes('RGB', target_framebuffer._framebuffer.size, target_framebuffer._framebuffer.read(), 'raw').show()

        #color_vals = np.frombuffer(target_framebuffer.get_attachment(0).read(), dtype=np.uint32)
        #print(color_vals.min(), color_vals.max())
        #depth_vals = np.frombuffer(target_framebuffer.get_attachment(-1).read(), dtype=np.float32)
        #print(depth_vals.min(), depth_vals.max())
        #        #Image.frombytes('RGB', intermediate_framebuffer._framebuffer.size, intermediate_framebuffer._framebuffer.read(), 'raw').show()
        #        #Image.frombytes('RGB', target_framebuffer._framebuffer.size, target_framebuffer._framebuffer.read(), 'raw').show()
        #        CopyPass()._render(
        #            input_framebuffer=intermediate_framebuffer,
        #            output_framebuffer=target_framebuffer,
        #            mobject=self,
        #            scene_config=scene_config
        #        )
        #        #Image.frombytes('RGB', target_framebuffer._framebuffer.size, target_framebuffer._framebuffer.read(), 'raw').show()
        #        intermediate_framebuffer.release()
