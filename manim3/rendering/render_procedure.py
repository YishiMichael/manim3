#__all__ = ["RenderProcedure"]


#from dataclasses import dataclass

#import moderngl

#from ..rendering.context import ContextSingleton
#from ..rendering.glsl_variables import (
#    TextureStorage,
#    UniformBlockBuffer
#)
#from ..rendering.vertex_array import VertexArray




#class RenderProcedure:
#    __slots__ = ()

#    def __new__(cls):
#        raise NotImplementedError

#    def __init_subclass__(cls) -> None:
#        raise NotImplementedError

#    @classmethod
#    def context_state(
#        cls,
#        *,
#        enable_only: int,
#        depth_func: str = "<",
#        blend_func: tuple[int, int] | tuple[int, int, int, int] = moderngl.DEFAULT_BLENDING,
#        blend_equation: int | tuple[int, int] = moderngl.FUNC_ADD,
#        front_face: str = "ccw",
#        cull_face: str = "back",
#        wireframe: bool = False
#    ) -> ContextState:
#        return ContextState(
#            enable_only=enable_only,
#            depth_func=depth_func,
#            blend_func=blend_func,
#            blend_equation=blend_equation,
#            front_face=front_face,
#            cull_face=cull_face,
#            wireframe=wireframe
#        )

#    @classmethod
#    def downsample_framebuffer(cls, src: moderngl.Framebuffer, dst: moderngl.Framebuffer) -> None:
#        ContextSingleton().copy_framebuffer(dst=dst, src=src)

#    #@classmethod
#    #def get_dynamic_array_lens(
#    #    cls,
#    #    *,
#    #    texture_storages: list[TextureStorage],
#    #    uniform_blocks: list[UniformBlockBuffer],
#    #    attributes: AttributesBuffer
#    #) -> dict[str, int]:
#    #    dynamic_array_lens: dict[str, int] = {}
#    #    for texture_storage in texture_storages:
#    #        dynamic_array_lens.update(texture_storage._dynamic_array_lens_)
#    #    for uniform_block in uniform_blocks:
#    #        dynamic_array_lens.update(uniform_block._dynamic_array_lens_)
#    #    dynamic_array_lens.update(attributes._dynamic_array_lens_)
#    #    return {
#    #        array_len_name: array_len
#    #        for array_len_name, array_len in dynamic_array_lens.items()
#    #        if not re.fullmatch(r"__\w+__", array_len_name)
#    #    }

#    #@classmethod
#    #def get_texture_storage_shape_dict(
#    #    cls,
#    #    *,
#    #    texture_storages: list[TextureStorage]
#    #) -> dict[str, tuple[int, ...]]:
#    #    return {
#    #        texture_storage._field_name_: texture_storage._texture_array.shape
#    #        for texture_storage in texture_storages
#    #    }

#    @classmethod
#    def render_step(
#        cls,
#        *,
#        #shader_filename: str,
#        #custom_macros: list[str],
#        vertex_array: VertexArray,
#        texture_storages: list[TextureStorage],
#        uniform_blocks: list[UniformBlockBuffer],
#        #attributes: AttributesBuffer,
#        #index_buffer: IndexBuffer,
#        framebuffer: moderngl.Framebuffer,
#        context_state: ContextState
#        #mode: int
#    ) -> None:
#        #if attributes._is_empty_ or index_buffer._is_empty_:
#        #    return
#        if vertex_array._vertex_array_ is None:
#            return

#        #dynamic_array_lens: dict[str, int] = {}
#        #for texture_storage in texture_storages:
#        #    dynamic_array_lens.update(texture_storage._dynamic_array_lens_)
#        #for uniform_block in uniform_blocks:
#        #    dynamic_array_lens.update(uniform_block._dynamic_array_lens_)
#        #dynamic_array_lens.update(attributes._dynamic_array_lens_)
#        #filtered_array_lens = {
#        #    array_len_name: array_len
#        #    for array_len_name, array_len in dynamic_array_lens.items()
#        #    if not re.fullmatch(r"__\w+__", array_len_name)
#        #}

#        #program_data = Program(
#        #    shader_filename=shader_filename,
#        #    custom_macros=custom_macros,
#        #    dynamic_array_lens=filtered_array_lens,
#        #    texture_storage_shape_dict={
#        #        texture_storage._field_name_: texture_storage._texture_array.shape
#        #        for texture_storage in texture_storages
#        #    }
#        #)._program_data_
#        #program = program_data.program
#        program_data = vertex_array._program_._program_data_
#        program = program_data.program

#        # texture storages
#        texture_storage_dict = {
#            texture_storage._field_name_: texture_storage
#            for texture_storage in texture_storages
#        }
#        texture_bindings: list[tuple[moderngl.Texture, int]] = []
#        for texture_storage_name, binding_offset in program_data.texture_binding_offset_dict.items():
#            texture_storage = texture_storage_dict[texture_storage_name]
#            assert not texture_storage._is_empty_
#            texture_bindings.extend(
#                (texture, binding)
#                for binding, texture in enumerate(texture_storage._texture_array.flat, start=binding_offset)
#            )

#        # uniform blocks
#        uniform_block_dict = {
#            uniform_block._field_name_: uniform_block
#            for uniform_block in uniform_blocks
#        }
#        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
#        for uniform_block_name, binding in program_data.uniform_block_binding_dict.items():
#            uniform_block = uniform_block_dict[uniform_block_name]
#            assert not uniform_block._is_empty_
#            program_uniform_block = program[uniform_block_name]
#            assert isinstance(program_uniform_block, moderngl.UniformBlock)
#            uniform_block._validate(program_uniform_block)
#            uniform_block_bindings.append((uniform_block._buffer_, binding))

#        # attributes
#        program_attributes = {
#            name: member
#            for name in program
#            if isinstance(member := program[name], moderngl.Attribute)
#        }
#        attributes._validate(program_attributes)
#        buffer_format, attribute_names = attributes._get_buffer_format(set(program_attributes))
#        vertex_array = context.vertex_array(
#            program=program,
#            content=[(attributes._buffer_, buffer_format, *attribute_names)],
#            index_buffer=index_buffer._buffer_,
#            mode=mode
#        )

#        context = ContextSingleton()
#        context.depth_func = context_state.depth_func
#        context.blend_func = context_state.blend_func
#        context.blend_equation = context_state.blend_equation
#        context.front_face = context_state.front_face
#        context.cull_face = context_state.cull_face
#        context.wireframe = context_state.wireframe
#        with context.scope(
#            framebuffer=framebuffer,
#            enable_only=context_state.enable_only,
#            textures=tuple(texture_bindings),
#            uniform_buffers=tuple(uniform_block_bindings)
#        ):
#            vertex_array._vertex_array_.render()
