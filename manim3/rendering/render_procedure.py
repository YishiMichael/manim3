__all__ = ["RenderProcedure"]


from dataclasses import dataclass
from functools import reduce
import operator as op
import os
import re

import moderngl
import numpy as np

from ..rendering.config import ConfigSingleton
from ..rendering.context import ContextSingleton
from ..rendering.glsl_variables import (
    AttributesBuffer,
    IndexBuffer,
    TextureStorage,
    UniformBlockBuffer
)
from ..utils.lazy import (
    LazyBase,
    lazy_basedata_cached,
    lazy_property
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ProgramData:
    program: moderngl.Program
    texture_binding_offset_dict: dict[str, int]
    uniform_block_binding_dict: dict[str, int]


class Program(LazyBase):
    __slots__ = ()

    #_CACHE: "ClassVar[dict[bytes, Program]]" = {}
    #_PROGRAM_DATA_CACHE: ClassVar[dict[bytes, NewData[ProgramData]]] = {}

    def __new__(
        cls,
        shader_filename: str,
        custom_macros: list[str],
        dynamic_array_lens: dict[str, int],
        texture_storage_shape_dict: dict[str, tuple[int, ...]]
    ):
        instance = super().__new__(cls)
        instance._shader_filename_ = shader_filename
        instance._custom_macros_ = custom_macros
        instance._dynamic_array_lens_ = dynamic_array_lens
        instance._texture_storage_shape_dict_ = texture_storage_shape_dict
        # TODO: move function to somewhere suitable
        #hash_val = CacheUtils.hash_items(shader_str, tuple(custom_macros), CacheUtils.dict_as_hashable(dynamic_array_lens))
        #cached_instance = cls._CACHE.get(hash_val)
        #if cached_instance is not None:
        #    return cached_instance

        #instance = super().__new__(cls)
        #moderngl_program = cls._construct_moderngl_program(shader_str, custom_macros, dynamic_array_lens)
        #instance._program_ = NewData(moderngl_program)
        #instance._texture_binding_offset_dict_ = NewData(
        #    cls._set_texture_bindings(moderngl_program, texture_storage_shape_dict)
        #)
        #instance._uniform_block_binding_dict_ = NewData(
        #    cls._set_uniform_block_bindings(moderngl_program)
        #)
        #cls._CACHE[hash_val] = instance
        return instance

    @staticmethod
    def __shader_filename_cacher(
        shader_filename: str
    ) -> str:
        return shader_filename

    @lazy_basedata_cached(__shader_filename_cacher)
    @staticmethod
    def _shader_filename_() -> str:
        return NotImplemented

    @staticmethod
    def __custom_macros_cacher(
        custom_macros: list[str]
    ) -> tuple[str, ...]:
        return tuple(custom_macros)

    @lazy_basedata_cached(__custom_macros_cacher)
    @staticmethod
    def _custom_macros_() -> list[str]:
        return NotImplemented

    @staticmethod
    def __dynamic_array_lens_cacher(
        dynamic_array_lens: dict[str, int]
    ) -> tuple[tuple[str, int], ...]:
        return tuple(dynamic_array_lens.items())

    @lazy_basedata_cached(__dynamic_array_lens_cacher)
    @staticmethod
    def _dynamic_array_lens_() -> dict[str, int]:
        return NotImplemented

    @staticmethod
    def __texture_storage_shape_dict_cacher(
        texture_storage_shape_dict: dict[str, tuple[int, ...]]
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple(texture_storage_shape_dict.items())

    @lazy_basedata_cached(__texture_storage_shape_dict_cacher)
    @staticmethod
    def _texture_storage_shape_dict_() -> dict[str, tuple[int, ...]]:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _program_data_(
        shader_filename: str,
        custom_macros: list[str],
        dynamic_array_lens: dict[str, int],
        texture_storage_shape_dict: dict[str, tuple[int, ...]]
    ) -> ProgramData:
        with open(os.path.join(ConfigSingleton().shaders_dir, f"{shader_filename}.glsl")) as shader_file:
            shader_str = shader_file.read()
        program = Program._construct_moderngl_program(shader_str, custom_macros, dynamic_array_lens)
        texture_binding_offset_dict = Program._set_texture_bindings(program, texture_storage_shape_dict)
        uniform_block_binding_dict = Program._set_uniform_block_bindings(program)
        return ProgramData(
            program=program,
            texture_binding_offset_dict=texture_binding_offset_dict,
            uniform_block_binding_dict=uniform_block_binding_dict
        )

    #@lazy_basedata
    #@staticmethod
    #def _program_() -> moderngl.Program:
    #    return NotImplemented

    #@lazy_basedata
    #@staticmethod
    #def _texture_binding_offset_dict_() -> dict[str, int]:
    #    return NotImplemented

    #@lazy_basedata
    #@staticmethod
    #def _uniform_block_binding_dict_() -> dict[str, int]:
    #    return NotImplemented

    @classmethod
    def _construct_moderngl_program(
        cls,
        shader_str: str,
        custom_macros: list[str],
        dynamic_array_lens: dict[str, int]
    ) -> moderngl.Program:
        version_string = f"#version {ContextSingleton().version_code} core"
        array_len_macros = [
            f"#define {array_len_name} {array_len}"
            for array_len_name, array_len in dynamic_array_lens.items()
        ]
        shaders = {
            shader_type: "\n".join([
                version_string,
                "\n",
                f"#define {shader_type}",
                *custom_macros,
                *array_len_macros,
                "\n",
                shader_str
            ])
            for shader_type in (
                "VERTEX_SHADER",
                "FRAGMENT_SHADER",
                "GEOMETRY_SHADER",
                "TESS_CONTROL_SHADER",
                "TESS_EVALUATION_SHADER"
            )
            if re.search(rf"\b{shader_type}\b", shader_str, flags=re.MULTILINE) is not None
        }
        program = ContextSingleton().program(
            vertex_shader=shaders["VERTEX_SHADER"],
            fragment_shader=shaders.get("FRAGMENT_SHADER"),
            geometry_shader=shaders.get("GEOMETRY_SHADER"),
            tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
            tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER"),
        )
        return program

    @classmethod
    def _set_texture_bindings(cls, program: moderngl.Program, texture_storage_shape_dict: dict[str, tuple[int, ...]]) -> dict[str, int]:
        texture_binding_offset_dict: dict[str, int] = {}
        binding_offset = 1
        texture_uniform_match_pattern = re.compile(r"""
            (?P<texture_name>\w+?)
            (?P<multi_index>(\[\d+?\])*)
        """, flags=re.VERBOSE)
        for name in program:
            member = program[name]
            if not isinstance(member, moderngl.Uniform):
                continue
            # Used as a sampler2D
            assert member.dimension == 1
            match_obj = texture_uniform_match_pattern.fullmatch(name)
            assert match_obj is not None
            texture_storage_name = match_obj.group("texture_name")
            texture_storage_shape = texture_storage_shape_dict[texture_storage_name]
            if texture_storage_name not in texture_binding_offset_dict:
                texture_binding_offset_dict[texture_storage_name] = binding_offset
                binding_offset += cls._int_prod(texture_storage_shape)
            multi_index = tuple(
                int(index_match.group(1))
                for index_match in re.finditer(r"\[(\d+?)\]", match_obj.group("multi_index"))
            )
            if not texture_storage_shape:
                assert not multi_index
                uniform_size = 1
                local_offset = 0
            else:
                assert len(multi_index) == len(texture_storage_shape) - 1
                uniform_size = texture_storage_shape[-1]
                local_offset = np.ravel_multi_index(multi_index, texture_storage_shape[:-1]) * uniform_size if multi_index else 0
            assert member.array_length == uniform_size
            offset = texture_binding_offset_dict[texture_storage_name] + local_offset
            member.value = offset if uniform_size == 1 else list(range(offset, offset + uniform_size))
        return texture_binding_offset_dict

    @classmethod
    def _set_uniform_block_bindings(cls, program: moderngl.Program) -> dict[str, int]:
        uniform_block_binding_dict: dict[str, int] = {}
        binding = 0
        for name in program:
            member = program[name]
            if not isinstance(member, moderngl.UniformBlock):
                continue
            # Ensure the name doesn't contain wierd symbols like `[]`
            assert re.fullmatch(r"\w+", name) is not None
            uniform_block_binding_dict[name] = binding
            member.binding = binding
            binding += 1
        return uniform_block_binding_dict

    @classmethod
    def _int_prod(cls, shape: tuple[int, ...]) -> int:
        return reduce(op.mul, shape, 1)  # TODO: redundant with the one in glsl_variables.py


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ContextState:
    enable_only: int
    depth_func: str
    blend_func: tuple[int, int] | tuple[int, int, int, int]
    blend_equation: int | tuple[int, int]
    front_face: str
    cull_face: str
    wireframe: bool


class RenderProcedure(LazyBase):
    __slots__ = ()

    #_SHADER_STRS: ClassVar[dict[str, str]] = {}

    def __new__(cls):
        raise NotImplementedError

    def __init_subclass__(cls) -> None:
        raise NotImplementedError

    #@classmethod
    #def read_shader(cls, filename: str) -> str:
    #    if (content := cls._SHADER_STRS.get(filename)) is not None:
    #        return content
    #    with open(os.path.join(ConfigSingleton().shaders_dir, f"{filename}.glsl")) as shader_file:
    #        content = shader_file.read()
    #    cls._SHADER_STRS[filename] = content
    #    return content

    @classmethod
    def context_state(
        cls,
        *,
        enable_only: int,
        depth_func: str = "<",
        blend_func: tuple[int, int] | tuple[int, int, int, int] = moderngl.DEFAULT_BLENDING,
        blend_equation: int | tuple[int, int] = moderngl.FUNC_ADD,
        front_face: str = "ccw",
        cull_face: str = "back",
        wireframe: bool = False
    ) -> ContextState:
        return ContextState(
            enable_only=enable_only,
            depth_func=depth_func,
            blend_func=blend_func,
            blend_equation=blend_equation,
            front_face=front_face,
            cull_face=cull_face,
            wireframe=wireframe
        )

    #@classmethod
    #def texture(
    #    cls,
    #    *,
    #    size: tuple[int, int] | None = None,
    #    components: int = 4,
    #    samples: int = 0,
    #    dtype: str = "f1"
    #) -> IntermediateTexture:
    #    if size is None:
    #        size = ConfigSingleton().pixel_size
    #    return IntermediateTexture(
    #        size=size,
    #        components=components,
    #        samples=samples,
    #        dtype=dtype
    #    )

    #@classmethod
    #def depth_texture(
    #    cls,
    #    *,
    #    size: tuple[int, int] | None = None,
    #    samples: int = 0
    #) -> IntermediateDepthTexture:
    #    if size is None:
    #        size = ConfigSingleton().pixel_size
    #    return IntermediateDepthTexture(
    #        size=size,
    #        samples=samples
    #    )

    #@classmethod
    #def framebuffer(
    #    cls,
    #    *,
    #    color_attachments: list[moderngl.Texture | moderngl.Renderbuffer],
    #    depth_attachment: moderngl.Texture | moderngl.Renderbuffer | None
    #) -> IntermediateFramebuffer:
    #    return IntermediateFramebuffer(
    #        color_attachments=color_attachments,
    #        depth_attachment=depth_attachment
    #    )

    @classmethod
    def downsample_framebuffer(cls, src: moderngl.Framebuffer, dst: moderngl.Framebuffer) -> None:
        ContextSingleton().copy_framebuffer(dst=dst, src=src)

    @classmethod
    def render_step(
        cls,
        *,
        shader_filename: str,
        custom_macros: list[str],
        texture_storages: list[TextureStorage],
        uniform_blocks: list[UniformBlockBuffer],
        attributes: AttributesBuffer,
        index_buffer: IndexBuffer,
        framebuffer: moderngl.Framebuffer,
        context_state: ContextState,
        mode: int
    ) -> None:
        #import pprint
        #pprint.pprint(attributes.__class__._field_info_.instance_to_basedata_dict)
        #print(attributes._is_empty_)
        if attributes._is_empty_ or index_buffer._is_empty_:
            return

        dynamic_array_lens: dict[str, int] = {}
        for texture_storage in texture_storages:
            dynamic_array_lens.update(texture_storage._dynamic_array_lens_)
        for uniform_block in uniform_blocks:
            dynamic_array_lens.update(uniform_block._dynamic_array_lens_)
        dynamic_array_lens.update(attributes._dynamic_array_lens_)
        filtered_array_lens = {
            array_len_name: array_len
            for array_len_name, array_len in dynamic_array_lens.items()
            if not re.fullmatch(r"__\w+__", array_len_name)
        }

        context = ContextSingleton()
        program_data = Program(
            shader_filename=shader_filename,
            custom_macros=custom_macros,
            dynamic_array_lens=filtered_array_lens,
            texture_storage_shape_dict={
                texture_storage._field_name_: texture_storage._texture_array.shape
                for texture_storage in texture_storages
            }
        )._program_data_
        program = program_data.program

        ## Remove redundancies
        #textures: list[moderngl.Texture] = list(dict.fromkeys(
        #    texture for texture_storage in texture_storages
        #    for texture in texture_storage._texture_list_
        #))

        # texture storages
        texture_storage_dict = {
            texture_storage._field_name_: texture_storage
            for texture_storage in texture_storages
        }
        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_storage_name, binding_offset in program_data.texture_binding_offset_dict.items():
            texture_storage = texture_storage_dict[texture_storage_name]
            assert not texture_storage._is_empty_
            texture_bindings.extend(
                (texture, binding)
                for binding, texture in enumerate(texture_storage._texture_array.flat, start=binding_offset)
            )

        # uniform blocks
        uniform_block_dict = {
            uniform_block._field_name_: uniform_block
            for uniform_block in uniform_blocks
        }
        uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
        for uniform_block_name, binding in program_data.uniform_block_binding_dict.items():
            uniform_block = uniform_block_dict[uniform_block_name]
            assert not uniform_block._is_empty_
            program_uniform_block = program[uniform_block_name]
            assert isinstance(program_uniform_block, moderngl.UniformBlock)
            uniform_block._validate(program_uniform_block)
            uniform_block_bindings.append((uniform_block._buffer_, binding))

        # subroutines
        #subroutine_indices: list[int] = [
        #    program._subroutines_[subroutines[subroutine_name]].index
        #    for subroutine_name in program._program_.subroutines
        #]

        # attributes
        program_attributes = {
            name: member
            for name in program
            if isinstance(member := program[name], moderngl.Attribute)
        }
        attributes._validate(program_attributes)
        buffer_format, attribute_names = attributes._get_buffer_format(set(program_attributes))
        vertex_array = context.vertex_array(
            program=program,
            content=[(attributes._buffer_, buffer_format, *attribute_names)],
            index_buffer=index_buffer._buffer_,
            mode=mode
        )

        context.depth_func = context_state.depth_func
        context.blend_func = context_state.blend_func
        context.blend_equation = context_state.blend_equation
        context.front_face = context_state.front_face
        context.cull_face = context_state.cull_face
        context.wireframe = context_state.wireframe
        with context.scope(
            framebuffer=framebuffer,
            enable_only=context_state.enable_only,
            textures=tuple(texture_bindings),
            uniform_buffers=tuple(uniform_block_bindings)
        ):
            #vertex_array.subroutines = tuple(subroutine_indices)
            vertex_array.render()

    # TODO
    #_FULLSCREEN_ATTRIBUTES: ClassVar[AttributesBuffer | None] = None
    #_FULLSCREEN_INDEX_BUFFER: ClassVar[IndexBuffer | None] = None

    #@classmethod
    #def fullscreen_render_step(
    #    cls,
    #    *,
    #    shader_filename: str,
    #    custom_macros: list[str],
    #    texture_storages: list[TextureStorage],
    #    uniform_blocks: list[UniformBlockBuffer],
    #    framebuffer: moderngl.Framebuffer,
    #    context_state: ContextState
    #) -> None:
    #    if cls._FULLSCREEN_ATTRIBUTES is None:
    #        cls._FULLSCREEN_ATTRIBUTES = AttributesBuffer(
    #            fields=[
    #                "vec3 in_position",
    #                "vec2 in_uv"
    #            ],
    #            num_vertex=4,
    #            data={
    #                "in_position": np.array((
    #                    [-1.0, -1.0, 0.0],
    #                    [1.0, -1.0, 0.0],
    #                    [1.0, 1.0, 0.0],
    #                    [-1.0, 1.0, 0.0],
    #                )),
    #                "in_uv": np.array((
    #                    [0.0, 0.0],
    #                    [1.0, 0.0],
    #                    [1.0, 1.0],
    #                    [0.0, 1.0],
    #                ))
    #            }
    #        )
    #    if cls._FULLSCREEN_INDEX_BUFFER is None:
    #        cls._FULLSCREEN_INDEX_BUFFER = IndexBuffer(
    #            data=np.array((
    #                0, 1, 2, 3
    #            ))
    #        )
    #    cls.render_step(
    #        shader_filename=shader_filename,
    #        custom_macros=custom_macros,
    #        texture_storages=texture_storages,
    #        uniform_blocks=uniform_blocks,
    #        attributes=cls._FULLSCREEN_ATTRIBUTES,
    #        index_buffer=cls._FULLSCREEN_INDEX_BUFFER,
    #        framebuffer=framebuffer,
    #        context_state=context_state,
    #        mode=moderngl.TRIANGLE_FAN
    #    )
