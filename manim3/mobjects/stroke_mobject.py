import itertools as it
from typing import Callable

import numpy as np

from ..config import Config
from ..constants import PI
from ..custom_typing import (
    NP_f8,
    NP_xf8,
    NP_x2f8,
    NP_3f8,
    NP_x3f8,
    NP_xu4
)
from ..lazy.lazy import Lazy
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import (
    AttributesBuffer,
    IndexBuffer,
    TransformFeedbackBuffer,
    UniformBlockBuffer
)
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import (
    IndexedAttributesBuffer,
    VertexArray
)
from ..shape.shape import MultiLineString
from ..utils.space import SpaceUtils
from .mobject import MobjectStyleMeta
from .renderable_mobject import RenderableMobject


class StrokeMobject(RenderableMobject):
    __slots__ = ()

    def __init__(
        self,
        multi_line_string: MultiLineString | None = None
    ) -> None:
        super().__init__()
        if multi_line_string is not None:
            self._multi_line_string_ = multi_line_string

    @MobjectStyleMeta.register(
        partial_method=MultiLineString.partial,
        interpolate_method=MultiLineString.interpolate,
        concatenate_method=MultiLineString.concatenate
    )
    @Lazy.variable
    @classmethod
    def _multi_line_string_(cls) -> MultiLineString:
        return MultiLineString()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_3f8
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _width_(cls) -> NP_f8:
        return Config().style.stroke_width * np.ones(())

    @MobjectStyleMeta.register()
    @Lazy.variable_hashable
    @classmethod
    def _single_sided_(cls) -> bool:
        return False

    @MobjectStyleMeta.register()
    @Lazy.variable_hashable
    @classmethod
    def _has_linecap_(cls) -> bool:
        return True

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _dilate_(cls) -> NP_f8:
        return np.zeros(())

    @Lazy.property_array
    @classmethod
    def _all_points_(
        cls,
        multi_line_string__line_strings__points: list[NP_x3f8]
    ) -> NP_x3f8:
        if not multi_line_string__line_strings__points:
            return np.zeros((0, 3))
        return np.concatenate(multi_line_string__line_strings__points)

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(
        cls,
        all_points: NP_x3f8
    ) -> NP_x3f8:
        return all_points

    @Lazy.property
    @classmethod
    def _stroke_preprocess_vertex_array_(
        cls,
        all_points: NP_x3f8,
        camera_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer
    ) -> VertexArray:
        indexed_attributes_buffer = IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=[
                    "vec3 in_position"
                ],
                num_vertex=len(all_points),
                data={
                    "in_position": all_points
                }
            ),
            mode=PrimitiveMode.POINTS
        )
        transform_feedback_buffer = TransformFeedbackBuffer(
            fields=[
                "vec3 out_position"
            ],
            num_vertex=len(all_points)
        )
        return VertexArray(
            shader_filename="stroke_preprocess",
            uniform_block_buffers=[
                camera_uniform_block_buffer,
                model_uniform_block_buffer
            ],
            indexed_attributes_buffer=indexed_attributes_buffer,
            transform_feedback_buffer=transform_feedback_buffer
        )

    @Lazy.property_array
    @classmethod
    def _all_position_(
        cls,
        stroke_preprocess_vertex_array: VertexArray
    ) -> NP_x3f8:
        data_dict = stroke_preprocess_vertex_array.transform()
        return data_dict["out_position"]

    @Lazy.property_external
    @classmethod
    def _position_list_(
        cls,
        all_position: NP_x3f8,
        multi_line_string__line_strings__points_len: list[int]
    ) -> list[NP_x3f8]:
        stops = np.array(multi_line_string__line_strings__points_len).cumsum()
        starts = np.roll(stops, 1)
        starts[0] = 0
        return [
            all_position[start:stop]
            for start, stop in zip(starts, stops, strict=True)
        ]

    @Lazy.property_hashable
    @classmethod
    def _winding_sign_(
        cls,
        position_list: list[NP_x3f8]
    ) -> bool:

        def get_signed_area(
            points: NP_x2f8
        ) -> float:
            return float(np.cross(points, np.roll(points, 1, axis=0)).sum()) / 2.0

        area = sum(
            get_signed_area(SpaceUtils.decrease_dimension(position))
            for position in position_list
        )
        return area >= 0.0

    @Lazy.property
    @classmethod
    def _stroke_uniform_block_buffer_(
        cls,
        color: NP_3f8,
        opacity: NP_f8,
        width: NP_f8,
        dilate: NP_f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_stroke",
            fields=[
                "vec4 u_color",
                "float u_width",
                "float u_dilate"
            ],
            data={
                "u_color": np.append(color, opacity),
                "u_width": width,
                "u_dilate": dilate
            }
        )

    @Lazy.property
    @classmethod
    def _winding_sign_uniform_block_buffer_(
        cls,
        winding_sign: bool
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_winding_sign",
            fields=[
                "float u_winding_sign"
            ],
            data={
                "u_winding_sign": np.array(1.0 if winding_sign else -1.0)
            }
        )

    @Lazy.property
    @classmethod
    def _position_attributes_buffer_(
        cls,
        position_list: list[NP_x3f8],
        multi_line_string__line_strings__is_ring: list[bool]
    ) -> AttributesBuffer:

        def get_angles(
            position: NP_x3f8,
            is_ring: bool
        ) -> tuple[NP_xf8, NP_xf8]:
            assert len(position)
            points = SpaceUtils.decrease_dimension(position)
            if is_ring:
                tail_vector = points[0] - points[-1]
            else:
                tail_vector = np.zeros((2,))
            vectors: NP_x2f8 = np.array((tail_vector, *(points[1:] - points[:-1]), tail_vector))
            # Replace zero-length vectors with former or latter ones.
            nonzero_length_indices = SpaceUtils.norm(vectors).nonzero()[0]
            if not len(nonzero_length_indices):
                filled_vectors = np.zeros_like(vectors)
                filled_vectors[:, 0] = 1.0
            else:
                index_increments = np.zeros(len(vectors), dtype=np.int32)
                index_increments[nonzero_length_indices[1:]] = 1
                filled_vectors = vectors[nonzero_length_indices[index_increments.cumsum()]]

            angles = np.arctan2(filled_vectors[:, 1], filled_vectors[:, 0])
            delta_angles = ((angles[1:] - angles[:-1] + PI) % (2.0 * PI) - PI) / 2.0
            direction_angles = angles[:-1] + delta_angles
            return direction_angles, delta_angles

        if not position_list:
            all_position = np.zeros((0, 3))
            direction_angle = np.zeros((0, 1))
            delta_angle = np.zeros((0, 1))
        else:
            direction_angles_tuple, delta_angles_tuple = zip(*(
                get_angles(position, is_ring)
                for position, is_ring in zip(position_list, multi_line_string__line_strings__is_ring, strict=True)
            ), strict=True)
            all_position = np.concatenate(position_list)
            direction_angle = np.concatenate(direction_angles_tuple)
            threshold = PI / 2.0 - 1e-5
            delta_angle = np.concatenate(delta_angles_tuple).clip(-threshold, threshold)
        return AttributesBuffer(
            fields=[
                "vec3 in_position",
                "float in_direction_angle",
                "float in_delta_angle"
            ],
            num_vertex=len(all_position),
            data={
                "in_position": all_position,
                "in_direction_angle": direction_angle,
                "in_delta_angle": delta_angle
            }
        )

    @Lazy.property_collection
    @classmethod
    def _stroke_vertex_arrays_(
        cls,
        multi_line_string__line_strings__points_len: list[int],
        multi_line_string__line_strings__is_ring: list[bool],
        camera_uniform_block_buffer: UniformBlockBuffer,
        stroke_uniform_block_buffer: UniformBlockBuffer,
        winding_sign_uniform_block_buffer: UniformBlockBuffer,
        is_transparent: bool,
        position_attributes_buffer: AttributesBuffer,
        single_sided: bool,
        has_linecap: bool
    ) -> list[VertexArray]:

        def lump_index_from_getter(
            index_getter: Callable[[int, bool], list[int]]
        ) -> NP_xu4:
            if not multi_line_string__line_strings__points_len:
                return np.zeros(0, dtype=np.uint32)
            offsets = np.cumsum((0, *multi_line_string__line_strings__points_len[:-1]))
            return np.concatenate([
                np.array(index_getter(points_len, is_ring), dtype=np.uint32) + offset
                for points_len, is_ring, offset in zip(
                    multi_line_string__line_strings__points_len,
                    multi_line_string__line_strings__is_ring,
                    offsets,
                    strict=True
                )
            ], dtype=np.uint32)

        def line_index_getter(
            points_len: int,
            is_ring: bool
        ) -> list[int]:
            if is_ring:
                # (0, 1, 1, 2, ..., n-2, n-1, n-1, 0)
                return list(it.chain.from_iterable(zip(*(
                    np.roll(range(points_len), -i)
                    for i in range(2)
                ), strict=True)))
            # (0, 1, 1, 2, ..., n-2, n-1)
            return list(it.chain.from_iterable(zip(*(
                range(i, points_len - 1 + i)
                for i in range(2)
            ), strict=True)))

        def join_index_getter(
            points_len: int,
            is_ring: bool
        ) -> list[int]:
            if is_ring:
                return list(range(points_len))
            return list(range(1, points_len - 1))

        def cap_index_getter(
            points_len: int,
            is_ring: bool
        ) -> list[int]:
            if is_ring:
                return []
            return [0, points_len - 1]

        uniform_block_buffers = [
            camera_uniform_block_buffer,
            stroke_uniform_block_buffer,
            winding_sign_uniform_block_buffer
        ]

        def get_vertex_array(
            index_getter: Callable[[int, bool], list[int]],
            mode: PrimitiveMode,
            custom_macros: list[str]
        ) -> VertexArray:
            if is_transparent:
                custom_macros.append("#define IS_TRANSPARENT")
            return VertexArray(
                shader_filename="stroke",
                custom_macros=custom_macros,
                uniform_block_buffers=uniform_block_buffers,
                indexed_attributes_buffer=IndexedAttributesBuffer(
                    attributes_buffer=position_attributes_buffer,
                    index_buffer=IndexBuffer(
                        data=lump_index_from_getter(index_getter)
                    ),
                    mode=mode
                )
            )

        subroutine_name = "single_sided" if single_sided else "both_sided"
        vertex_arrays = [
            get_vertex_array(line_index_getter, PrimitiveMode.LINES, [
                "#define STROKE_LINE",
                f"#define line_subroutine {subroutine_name}"
            ]),
            get_vertex_array(join_index_getter, PrimitiveMode.POINTS, [
                "#define STROKE_JOIN",
                f"#define join_subroutine {subroutine_name}"
            ])
        ]
        if has_linecap and not single_sided:
            vertex_arrays.append(
                get_vertex_array(cap_index_getter, PrimitiveMode.LINES, [
                    "#define STROKE_CAP"
                ])
            )
        return vertex_arrays

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        for vertex_array in self._stroke_vertex_arrays_:
            vertex_array.render(
                framebuffer=target_framebuffer
            )
