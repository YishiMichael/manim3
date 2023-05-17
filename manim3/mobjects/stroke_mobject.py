import itertools as it
from typing import Callable

import numpy as np

from ..constants import PI
from ..custom_typing import (
    FloatsT,
    Vec2sT,
    Vec3T,
    Vec3sT,
    VertexIndexT
)
from ..lazy.lazy import Lazy
from ..mobjects.mobject import MobjectMeta
from ..mobjects.renderable_mobject import RenderableMobject
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


class StrokeMobject(RenderableMobject):
    __slots__ = ()

    def __init__(
        self,
        multi_line_string: MultiLineString | None = None
    ) -> None:
        super().__init__()
        if multi_line_string is not None:
            self._multi_line_string_ = multi_line_string

    @MobjectMeta.register(
        partial_method=MultiLineString.partial,
        interpolate_method=MultiLineString.interpolate,
        concatenate_method=MultiLineString.concatenate
    )
    @Lazy.variable
    @classmethod
    def _multi_line_string_(cls) -> MultiLineString:
        return MultiLineString()

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_vec3
    )
    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((RenderableMobject._is_transparent_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float
    )
    @Lazy.variable_external
    @classmethod
    def _width_(cls) -> float:
        return 0.04  # TODO: check if the auto-scaling remains

    @MobjectMeta.register(
        interpolate_method=NotImplemented
    )
    @Lazy.variable_external
    @classmethod
    def _single_sided_(cls) -> bool:
        return False

    @MobjectMeta.register(
        interpolate_method=NotImplemented
    )
    @Lazy.variable_external
    @classmethod
    def _has_linecap_(cls) -> bool:
        return True

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((RenderableMobject._is_transparent_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _dilate_(cls) -> float:
        return 0.0

    @Lazy.property_external
    @classmethod
    def _all_points_(
        cls,
        multi_line_string__line_strings__points: list[Vec3sT]
    ) -> Vec3sT:
        if not multi_line_string__line_strings__points:
            return np.zeros((0, 3))
        return np.concatenate(multi_line_string__line_strings__points)

    @Lazy.property_external
    @classmethod
    def _local_sample_points_(
        cls,
        all_points: Vec3sT
    ) -> Vec3sT:
        return all_points

    @Lazy.property
    @classmethod
    def _stroke_preprocess_vertex_array_(
        cls,
        all_points: Vec3sT,
        _camera__camera_uniform_block_buffer_: UniformBlockBuffer,
        _model_uniform_block_buffer_: UniformBlockBuffer
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
                _camera__camera_uniform_block_buffer_,
                _model_uniform_block_buffer_
            ],
            indexed_attributes_buffer=indexed_attributes_buffer,
            transform_feedback_buffer=transform_feedback_buffer
        )

    @Lazy.property_external
    @classmethod
    def _all_position_(
        cls,
        _stroke_preprocess_vertex_array_: VertexArray
    ) -> Vec3sT:
        data_dict = _stroke_preprocess_vertex_array_.transform()
        return data_dict["out_position"]

    @Lazy.property_external
    @classmethod
    def _position_list_(
        cls,
        all_position: Vec3sT,
        multi_line_string__line_strings__points_len: list[int]
    ) -> list[Vec3sT]:
        stops = np.array(multi_line_string__line_strings__points_len).cumsum()
        starts = np.roll(stops, 1)
        starts[0] = 0
        return [
            all_position[start:stop]
            for start, stop in zip(starts, stops, strict=True)
        ]

    @Lazy.property_shared
    @classmethod
    def _winding_sign_(
        cls,
        position_list: list[Vec3sT]
    ) -> bool:

        def get_signed_area(
            points: Vec2sT
        ) -> float:
            return np.cross(points, np.roll(points, 1, axis=0)).sum() / 2.0

        area = sum(
            get_signed_area(SpaceUtils.decrease_dimension(position))
            for position in position_list
        )
        return area >= 0.0

    @Lazy.property
    @classmethod
    def _stroke_uniform_block_buffer_(
        cls,
        color: Vec3T,
        opacity: float,
        width: float,
        dilate: float
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
                "u_width": np.array(width),
                "u_dilate": np.array(dilate)
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
        position_list: list[Vec3sT],
        multi_line_string__line_strings__is_ring: list[bool]
    ) -> AttributesBuffer:

        def get_angles(
            position: Vec3sT,
            is_ring: bool
        ) -> tuple[FloatsT, FloatsT]:
            assert len(position)
            points = SpaceUtils.decrease_dimension(position)
            if is_ring:
                tail_vector = points[0] - points[-1]
            else:
                tail_vector = np.zeros(2)
            vectors: Vec2sT = np.array((tail_vector, *(points[1:] - points[:-1]), tail_vector))
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
        width: float,
        multi_line_string__line_strings__points_len: list[int],
        multi_line_string__line_strings__is_ring: list[bool],
        _camera__camera_uniform_block_buffer_: UniformBlockBuffer,
        _stroke_uniform_block_buffer_: UniformBlockBuffer,
        _winding_sign_uniform_block_buffer_: UniformBlockBuffer,
        is_transparent: bool,
        _position_attributes_buffer_: AttributesBuffer,
        single_sided: bool,
        has_linecap: bool
    ) -> list[VertexArray]:
        if np.isclose(width, 0.0):
            return []

        def lump_index_from_getter(
            index_getter: Callable[[int, bool], list[int]]
        ) -> VertexIndexT:
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
            _camera__camera_uniform_block_buffer_,
            _stroke_uniform_block_buffer_,
            _winding_sign_uniform_block_buffer_
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
                    attributes_buffer=_position_attributes_buffer_,
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

    @property
    def multi_line_string(self) -> MultiLineString:
        return self._multi_line_string_

    @property
    def color(self) -> Vec3T:
        return self._color_.value

    @property
    def opacity(self) -> float:
        return self._opacity_.value

    @property
    def width(self) -> float:
        return self._width_.value

    @property
    def single_sided(self) -> bool:
        return self._single_sided_.value

    @property
    def has_linecap(self) -> bool:
        return self._has_linecap_.value

    @property
    def dilate(self) -> float:
        return self._dilate_.value
