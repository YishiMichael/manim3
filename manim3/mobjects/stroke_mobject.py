__all__ = ["StrokeMobject"]


import itertools as it
from typing import (
    #Callable,
    Callable,
    Generator,
    Iterable
)

import moderngl
import numpy as np

from ..constants import PI
from ..custom_typing import (
    ColorType,
    FloatsT,
    #FloatsT,
    #Mat4T,
    Vec2sT,
    Vec3T,
    Vec3sT,
    VertexIndexType,
    #VertexIndexType
)
from ..lazy.core import (
    LazyDynamicVariableDescriptor,
    LazyWrapper
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..mobjects.mobject import Mobject
from ..rendering.context import (
    ContextState,
    PrimitiveMode
)
from ..rendering.gl_buffer import (
    AttributesBuffer,
    IndexBuffer,
    TransformFeedbackBuffer,
    UniformBlockBuffer
)
from ..rendering.vertex_array import (
    IndexedAttributesBuffer,
    VertexArray
)
from ..utils.color import ColorUtils
from ..utils.shape import MultiLineString
from ..utils.space import SpaceUtils


class StrokeMobject(Mobject):
    __slots__ = ()

    def __init__(
        self,
        multi_line_string: MultiLineString | None = None
    ) -> None:
        super().__init__()
        if multi_line_string is not None:
            self._multi_line_string_ = multi_line_string

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _multi_line_string_(cls) -> MultiLineString:
        return MultiLineString()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _width_(cls) -> float:
        return 0.04

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _single_sided_(cls) -> bool:
        return False

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _has_linecap_(cls) -> bool:
        return True

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _dilate_(cls) -> float:
        return 0.0

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _all_points_(
        cls,
        multi_line_string__line_strings__points: list[Vec3sT]
    ) -> Vec3sT:
        if not multi_line_string__line_strings__points:
            return np.zeros((0, 3))
        return np.concatenate(multi_line_string__line_strings__points)

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _local_sample_points_(
        cls,
        all_points: Vec3sT
    ) -> Vec3sT:
        return all_points

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _all_position_(
        cls,
        all_points: Vec3sT,
        _scene_state__camera__ub_camera_: UniformBlockBuffer,
        _ub_model_: UniformBlockBuffer
    ) -> Vec3sT:
        #if not _multi_line_string_._line_strings_:
        #    position = np.zeros((0, 3))
        #else:
        #    position = np.concatenate([
        #        line_string._points_.value
        #        for line_string in _multi_line_string_._line_strings_
        #    ])

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
            #index_buffer=IndexBuffer(
            #    data=np.arange(len(all_points), dtype=np.uint32)
            #),
            mode=PrimitiveMode.POINTS
        )
        transform_feedback_buffer = TransformFeedbackBuffer(
            fields=[
                "vec3 out_position"
            ],
            num_vertex=len(all_points)
        )
        vertex_array = VertexArray(
            shader_filename="stroke_preprocess",
            uniform_block_buffers=[
                _scene_state__camera__ub_camera_,
                _ub_model_
            ],
            indexed_attributes_buffer=indexed_attributes_buffer,
            transform_feedback_buffer=transform_feedback_buffer
        )
        #print(model_matrix)
        #print(np.frombuffer(_ub_model_.get_buffer().read(), dtype=np.float32))
        data_dict = vertex_array.transform()
        #print(data_dict)
        return data_dict["out_position"]

        #return [get_position(points) for points in multi_line_string__line_strings__points]

    @Lazy.property(LazyMode.UNWRAPPED)
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

    @Lazy.property(LazyMode.SHARED)
    @classmethod
    def _winding_sign_(
        cls,
        position_list: list[Vec3sT],
        width: float
    ) -> bool:

        def get_signed_area(
            points: Vec2sT
        ) -> float:
            return np.cross(points, np.roll(points, 1, axis=0)).sum() / 2.0

        area = sum(
            get_signed_area(SpaceUtils.decrease_dimension(position))
            for position in position_list
        )
        return bool(area * width >= 0.0)

    #@Lazy.property(LazyMode.SHARED)
    #@classmethod
    #def _winding_sign_(
    #    cls,
    #    scene_state__camera__projection_matrix: Mat4T,
    #    scene_state__camera__view_matrix: Mat4T,
    #    model_matrix: Mat4T,
    #    multi_line_string__line_strings__points: list[Vec3sT],
    #    width: float
    #) -> bool:
    #    # TODO: The calculation here is somehow redundant with what shader does...

    #    def get_signed_area(
    #        points: Vec2sT
    #    ) -> float:
    #        return np.cross(points, np.roll(points, -1, axis=0)).sum() / 2.0

    #    transform = scene_state__camera__projection_matrix @ scene_state__camera__view_matrix @ model_matrix
    #    area = sum(
    #        get_signed_area(SpaceUtils.decrease_dimension(SpaceUtils.apply_affine(transform, points)))
    #        for points in multi_line_string__line_strings__points
    #    )
    #    return bool(area * width >= 0.0)

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_stroke_(
        cls,
        width: float,
        color: Vec3T,
        opacity: float,
        dilate: float
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_stroke",
            fields=[
                "float u_width",
                "vec4 u_color",
                "float u_dilate"
            ],
            data={
                "u_width": np.array(abs(width)),
                "u_color": np.append(color, opacity),
                "u_dilate": np.array(dilate)
            }
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _ub_winding_sign_(
        cls,
        #position_list: list[Vec3sT],
        #width: float,
        winding_sign: bool
    ) -> UniformBlockBuffer:

        #def get_winding_sign(
        #    position: Vec3sT
        #) -> float:
        #    points = SpaceUtils.decrease_dimension(position)
        #    signed_area = float(np.cross(points, np.roll(points, 1, axis=0)).sum()) / 2.0
        #    return 1.0 if signed_area * width >= 0.0 else -1.0

        return UniformBlockBuffer(
            name="ub_winding_sign",
            fields=[
                "float u_winding_sign"
            ],
            data={
                "u_winding_sign": np.array(1.0 if winding_sign else -1.0)
            }
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _attributes_buffer_(
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
                #diff = vectors.copy()
                #diff[nonzero_length_indices[1:]] -= diff[nonzero_length_indices[:-1]]
                #filled_vectors = diff.cumsum(axis=0)
                #filled_vectors[:nonzero_length_indices[0]] += diff[nonzero_length_indices[0]]

            angles = np.arctan2(filled_vectors[:, 1], filled_vectors[:, 0])
            delta_angles = ((angles[1:] - angles[:-1] + PI) % (2.0 * PI) - PI) / 2.0
            direction_angles = angles[:-1] + delta_angles
            return direction_angles, delta_angles

        #def get_angles(
        #    position: Vec3sT,
        #    is_ring: bool
        #) -> tuple[FloatsT, FloatsT]:
        #    points = SpaceUtils.decrease_dimension(position)
        #    if not is_ring:
        #        return get_angles(points)
        #    points_extended = np.array((points[-1], *points, points[0]))
        #    direction_angles, delta_angles = get_angles(points_extended)
        #    return direction_angles[1:-1], delta_angles[1:-1]


        #if not _multi_line_string_._line_strings_:
        #    position = np.zeros((0, 3))
        #else:
        #    position = np.concatenate([
        #        line_string._points_.value
        #        for line_string in _multi_line_string_._line_strings_
        #    ])
        if not position_list:
            all_position = np.zeros((0, 3))
            direction_angle = np.zeros((0, 1))
            delta_angle = np.zeros((0, 1))
        else:
            direction_angles_tuple, delta_angles_tuple = zip(*(
                get_angles(position, is_ring)
                for position, is_ring in zip(position_list, multi_line_string__line_strings__is_ring, strict=True)
            ))
            all_position = np.concatenate(position_list)
            direction_angle = np.concatenate(direction_angles_tuple)
            delta_angle = np.concatenate(delta_angles_tuple)
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

    @Lazy.property(LazyMode.COLLECTION)
    @classmethod
    def _vertex_arrays_(
        cls,
        multi_line_string__line_strings__points_len: list[int],
        multi_line_string__line_strings__is_ring: list[bool],
        _scene_state__camera__ub_camera_: UniformBlockBuffer,
        #_ub_model_: UniformBlockBuffer,
        _ub_stroke_: UniformBlockBuffer,
        _ub_winding_sign_: UniformBlockBuffer,
        _attributes_buffer_: AttributesBuffer,
        single_sided: bool,
        has_linecap: bool
    ) -> list[VertexArray]:

        def lump_index_from_getter(
            index_getter: Callable[[int, bool], list[int]]
        ) -> VertexIndexType:
            if not multi_line_string__line_strings__points_len:
                return np.zeros(0, dtype=np.uint32)
            offsets = np.array((0, *multi_line_string__line_strings__points_len[:-1])).cumsum()
            return np.concatenate([
                np.array(index_getter(points_len, is_ring), dtype=np.uint32) + offset
                for points_len, is_ring, offset in zip(
                    multi_line_string__line_strings__points_len,
                    multi_line_string__line_strings__is_ring,
                    offsets,
                    strict=True
                )
            ], dtype=np.uint32)
            #index_arrays: list[VertexIndexType] = []
            #for points_len, is_ring in zip(
            #    multi_line_string__line_strings__points_len,
            #    multi_line_string__line_strings__is_ring,
            #    strict=True
            #):
            #    #points_len = line_string._points_len_.value
            #    #is_ring = line_string._is_ring_.value
            #    index_arrays.append(np.array(index_getter(points_len, is_ring), dtype=np.uint32) + offset)
            #    offset += points_len
            #if not index_arrays:
            #    return np.zeros(0, dtype=np.uint32)
            #return np.concatenate(index_arrays, dtype=np.uint32)

        def line_index_getter(
            points_len: int,
            is_ring: bool
        ) -> list[int]:
            if is_ring:
                # (0, 1, 1, 2, ..., n-2, n-1, n-1, 0)
                return list(it.chain(*zip(*(
                    np.roll(range(points_len), -i)
                    for i in range(2)
                ))))
            # (0, 1, 1, 2, ..., n-2, n-1)
            return list(it.chain(*zip(*(
                range(i, points_len - 1 + i)
                for i in range(2)
            ))))

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
            _scene_state__camera__ub_camera_,
            #_ub_model_,
            _ub_stroke_,
            _ub_winding_sign_
        ]

        def get_vertex_array(
            index_getter: Callable[[int, bool], list[int]],
            mode: PrimitiveMode,
            custom_macros: list[str]
        ) -> VertexArray:
            return VertexArray(
                shader_filename="stroke",
                custom_macros=custom_macros,
                uniform_block_buffers=uniform_block_buffers,
                indexed_attributes_buffer=IndexedAttributesBuffer(
                    attributes_buffer=_attributes_buffer_,
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
                #get_vertex_array(cls._point_index_getter, moderngl.POINTS, [
                #    "#define STROKE_POINT"
                #])
            )
        return vertex_arrays

    #@classmethod
    #def _point_index_getter(
    #    cls,
    #    points_len: int,
    #    is_ring: bool
    #) -> list[int]:
    #    if kind == LineStringKind.POINT:
    #        return [0]
    #    if kind == LineStringKind.LINE_STRING:
    #        return []
    #    if kind == LineStringKind.LINEAR_RING:
    #        return []

    def _render(
        self,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        # TODO: Is this already the best practice?
        # Render color.
        #target_framebuffer.depth_mask = False
        for vertex_array in self._vertex_arrays_:
            vertex_array.render(
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
                    #blend_func=moderngl.ADDITIVE_BLENDING
                    #blend_equation=moderngl.MAX
                )
            )
        #target_framebuffer.depth_mask = True
        # Render depth.
        #target_framebuffer.color_mask = (False, False, False, False)
        #for vertex_array in self._vertex_arrays_:
        #    vertex_array.render(
        #        framebuffer=target_framebuffer,
        #        context_state=ContextState(
        #            enable_only=moderngl.DEPTH_TEST
        #        )
        #    )
        #target_framebuffer.color_mask = (True, True, True, True)

    def iter_stroke_descendants(
        self,
        broadcast: bool = True
    ) -> "Generator[StrokeMobject, None, None]":
        for mobject in self.iter_descendants(broadcast=broadcast):
            if isinstance(mobject, StrokeMobject):
                yield mobject

    @classmethod
    def class_set_style(
        cls,
        mobjects: "Iterable[StrokeMobject]",
        *,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        apply_oit: bool | None = None
    ) -> None:
        width_value = LazyWrapper(width) if width is not None else None
        single_sided_value = LazyWrapper(single_sided) if single_sided is not None else None
        has_linecap_value = LazyWrapper(has_linecap) if has_linecap is not None else None
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = LazyWrapper(color_component) if color_component is not None else None
        opacity_value = LazyWrapper(opacity_component) if opacity_component is not None else None
        dilate_value = LazyWrapper(dilate) if dilate is not None else None
        apply_oit_value = apply_oit if apply_oit is not None else \
            True if any(param is not None for param in (
                opacity_component,
                dilate
            )) else None
        for mobject in mobjects:
            if width_value is not None:
                mobject._width_ = width_value
            if single_sided_value is not None:
                mobject._single_sided_ = single_sided_value
            if has_linecap_value is not None:
                mobject._has_linecap_ = has_linecap_value
            if color_value is not None:
                mobject._color_ = color_value
            if opacity_value is not None:
                mobject._opacity_ = opacity_value
            if dilate_value is not None:
                mobject._dilate_ = dilate_value
            if apply_oit_value is not None:
                mobject._apply_oit_ = apply_oit_value

    def set_style(
        self,
        *,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        color: ColorType | None = None,
        opacity: float | None = None,
        dilate: float | None = None,
        apply_oit: bool | None = None,
        broadcast: bool = True
    ):
        self.class_set_style(
            mobjects=self.iter_stroke_descendants(broadcast=broadcast),
            width=width,
            single_sided=single_sided,
            has_linecap=has_linecap,
            color=color,
            opacity=opacity,
            dilate=dilate,
            apply_oit=apply_oit
        )
        return self

    @classmethod
    def class_concatenate(
        cls,
        *mobjects: "StrokeMobject"
    ) -> "StrokeMobject":
        if not mobjects:
            return StrokeMobject()
        result = mobjects[0]._copy()
        for descriptor in cls._LAZY_VARIABLE_DESCRIPTORS:
            if isinstance(descriptor, LazyDynamicVariableDescriptor):
                continue
            if descriptor is cls._multi_line_string_:
                continue
            assert all(
                descriptor.__get__(result) is descriptor.__get__(mobject)
                for mobject in mobjects
            )
        result._multi_line_string_ = MultiLineString.concatenate(
            mobject._multi_line_string_
            for mobject in mobjects
        )
        return result
