from dataclasses import dataclass
from functools import reduce
import itertools as it
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    TypeVar,
    overload
)
import warnings
import weakref

import numpy as np
from scipy.spatial.transform import Rotation

from ..cameras.camera import Camera
from ..constants import (
    ORIGIN,
    PI,
    RIGHT
)
from ..custom_typing import (
    ColorT,
    Mat4T,
    Vec3T,
    Vec3sT,
    Vec4T
)
from ..lazy.lazy import (
    Lazy,
    LazyContainer,
    LazyObject,
    LazyVariableDescriptor,
    LazyWrapper
)
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.color import ColorUtils
from ..utils.space import SpaceUtils


_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_DescriptorGetT = TypeVar("_DescriptorGetT")
_DescriptorSetT = TypeVar("_DescriptorSetT")
_DescriptorRGetT = TypeVar("_DescriptorRGetT")
_MobjectT = TypeVar("_MobjectT", bound="Mobject")


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class BoundingBox:
    maximum: Vec3T
    minimum: Vec3T

    @property
    def origin(self) -> Vec3T:
        return (self.maximum + self.minimum) / 2.0

    @property
    def radius(self) -> Vec3T:
        radius = (self.maximum - self.minimum) / 2.0
        # For zero-width dimensions of radius, thicken a little bit to avoid zero division.
        radius[np.isclose(radius, 0.0)] = 1e-8
        return radius


class Mobject(LazyObject):
    __slots__ = (
        "_parents",
        "_real_ancestors"
    )

    def __init__(self) -> None:
        super().__init__()
        # If `_parents` and `_real_ancestors` are implemented with `LazyDynamicContainer` also,
        # loops will pop up in the DAG of the lazy system.
        self._parents: weakref.WeakSet[Mobject] = weakref.WeakSet()
        self._real_ancestors: weakref.WeakSet[Mobject] = weakref.WeakSet()

    def __iter__(self) -> "Iterator[Mobject]":
        return iter(self._children_)

    @overload
    def __getitem__(
        self,
        index: int
    ) -> "Mobject": ...

    @overload
    def __getitem__(
        self,
        index: slice
    ) -> "list[Mobject]": ...

    def __getitem__(
        self,
        index: int | slice
    ) -> "Mobject | list[Mobject]":
        return self._children_.__getitem__(index)

    # family matters
    # These methods implement a DAG (directed acyclic graph).

    @Lazy.variable_collection
    @classmethod
    def _children_(cls) -> "list[Mobject]":
        return []

    @Lazy.variable_collection
    @classmethod
    def _real_descendants_(cls) -> "list[Mobject]":
        return []

    @classmethod
    def _refresh_families(
        cls,
        *mobjects: "Mobject"
    ) -> None:

        def iter_descendants_by_children(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for child in mobject._children_:
                yield from iter_descendants_by_children(child)

        def iter_ancestors_by_parents(
            mobject: Mobject
        ) -> Iterator[Mobject]:
            yield mobject
            for parent in mobject._parents:
                yield from iter_ancestors_by_parents(parent)

        for ancestor in dict.fromkeys(it.chain.from_iterable(
            iter_ancestors_by_parents(mobject)
            for mobject in mobjects
        )):
            ancestor._real_descendants_.reset(dict.fromkeys(it.chain.from_iterable(
                iter_descendants_by_children(child)
                for child in ancestor._children_
            )))
        for descendant in dict.fromkeys(it.chain.from_iterable(
            iter_descendants_by_children(mobject)
            for mobject in mobjects
        )):
            descendant._real_ancestors.clear()
            descendant._real_ancestors.update(dict.fromkeys(it.chain.from_iterable(
                iter_ancestors_by_parents(parent)
                for parent in descendant._parents
            )))

    def iter_children(self) -> "Iterator[Mobject]":
        yield from self._children_

    def iter_parents(self) -> "Iterator[Mobject]":
        yield from self._parents

    def iter_descendants(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_descendants_

    def iter_ancestors(
        self,
        *,
        broadcast: bool = True
    ) -> "Iterator[Mobject]":
        yield self
        if broadcast:
            yield from self._real_ancestors

    def iter_children_by_type(
        self,
        mobject_type: type[_MobjectT]
    ) -> Iterator[_MobjectT]:
        for mobject in self.iter_children():
            if isinstance(mobject, mobject_type):
                yield mobject

    def iter_descendants_by_type(
        self,
        mobject_type: type[_MobjectT],
        broadcast: bool = True
    ) -> Iterator[_MobjectT]:
        for mobject in self.iter_descendants(broadcast=broadcast):
            if isinstance(mobject, mobject_type):
                yield mobject

    def add(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._children_
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_ancestors()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {invalid_mobjects} to {self}")
        self._children_.extend(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.add(self)
        self._refresh_families(self)
        return self

    def discard(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._children_
        ]
        self._children_.eliminate(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._parents.remove(self)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def added_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject not in self._parents
        ]
        if (invalid_mobjects := [
            mobject for mobject in filtered_mobjects
            if mobject in self.iter_descendants()
        ]):
            raise ValueError(f"Circular relationship occurred when adding {self} to {invalid_mobjects}")
        self._parents.update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.append(self)
        self._refresh_families(self)
        return self

    def discarded_by(
        self,
        *mobjects: "Mobject"
    ):
        filtered_mobjects = [
            mobject for mobject in dict.fromkeys(mobjects)
            if mobject in self._parents
        ]
        self._parents.difference_update(filtered_mobjects)
        for mobject in filtered_mobjects:
            mobject._children_.remove(self)
        self._refresh_families(self, *filtered_mobjects)
        return self

    def clear(self):
        self.discard(*self.iter_children())
        return self

    def copy(self):
        # Copy all descendants. The result is not bound to any mobject.
        result = self._copy()
        real_descendants = list(self._real_descendants_)
        real_descendants_copy = [
            descendant._copy()
            for descendant in real_descendants
        ]
        descendants = [self, *real_descendants]
        descendants_copy = [result, *real_descendants_copy]

        def match_copies(
            mobjects: Iterable[Mobject]
        ) -> Iterator[Mobject]:
            return (
                descendants_copy[descendants.index(mobject)] if mobject in descendants else mobject
                for mobject in mobjects
            )

        result._children_.reset(match_copies(self._children_))
        result._parents.clear()
        for real_descendant, real_descendant_copy in zip(real_descendants, real_descendants_copy, strict=True):
            real_descendant_copy._children_.reset(match_copies(real_descendant._children_))
            real_descendant_copy._parents.clear()
            real_descendant_copy._parents.update(match_copies(real_descendant._parents))

        self._refresh_families(*descendants_copy)
        return result

    def copy_standalone(self):
        # Copy without any real descendant. The result is not bound to any mobject.
        result = self._copy()
        result._children_.clear()
        result._parents.clear()
        self._refresh_families(result)
        return result

    # matrix & transform

    @Lazy.variable_external
    @classmethod
    def _model_matrix_(cls) -> Mat4T:
        return np.identity(4)

    @Lazy.property
    @classmethod
    def _model_uniform_block_buffer_(
        cls,
        model_matrix: Mat4T
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_model",
            fields=[
                "mat4 u_model_matrix"
            ],
            data={
                "u_model_matrix": model_matrix.T
            }
        )

    @Lazy.property_external
    @classmethod
    def _local_sample_points_(cls) -> Vec3sT:
        # Implemented in subclasses.
        return np.zeros((0, 3))

    @Lazy.property_external
    @classmethod
    def _has_local_sample_points_(
        cls,
        local_sample_points: Vec3sT
    ) -> bool:
        return bool(len(local_sample_points))

    @Lazy.property_external
    @classmethod
    def _bounding_box_without_descendants_(
        cls,
        model_matrix: Mat4T,
        local_sample_points: Vec3sT,
        has_local_sample_points: bool
    ) -> BoundingBox | None:
        if not has_local_sample_points:
            return None
        world_sample_points = SpaceUtils.apply_affine(model_matrix, local_sample_points)
        return BoundingBox(
            maximum=world_sample_points.max(axis=0),
            minimum=world_sample_points.min(axis=0)
        )

    @Lazy.property_external
    @classmethod
    def _bounding_box_with_descendants_(
        cls,
        bounding_box_without_descendants: BoundingBox | None,
        real_descendants__bounding_box_without_descendants: list[BoundingBox | None]
    ) -> BoundingBox | None:
        points_array = np.array(list(it.chain.from_iterable(
            (bounding_box.maximum, bounding_box.minimum)
            for bounding_box in (
                bounding_box_without_descendants,
                *real_descendants__bounding_box_without_descendants
            )
            if bounding_box is not None
        )))
        if not len(points_array):
            return None
        return BoundingBox(
            maximum=points_array.max(axis=0),
            minimum=points_array.min(axis=0)
        )

    def get_bounding_box(
        self,
        *,
        broadcast: bool = True
    ) -> BoundingBox:
        if broadcast:
            result = self._bounding_box_with_descendants_.value
        else:
            result = self._bounding_box_without_descendants_.value
        assert result is not None, "Trying to calculate the bounding box of some mobject with no points"
        return result

    def get_bounding_box_size(
        self,
        *,
        broadcast: bool = True
    ) -> Vec3T:
        bounding_box = self.get_bounding_box(broadcast=broadcast)
        return bounding_box.radius * 2.0

    def get_bounding_box_point(
        self,
        direction: Vec3T,
        *,
        broadcast: bool = True
    ) -> Vec3T:
        bounding_box = self.get_bounding_box(broadcast=broadcast)
        return bounding_box.origin + direction * bounding_box.radius

    def get_center(
        self,
        *,
        broadcast: bool = True
    ) -> Vec3T:
        return self.get_bounding_box_point(ORIGIN, broadcast=broadcast)

    @classmethod
    def get_relative_transform_matrix(
        cls,
        matrix: Mat4T,
        about_point: Vec3T
    ) -> Mat4T:
        return reduce(np.ndarray.__matmul__, (
            SpaceUtils.matrix_from_translation(about_point),
            matrix,
            SpaceUtils.matrix_from_translation(-about_point)
        ))

    def apply_transform(
        self,
        matrix: Mat4T,
        *,
        broadcast: bool = True
    ):
        # Avoid redundant caculations.
        transform_dict: dict[LazyWrapper[Mat4T], LazyWrapper[Mat4T]] = {}
        for mobject in self.iter_descendants(broadcast=broadcast):
            original_matrix = mobject._model_matrix_
            if (transformed_matrix := transform_dict.get(original_matrix)) is None:
                transformed_matrix = LazyWrapper(matrix @ original_matrix.value)
                transform_dict[original_matrix] = transformed_matrix
            mobject._model_matrix_ = matrix @ original_matrix.value
        return self

    def apply_relative_transform(
        self,
        matrix: Mat4T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        # Defaults to `about_point=ORIGIN`.
        if about_point is None:
            if about_edge is None:
                about_point = ORIGIN
            else:
                about_point = self.get_bounding_box_point(about_edge, broadcast=broadcast)
        elif about_edge is not None:
            raise AttributeError("Cannot specify both parameters `about_point` and `about_edge`")

        relative_matrix = self.get_relative_transform_matrix(
            matrix=matrix,
            about_point=about_point
        )
        if np.isclose(np.linalg.det(relative_matrix), 0.0):
            warnings.warn("Applying a singular matrix transform")
        self.apply_transform(
            matrix=relative_matrix,
            broadcast=broadcast
        )
        return self

    # shift relatives

    def shift(
        self,
        vector: Vec3T,
        *,
        coor_mask: Vec3T | None = None,
        broadcast: bool = True
    ):
        if coor_mask is not None:
            vector *= coor_mask
        matrix = SpaceUtils.matrix_from_translation(vector)
        # `about_point` and `about_edge` are meaningless when shifting.
        self.apply_relative_transform(
            matrix=matrix,
            broadcast=broadcast
        )
        return self

    def move_to(
        self,
        mobject_or_point: "Mobject | Vec3T",
        aligned_edge: Vec3T = ORIGIN,
        *,
        coor_mask: Vec3T | None = None,
        broadcast: bool = True
    ):
        if isinstance(mobject_or_point, Mobject):
            target_point = mobject_or_point.get_bounding_box_point(aligned_edge, broadcast=broadcast)
        else:
            target_point = mobject_or_point
        point_to_align = self.get_bounding_box_point(aligned_edge, broadcast=broadcast)
        vector = target_point - point_to_align
        self.shift(
            vector=vector,
            coor_mask=coor_mask,
            broadcast=broadcast
        )
        return self

    def center(
        self,
        *,
        coor_mask: Vec3T | None = None,
        broadcast: bool = True
    ):
        self.move_to(
            mobject_or_point=ORIGIN,
            coor_mask=coor_mask,
            broadcast=broadcast
        )
        return self

    def next_to(
        self,
        mobject_or_point: "Mobject | Vec3T",
        direction: Vec3T = RIGHT,
        buff: float = 0.25,
        *,
        coor_mask: Vec3T | None = None,
        broadcast: bool = True
    ):
        if isinstance(mobject_or_point, Mobject):
            target_point = mobject_or_point.get_bounding_box_point(direction, broadcast=broadcast)
        else:
            target_point = mobject_or_point
        point_to_align = self.get_bounding_box_point(-direction, broadcast=broadcast)
        vector = target_point - point_to_align + buff * direction
        self.shift(
            vector=vector,
            coor_mask=coor_mask,
            broadcast=broadcast
        )
        return self

    # scale relatives

    def scale(
        self,
        factor: float | Vec3T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        matrix = SpaceUtils.matrix_from_scale(factor)
        self.apply_relative_transform(
            matrix=matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def scale_about_origin(
        self,
        factor: float | Vec3T,
        *,
        broadcast: bool = True
    ):
        self.scale(
            factor=factor,
            about_point=ORIGIN,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_size(
        self,
        target_size: Vec3T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        factor_vector = target_size / self.get_bounding_box_size(broadcast=broadcast)
        self.scale(
            factor=factor_vector,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_dim(
        self,
        target_length: float,
        dim: int,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        factor_vector = np.ones(3)
        factor_vector[dim] = target_length / self.get_bounding_box_size(broadcast=broadcast)[dim]
        self.scale(
            factor=factor_vector,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_width(
        self,
        target_length: float,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        self.stretch_to_fit_dim(
            target_length=target_length,
            dim=0,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_height(
        self,
        target_length: float,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        self.stretch_to_fit_dim(
            target_length=target_length,
            dim=1,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def stretch_to_fit_depth(
        self,
        target_length: float,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        self.stretch_to_fit_dim(
            target_length=target_length,
            dim=2,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    @classmethod
    def _get_frame_scale_vector(
        cls,
        *,
        original_width: float,
        original_height: float,
        specified_width: float | None,
        specified_height: float | None,
        specified_frame_scale: float | None
    ) -> tuple[float, float]:
        if specified_width is not None and specified_height is not None:
            return specified_width / original_width, specified_height / original_height
        if specified_width is not None and specified_height is None:
            scale_factor = specified_width / original_width
        elif specified_width is None and specified_height is not None:
            scale_factor = specified_height / original_height
        elif specified_width is None and specified_height is None:
            if specified_frame_scale is None:
                scale_factor = 1.0
            else:
                scale_factor = specified_frame_scale
        else:
            raise ValueError  # never
        return scale_factor, scale_factor

    # rotate relatives

    def rotate(
        self,
        rotation: Rotation,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        matrix = SpaceUtils.matrix_from_rotation(rotation)
        self.apply_relative_transform(
            matrix=matrix,
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    def flip(
        self,
        axis: Vec3T,
        *,
        about_point: Vec3T | None = None,
        about_edge: Vec3T | None = None,
        broadcast: bool = True
    ):
        self.rotate(
            rotation=Rotation.from_rotvec(SpaceUtils.normalize(axis) * PI),
            about_point=about_point,
            about_edge=about_edge,
            broadcast=broadcast
        )
        return self

    # color
    # Meanings of `_color_` and `_opacity_` may vary among subclasses,
    # so they may currently be understood as abstract variables.
    # - `MeshMobject`: color of the mesh.
    # - `StrokeMobject`: color of the stroke.
    # - `SceneFrame`: background color of the scene.
    # - `AmbientLight`: color of the ambient light.
    # - `PointLight`: color of the point light.
    # Fading animations control the common `_opacity_` variable.

    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @Lazy.variable_shared
    @classmethod
    def _is_transparent_(cls) -> bool:
        return False

    @Lazy.property_external
    @classmethod
    def _color_vec4_(
        cls,
        color: Vec3T,
        opacity: float
    ) -> Vec4T:
        return np.append(color, opacity)

    @Lazy.property
    @classmethod
    def _color_uniform_block_buffer_(
        cls,
        color_vec4: Vec4T
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_color",
            fields=[
                "vec4 u_color"
            ],
            data={
                "u_color": color_vec4
            }
        )

    def set_color(
        self,
        color: ColorT | None = None,
        *,
        opacity: float | None = None,
        is_transparent: bool | None = None,
        broadcast: bool = True,
        type_filter: "type[Mobject] | None" = None
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = LazyWrapper(color_component) if color_component is not None else None
        opacity_value = LazyWrapper(opacity_component) if opacity_component is not None else None
        is_transparent_value = is_transparent if is_transparent is not None else \
            True if opacity_component is not None else None
        if type_filter is None:
            type_filter = Mobject
        for mobject in self.iter_descendants_by_type(mobject_type=type_filter, broadcast=broadcast):
            if color_value is not None:
                mobject._color_ = color_value
            if opacity_value is not None:
                mobject._opacity_ = opacity_value
            if is_transparent_value is not None:
                mobject._is_transparent_ = is_transparent_value
        return self

    # render

    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:  # Keep updated with `Scene._camera`.
        return Camera()

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        # Implemented in subclasses.
        pass

    # class-level interface

    _descriptor_partial_method: ClassVar[dict[LazyVariableDescriptor, Callable[[Any], Callable[[float, float], Any]]]] = {}
    _descriptor_interpolate_method: ClassVar[dict[LazyVariableDescriptor, Callable[[Any, Any], Callable[[float], Any]]]] = {}
    _descriptor_concatenate_method: ClassVar[dict[LazyVariableDescriptor, Callable[[Iterable[Any]], Any]]] = {}
    _descriptor_related_styles: ClassVar[dict[LazyVariableDescriptor, tuple[tuple[LazyVariableDescriptor, Any], ...]]] = {}

    @classmethod
    def register(
        cls,
        partial_method: Callable[[_DescriptorRGetT], Callable[[float, float], _DescriptorSetT]] | None = None,
        interpolate_method: Callable[[_DescriptorRGetT, _DescriptorRGetT], Callable[[float], _DescriptorSetT]] | None = None,
        concatenate_method: Callable[[Iterable[_DescriptorRGetT]], _DescriptorSetT] | None = None,
        related_styles: tuple[tuple[LazyVariableDescriptor, Any], ...] = ()
    ) -> Callable[
        [LazyVariableDescriptor[_InstanceT, _ContainerT, _DescriptorGetT, _DescriptorSetT, _DescriptorRGetT]],
        LazyVariableDescriptor[_InstanceT, _ContainerT, _DescriptorGetT, _DescriptorSetT, _DescriptorRGetT]
    ]:

        def callback(
            descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DescriptorGetT, _DescriptorSetT, _DescriptorRGetT]
        ) -> LazyVariableDescriptor[_InstanceT, _ContainerT, _DescriptorGetT, _DescriptorSetT, _DescriptorRGetT]:
            if partial_method is not None:
                cls._descriptor_partial_method[descriptor] = partial_method
            if interpolate_method is not None:
                cls._descriptor_interpolate_method[descriptor] = interpolate_method
            if concatenate_method is not None:
                cls._descriptor_concatenate_method[descriptor] = concatenate_method
            if related_styles is not None:
                cls._descriptor_related_styles[descriptor] = related_styles
            return descriptor

        return callback

    @classmethod
    def _partial(
        cls,
        src: _MobjectT
    ) -> Callable[[_MobjectT], Callable[[float, float], None]]:

        def get_descriptor_callback(
            src: _MobjectT,
            descriptor: LazyVariableDescriptor,
            partial_method: Callable[[Any], Callable[[float, float], Any]]
        ) -> Callable[[_MobjectT], Callable[[float, float], None]]:
            partial_callback = partial_method(
                descriptor.converter.convert_rget(descriptor.get_container(src))
            )

            def descriptor_callback(
                dst: _MobjectT
            ) -> Callable[[float, float], None]:

                def descriptor_dst_callback(
                    start: float,
                    stop: float
                ) -> None:
                    new_container = descriptor.converter.convert_set(partial_callback(start, stop))
                    descriptor.set_container(dst, new_container)

                return descriptor_dst_callback

            return descriptor_callback

        descriptor_callbacks = [
            get_descriptor_callback(src, descriptor, partial_method)
            for descriptor, partial_method in cls._descriptor_partial_method.items()
            if descriptor in type(src)._lazy_variable_descriptors
        ]

        def callback(
            dst: _MobjectT
        ) -> Callable[[float, float], None]:
            descriptor_dst_callbacks = [
                descriptor_callback(dst)
                for descriptor_callback in descriptor_callbacks
            ]

            def dst_callback(
                start: float,
                stop: float
            ) -> None:
                for descriptor_dst_callback in descriptor_dst_callbacks:
                    descriptor_dst_callback(start, stop)

            return dst_callback

        return callback

    @classmethod
    def _interpolate(
        cls,
        src_0: _MobjectT,
        src_1: _MobjectT
    ) -> Callable[[_MobjectT], Callable[[float], None]]:

        def get_descriptor_callback(
            src_0: _MobjectT,
            src_1: _MobjectT,
            descriptor: LazyVariableDescriptor,
            interpolate_method: Callable[[Any, Any], Callable[[float], Any]]
        ) -> Callable[[_MobjectT], Callable[[float], None]]:
            interpolate_callback = interpolate_method(
                descriptor.converter.convert_rget(descriptor.get_container(src_0)),
                descriptor.converter.convert_rget(descriptor.get_container(src_1))
            )

            def descriptor_callback(
                dst: _MobjectT
            ) -> Callable[[float], None]:

                def descriptor_dst_callback(
                    alpha: float
                ) -> None:
                    new_container = descriptor.converter.convert_set(interpolate_callback(alpha))
                    descriptor.set_container(dst, new_container)

                return descriptor_dst_callback

            return descriptor_callback

        descriptor_callbacks = [
            get_descriptor_callback(src_0, src_1, descriptor, interpolate_method)
            for descriptor in type(src_0)._lazy_variable_descriptors
            if (interpolate_method := cls._descriptor_interpolate_method.get(descriptor)) is not None
        ]

        def callback(
            dst: _MobjectT
        ) -> Callable[[float], None]:
            descriptor_dst_callbacks = [
                descriptor_callback(dst)
                for descriptor_callback in descriptor_callbacks
            ]

            def dst_callback(
                alpha: float
            ) -> None:
                for descriptor_dst_callback in descriptor_dst_callbacks:
                    descriptor_dst_callback(alpha)

            return dst_callback

        return callback
