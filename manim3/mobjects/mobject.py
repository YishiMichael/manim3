from dataclasses import dataclass
from functools import reduce
import itertools as it
import operator as op
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    TypeVar,
    ParamSpec,
    overload
)
import warnings
import weakref
import moderngl

import numpy as np
from scipy.spatial.transform import Rotation
from shapely import Geometry

from ..constants import (
    ORIGIN,
    PI
)
from ..config import ConfigSingleton
from ..custom_typing import (
    ColorT,
    Mat4T,
    Vec3T,
    Vec3sT
)
from ..lazy.lazy import (
    Lazy,
    LazyContainer,
    LazyObject,
    LazyVariableDescriptor,
    LazyWrapper
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..shape.line_string import MultiLineString
from ..shape.shape import Shape
from ..utils.color import ColorUtils
from ..utils.space import SpaceUtils


_MobjectT = TypeVar("_MobjectT", bound="Mobject")
_ContainerT = TypeVar("_ContainerT", bound="LazyContainer")
_InstanceT = TypeVar("_InstanceT", bound="LazyObject")
_DescriptorGetT = TypeVar("_DescriptorGetT")
_DescriptorSetT = TypeVar("_DescriptorSetT")
_DescriptorRawT = TypeVar("_DescriptorRawT")
_MethodParams = ParamSpec("_MethodParams")


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


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class About:
    point: Vec3T | None = None
    direction: Vec3T | None = None
    #mobject: "Mobject | None" = None

    def get_about_point(
        self,
        mobject: "Mobject"
    ) -> Vec3T | None:
        match self.point, self.direction:
            case np.ndarray() as point, None:
                return point
            case None, np.ndarray() as direction:
                return mobject.get_bounding_box_point(direction)
            case None, None:
                return None
            case _:
                raise ValueError("Can specify at most one of `point` and `direction` in `About` object")
        #if self.point is not None:
        #    return self.point
        #if self.direction is not None:
        #    #if self.mobject is not None:
        #    #    mobject = self.mobject
        #    return mobject.get_bounding_box_point(self.direction)
        #return None


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class Align:
    point: Vec3T | None = None
    mobject: "Mobject | None" = None
    border: Vec3T | None = None
    direction: Vec3T = ORIGIN
    buff: float = 0.0
    coor_mask: Vec3T | None = None

    def get_target_point(self) -> Vec3T:
        match self.point, self.mobject, self.border:
            case np.ndarray() as point, None, None:
                return point
            case None, Mobject() as mobject, None:
                return mobject.get_bounding_box_point(self.direction)
            case None, None, np.ndarray() as border:
                return border * np.append(ConfigSingleton().size.frame_size, 0.0) / 2.0
            case _:
                raise ValueError("Can specify exactly one of `point`, `mobject` and `edge` in `Align` object")

    def get_shift_vector(
        self,
        mobject: "Mobject",
        direction_sign: float
    ) -> Vec3T:
        target_point = self.get_target_point()
        direction = direction_sign * self.direction
        point_to_align = mobject.get_bounding_box_point(direction)
        vector = target_point - (point_to_align + self.buff * direction)
        if (coor_mask := self.coor_mask) is not None:
            vector *= coor_mask
        return vector


class MobjectMeta:
    __slots__ = ()

    _descriptor_partial_method: ClassVar[dict[LazyVariableDescriptor, Callable[[Any], Callable[[float, float], Any]]]] = {}
    _descriptor_interpolate_method: ClassVar[dict[LazyVariableDescriptor, Callable[[Any, Any], Callable[[float], Any]]]] = {}
    _descriptor_concatenate_method: ClassVar[dict[LazyVariableDescriptor, Callable[..., Callable[[], Any]]]] = {}
    _descriptor_related_styles: ClassVar[dict[LazyVariableDescriptor, tuple[tuple[LazyVariableDescriptor, Any], ...]]] = {}

    def __new__(cls):
        raise TypeError

    @classmethod
    def register(
        cls,
        partial_method: Callable[[_DescriptorRawT], Callable[[float, float], _DescriptorRawT]] | None = None,
        interpolate_method: Callable[[_DescriptorRawT, _DescriptorRawT], Callable[[float], _DescriptorRawT]] | None = None,
        concatenate_method: Callable[..., Callable[[], _DescriptorRawT]] | None = None,
        related_styles: tuple[tuple[LazyVariableDescriptor, Any], ...] = ()
    ):

        def callback(
            descriptor: LazyVariableDescriptor[_InstanceT, _ContainerT, _DescriptorRawT, _DescriptorGetT, _DescriptorSetT]
        ) -> LazyVariableDescriptor[_InstanceT, _ContainerT, _DescriptorRawT, _DescriptorGetT, _DescriptorSetT]:
            if partial_method is not None:
                cls._descriptor_partial_method[descriptor] = partial_method
            if interpolate_method is not None:
                cls._descriptor_interpolate_method[descriptor] = interpolate_method
            if concatenate_method is not None:
                cls._descriptor_concatenate_method[descriptor] = concatenate_method
            cls._descriptor_related_styles[descriptor] = related_styles
            return descriptor

        return callback

    @classmethod
    def _get_callback_from(
        cls,
        method_dict: dict[LazyVariableDescriptor, Callable[..., Callable[_MethodParams, Any]]],
        ignore_condition: Callable[..., bool] | None,
        srcs: "tuple[Mobject, ...]"
    ) -> "Callable[[Mobject], Callable[_MethodParams, None]]":

        def get_descriptor_callback(
            descriptor: LazyVariableDescriptor,
            method: Callable[..., Callable[_MethodParams, Any]],
            srcs: tuple[Mobject, ...]
        ) -> Callable[[Mobject], Callable[_MethodParams, None]] | None:
            src_args = tuple(
                descriptor.converter.convert_rget(descriptor.get_container(src))
                for src in srcs
            )
            if ignore_condition is not None and ignore_condition(*src_args):
                return None

            method_callback = method(*src_args)

            def descriptor_callback(
                dst: Mobject
            ) -> Callable[_MethodParams, None]:

                def descriptor_dst_callback(
                    *args: _MethodParams.args,
                    **kwargs: _MethodParams.kwargs
                ) -> None:
                    if descriptor not in type(dst)._lazy_variable_descriptors:
                        return
                    new_container = descriptor.converter.convert_set(method_callback(*args, **kwargs))
                    descriptor.set_container(dst, new_container)

                return descriptor_dst_callback

            return descriptor_callback

        descriptor_callbacks = [
            descriptor_callback
            for descriptor, method in method_dict.items()
            if all(
                descriptor in type(src)._lazy_variable_descriptors
                for src in srcs
            )
            and (descriptor_callback := get_descriptor_callback(descriptor, method, srcs)) is not None
        ]

        def callback(
            dst: Mobject
        ) -> Callable[_MethodParams, None]:
            descriptor_dst_callbacks = [
                descriptor_callback(dst)
                for descriptor_callback in descriptor_callbacks
            ]

            def dst_callback(
                *args: _MethodParams.args,
                **kwargs: _MethodParams.kwargs
            ) -> None:
                for descriptor_dst_callback in descriptor_dst_callbacks:
                    descriptor_dst_callback(*args, **kwargs)

            return dst_callback

        return callback

    @classmethod
    def _partial(
        cls,
        src: "Mobject"
    ) -> "Callable[[Mobject], Callable[[float, float], None]]":
        return cls._get_callback_from(
            method_dict=cls._descriptor_partial_method,
            ignore_condition=None,
            srcs=(src,)
        )

    @classmethod
    def _interpolate(
        cls,
        src_0: "Mobject",
        src_1: "Mobject"
    ) -> "Callable[[Mobject], Callable[[float], None]]":
        return cls._get_callback_from(
            method_dict=cls._descriptor_interpolate_method,
            ignore_condition=op.is_,
            srcs=(src_0, src_1)
        )

    @classmethod
    def _concatenate(
        cls,
        *srcs: "Mobject"
    ) -> "Callable[[Mobject], Callable[[], None]]":
        return cls._get_callback_from(
            method_dict=cls._descriptor_concatenate_method,
            ignore_condition=None,
            srcs=srcs
        )

    @classmethod
    def _set_style(
        cls,
        mobjects: "Iterable[Mobject]",
        style: dict[str, Any],
        handle_related_styles: bool
    ) -> None:
        values = {
            f"_{key}_": value if isinstance(value, LazyObject) else LazyWrapper(value)
            for key, value in style.items() if value is not None
        }
        for mobject in mobjects:
            for style_name, style_value in values.items():
                descriptor = type(mobject)._lazy_descriptor_dict.get(style_name)
                if descriptor is None:
                    continue
                if not isinstance(descriptor, LazyVariableDescriptor):
                    continue
                related_styles = cls._descriptor_related_styles.get(descriptor)
                if related_styles is None:
                    continue
                descriptor.__set__(mobject, style_value)
                if handle_related_styles:
                    for related_style_descriptor, related_style_value in related_styles:
                        if related_style_descriptor.method.__name__ not in values:
                            related_style_descriptor.__set__(mobject, related_style_value)


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

    # bounding box

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_mat4
    )
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

    # transform

    @classmethod
    def get_relative_transform_matrix(
        cls,
        matrix: Mat4T,
        about_point: Vec3T | None
    ) -> Mat4T:
        if about_point is None:
            return matrix
        return reduce(np.ndarray.__matmul__, (
            SpaceUtils.matrix_from_translation(about_point),
            matrix,
            SpaceUtils.matrix_from_translation(-about_point)
        ))

    #def get_about_point(
    #    self,
    #    about_point: Vec3T | None = None,
    #    about_edge: Vec3T | None = None,
    #    broadcast: bool = True
    #) -> Vec3T | None:
    #    if about_edge is not None:
    #        if about_point is not None:
    #            raise AttributeError("Cannot specify both parameters `about_point` and `about_edge`")
    #        about_point = self.get_bounding_box_point(about_edge, broadcast=broadcast)
    #    return about_point

    def apply_transform(
        self,
        matrix: Mat4T,
    ):
        # Avoid redundant caculations.
        transform_dict: dict[LazyWrapper[Mat4T], LazyWrapper[Mat4T]] = {}
        for mobject in self.iter_descendants():
            original_matrix = mobject._model_matrix_
            if (transformed_matrix := transform_dict.get(original_matrix)) is None:
                transformed_matrix = LazyWrapper(matrix @ original_matrix.value)
                transform_dict[original_matrix] = transformed_matrix
            mobject._model_matrix_ = matrix @ original_matrix.value
        return self

    def apply_relative_transform(
        self,
        matrix: Mat4T,
        about: About
    ):
        relative_matrix = self.get_relative_transform_matrix(
            matrix=matrix,
            about_point=about.get_about_point(mobject=self)
        )
        if np.isclose(np.linalg.det(relative_matrix), 0.0):
            warnings.warn("Applying a singular matrix transform")
        self.apply_transform(
            matrix=relative_matrix
        )
        return self

    # shift relatives

    def shift(
        self,
        vector: Vec3T
        #*,
        #coor_mask: Vec3T | None = None
    ):
        #if coor_mask is not None:
        #    vector *= coor_mask
        matrix = SpaceUtils.matrix_from_translation(vector)
        # `about` is meaningless for shifting.
        self.apply_relative_transform(
            matrix=matrix,
            about=About()
        )
        return self

    def move_to(
        self,
        #about: About,
        #buff: float = 0.0,
        #mobject_or_point: "Mobject | Vec3T",
        #aligned_edge: Vec3T = ORIGIN,
        align: Align
        #*,
        #coor_mask: Vec3T | None = None
    ):
        #if isinstance(mobject_or_point, Mobject):
        #    target_point = mobject_or_point.get_bounding_box_point(aligned_edge, broadcast=broadcast)
        #else:
        #    target_point = mobject_or_point
        #target_point = about.get_about_point(self)
        #assert target_point is not None
        #aligned_edge = edge if (edge := about.edge) is not None else ORIGIN
        #point_to_align = self.get_bounding_box_point(aligned_edge)
        #vector = target_point - point_to_align
        self.shift(
            vector=align.get_shift_vector(mobject=self, direction_sign=1.0)
        )
        return self

    def next_to(
        self,
        #mobject_or_point: "Mobject | Vec3T",
        #direction: Vec3T = RIGHT,
        #about: About,
        #buff: float = 0.0,
        align: Align
        #*,
        #coor_mask: Vec3T | None = None
        #broadcast: bool = True
    ):
        #if isinstance(mobject_or_point, Mobject):
        #    target_point = mobject_or_point.get_bounding_box_point(direction, broadcast=broadcast)
        #else:
        #    target_point = mobject_or_point
        #target_point = about.get_about_point(self)
        #assert target_point is not None
        #direction = edge if (edge := about.edge) is not None else ORIGIN
        #point_to_align = self.get_bounding_box_point(-direction)
        #vector = target_point - point_to_align + buff * direction
        self.shift(
            vector=align.get_shift_vector(mobject=self, direction_sign=-1.0)
        )
        return self

    #def center(
    #    self,
    #    *,
    #    coor_mask: Vec3T | None = None
    #):
    #    self.move_to(
    #        align=Align(point=ORIGIN)
    #    )
    #    return self

    # scale relatives

    def scale(
        self,
        factor: float | Vec3T,
        about: About = About()
    ):
        matrix = SpaceUtils.matrix_from_scale(factor)
        self.apply_relative_transform(
            matrix=matrix,
            about=about
        )
        return self

    def stretch_to_fit_size(
        self,
        target_size: Vec3T,
        about: About = About()
    ):
        factor_vector = target_size / self.get_bounding_box_size()
        self.scale(
            factor=factor_vector,
            about=about
        )
        return self

    #def stretch_to_fit_dim(
    #    self,
    #    target_length: float,
    #    dim: int,
    #    about: About = About()
    #):
    #    factor_vector = np.ones(3)
    #    factor_vector[dim] = target_length / self.get_bounding_box_size()[dim]
    #    self.scale(
    #        factor=factor_vector,
    #        about=about
    #    )
    #    return self

    #def stretch_to_fit_width(
    #    self,
    #    target_length: float,
    #    about: About = About()
    #):
    #    self.stretch_to_fit_dim(
    #        target_length=target_length,
    #        dim=0,
    #        about=about
    #    )
    #    return self

    #def stretch_to_fit_height(
    #    self,
    #    target_length: float,
    #    about: About = About()
    #):
    #    self.stretch_to_fit_dim(
    #        target_length=target_length,
    #        dim=1,
    #        about=about
    #    )
    #    return self

    #def stretch_to_fit_depth(
    #    self,
    #    target_length: float,
    #    about: About = About()
    #):
    #    self.stretch_to_fit_dim(
    #        target_length=target_length,
    #        dim=2,
    #        about=about
    #    )
    #    return self

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
        match specified_width, specified_height:
            case float(), float():
                return specified_width / original_width, specified_height / original_height
            case float(), None:
                scale_factor = specified_width / original_width
            case None, float():
                scale_factor = specified_height / original_height
            case None, None:
                scale_factor = specified_frame_scale if specified_frame_scale is not None else 1.0
            case _:
                raise ValueError  # never
        return scale_factor, scale_factor

    # rotate relatives

    def rotate(
        self,
        rotation: Rotation,
        about: About = About()
    ):
        matrix = SpaceUtils.matrix_from_rotation(rotation)
        self.apply_relative_transform(
            matrix=matrix,
            about=about
        )
        return self

    def flip(
        self,
        axis: Vec3T,
        about: About = About()
    ):
        self.rotate(
            rotation=Rotation.from_rotvec(SpaceUtils.normalize(axis) * PI),
            about=about
        )
        return self

    # meta methods

    def concatenate(self):
        MobjectMeta._concatenate(*self.iter_children())(self)()
        self.clear()
        return self

    def set_style(
        self,
        *,
        color: ColorT | None = None,
        opacity: float | None = None,

        # Mobject
        model_matrix: Mat4T | None = None,

        # RenderableMobject
        is_transparent: bool | None = None,

        # MeshMobject
        geometry: Geometry | None = None,
        color_map: moderngl.Texture | None = None,  # TODO: clashes with `None` itself.
        enable_phong_lighting: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,

        # ShapeMobject
        shape: Shape | None = None,

        # StrokeMobject
        multi_line_string: MultiLineString | None = None,
        width: float | None = None,
        single_sided: bool | None = None,
        has_linecap: bool | None = None,
        dilate: float | None = None,

        broadcast: bool = True,
        type_filter: "type[Mobject] | None" = None,
        handle_related_styles: bool = True
    ):
        color_component, opacity_component = ColorUtils.standardize_color_input(color, opacity)
        if type_filter is None:
            type_filter = Mobject
        MobjectMeta._set_style(
            self.iter_descendants_by_type(mobject_type=type_filter, broadcast=broadcast),
            {
                "color": color_component,
                "opacity": opacity_component,
                "model_matrix": model_matrix,
                "is_transparent": is_transparent,
                "geometry": geometry,
                "color_map": color_map,
                "enable_phong_lighting": enable_phong_lighting,
                "ambient_strength": ambient_strength,
                "specular_strength": specular_strength,
                "shininess": shininess,
                "shape": shape,
                "multi_line_string": multi_line_string,
                "width": width,
                "single_sided": single_sided,
                "has_linecap": has_linecap,
                "dilate": dilate
            },
            handle_related_styles=handle_related_styles
        )
        return self

    @property
    def model_matrix(self) -> Mat4T:
        return self._model_matrix_.value
