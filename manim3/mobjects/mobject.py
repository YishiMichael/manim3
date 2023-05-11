from dataclasses import dataclass
from functools import reduce
import itertools as it
from typing import (
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
    Mat4T,
    Vec3T,
    Vec3sT
)
from ..lazy.lazy import (
    Lazy,
    LazyObject,
    LazyWrapper
)
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import UniformBlockBuffer
from ..utils.space import SpaceUtils


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

    @Lazy.property_external
    @classmethod
    def _local_sample_points_(cls) -> Vec3sT:
        # Implemented in subclasses.
        return np.zeros((0, 3))

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
        if result is None:
            warnings.warn("Trying to calculate the bounding box of some mobject with no points")
            return BoundingBox(
                maximum=ORIGIN,
                minimum=ORIGIN
            )
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
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
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

    def rotate_about_origin(
        self,
        rotation: Rotation,
        *,
        broadcast: bool = True
    ):
        self.rotate(
            rotation=rotation,
            about_point=ORIGIN,
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

    # render

    @Lazy.variable_shared
    @classmethod
    def _is_transparent_(cls) -> bool:
        return False

    @Lazy.variable
    @classmethod
    def _camera_(cls) -> Camera:
        return Camera()

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        # Implemented in subclasses.
        pass
